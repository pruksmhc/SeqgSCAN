import logging
import torch
import os
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from seq2seq.model import Model
from seq2seq.rollout import Rollout
from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.helpers import log_parameters
from seq2seq.evaluate import evaluate
from seq2seq.discriminator import Discriminator

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False
import torch.nn as nn
import torch.optim as optim
import math
import pickle
from tqdm import tqdm
import random


def train_discriminator(training_set, discriminator, training_batch_size, generator, seed, epochs, name):
    random.seed(seed)
    loss_fn = nn.BCELoss()
    # PRE-TRAIN DISCRIMINATOR
    dis_opt = optim.Adam(discriminator.parameters())
    for _ in tqdm(range(epochs)):
        total_loss = 0
        total_acc = 0
        i = 0
        num_examples_seen = 0
        for (input_batch, input_lengths, _, situation_batch, _, positive_samples,
             target_lengths, agent_positions, target_positions) in \
                tqdm(training_set.get_data_iterator(batch_size=training_batch_size)):
            i += 1
            # and then we sample from the generator
            neg_samples = generator.sample(batch_size=len(input_batch),
                                           max_seq_len=max(target_lengths).astype(int),
                                           commands_input=input_batch, commands_lengths=input_lengths,
                                           target_batch=positive_samples,
                                           situations_input=situation_batch,
                                           sos_idx=training_set.input_vocabulary.sos_idx,
                                           eos_idx=training_set.input_vocabulary.eos_idx
                                           )

            positive_samples = positive_samples.cpu().numpy().tolist()
            neg_samples = neg_samples.cpu().numpy().tolist()
            labels = [[1] * len(positive_samples)] + [[0] * len(neg_samples)]
            labels = [x for y in labels for x in y]
            target_batch = positive_samples + neg_samples
            num_examples_seen += len(target_batch)
            # Now, let's randomly shuffle these up.
            exs = list(zip(target_batch, labels))
            random.shuffle(exs)
            target_batch, labels = zip(*exs)
            target_batch = torch.Tensor(target_batch)
            if use_cuda:
                labels = torch.Tensor(labels).cuda()
            else:
                labels = torch.Tensor(labels)
            out = discriminator.batchClassify(target_batch.long())
            del target_batch
            loss = loss_fn(out, labels)
            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()

            total_loss += loss.data.item()
            del loss

            total_acc += torch.sum((out > 0.5) == (labels > 0.5)).data.item()
            if i % 500 == 0:  # we print statistics every 500 steps
                print_loss = float(total_loss) / float(i)  # divide by how many updates there were
                print_acc = total_acc / float(num_examples_seen)
                print('average_loss = %.4f, train_acc = %.4f' % (print_loss, print_acc))
                torch.save(discriminator.state_dict(), "%s.ckpt" % name)

        training_set_size = len(training_set._examples)
        total_loss /= math.ceil(2 * training_set_size / float(training_batch_size))
        total_acc /= float(num_examples_seen)
        if total_acc > 1.0:
            import pdb
            pdb.set_trace()
        print('average_loss = %.4f, train_acc = %.4f' % (total_loss, total_acc))
        torch.save(discriminator.state_dict(), "{}.ckpt".format(name))
        del total_loss, total_acc


def pre_train_generator(training_set, training_batch_size, generator, seed, epochs, name):
    random.seed(seed)
    # gen_opt = optim.Adam(generator.parameters(), lr=0.0005, eps=1e-8)
    gen_opt = optim.Adam(generator.parameters(), lr=0.0005)
    warmup_steps = epochs // 10
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     gen_opt, gamma=0.1, step_size=1
    # )
    loss_fn = nn.NLLLoss(reduction='sum')
    for _ in tqdm(range(epochs)):
        # scheduler.step()
        total_loss = 0
        total_words = 0
        i = 0
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in \
                tqdm(training_set.get_data_iterator(batch_size=training_batch_size)):
            i += 1
            if use_cuda:
                input_batch, target_batch = input_batch.cuda(), target_batch.cuda()

            samples = generator.sample(batch_size=len(input_batch),
                                       max_seq_len=max(target_lengths).astype(int),
                                       commands_input=input_batch, commands_lengths=input_lengths,
                                       situations_input=situation_batch,
                                       target_batch=target_batch,
                                       sos_idx=training_set.input_vocabulary.sos_idx,
                                       eos_idx=training_set.input_vocabulary.eos_idx)

            pred = generator.get_normalized_logits(commands_input=input_batch,
                                                   commands_lengths=input_lengths,
                                                   situations_input=situation_batch,
                                                   samples=samples,
                                                   sample_lengths=target_lengths,
                                                   sos_idx=training_set.input_vocabulary.sos_idx)
            target = target_batch.contiguous().view(-1)

            loss = loss_fn(pred, target)
            del target, samples
            total_loss += loss.item()
            total_words += pred.size(0) * pred.size(1)
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()
            del loss
        print('Pretraining Gen Loss = {}'.format(math.exp(total_loss / total_words)))
        torch.save(generator.state_dict(), "{}.ckpt".format(name))


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, attention_type: str, k: int,
          max_training_examples, max_testing_examples,
          # SeqGAN params begin
          pretrain_gen_path, pretrain_gen_epochs,
          pretrain_disc_path, pretrain_disc_epochs,
          rollout_trails, rollout_update_rate,
          disc_emb_dim, disc_hid_dim,
          load_tensors_from_path,
          # SeqGAN params end
          seed=42,
          **kwargs):
    device = torch.device("cpu")
    cfg = locals().copy()
    torch.manual_seed(seed)

    logger.info("Loading Training set...")

    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies, k=k)
    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation,
                              load_tensors_from_path=load_tensors_from_path)  # set this to False if no pickle file available

    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    # logger.info("Loading Dev. set...")
    # test_set = GroundedScanDataset(data_path, data_directory, split="dev",
    #                                input_vocabulary_file=input_vocab_path,
    #                                target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0)
    # test_set.read_dataset(max_examples=max_testing_examples,
    #                       simple_situation_representation=simple_situation_representation)
    #
    # # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    # test_set.shuffle_data()

    # logger.info("Done Loading Dev. set.")

    generator = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                      target_vocabulary_size=training_set.target_vocabulary_size,
                      num_cnn_channels=training_set.image_channels,
                      input_padding_idx=training_set.input_vocabulary.pad_idx,
                      target_pad_idx=training_set.target_vocabulary.pad_idx,
                      target_eos_idx=training_set.target_vocabulary.eos_idx,
                      **cfg)
    total_vocabulary = set(list(training_set.input_vocabulary._word_to_idx.keys()) + list(
        training_set.target_vocabulary._word_to_idx.keys()))
    total_vocabulary_size = len(total_vocabulary)
    discriminator = Discriminator(embedding_dim=disc_emb_dim, hidden_dim=disc_hid_dim,
                                  vocab_size=total_vocabulary_size, max_seq_len=max_decoding_steps)

    generator = generator.cuda() if use_cuda else generator
    discriminator = discriminator.cuda() if use_cuda else discriminator
    rollout = Rollout(generator, rollout_update_rate)
    log_parameters(generator)
    trainable_parameters = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = generator.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = generator.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    if pretrain_gen_path is None:
        print('Pretraining generator with MLE...')
        pre_train_generator(training_set, training_batch_size, generator, seed, pretrain_gen_epochs,
                            name='pretrained_generator')
    else:
        print('Load pretrained generator weights')
        generator_weights = torch.load(pretrain_gen_path)
        generator.load_state_dict(generator_weights)

    if pretrain_disc_path is None:
        print('Pretraining Discriminator....')
        train_discriminator(training_set, discriminator, training_batch_size, generator, seed, pretrain_disc_epochs,
                            name="pretrained_discriminator")
    else:
        print('Loading Discriminator....')
        discriminator_weights = torch.load(pretrain_disc_path)
        discriminator.load_state_dict(discriminator_weights)

    logger.info("Training starts..")
    training_iteration = start_iteration
    torch.autograd.set_detect_anomaly(True)
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()

        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in \
                training_set.get_data_iterator(batch_size=training_batch_size):
            is_best = False
            generator.train()
            # Forward pass.
            samples = generator.sample(batch_size=len(input_batch),
                                       max_seq_len=max(target_lengths).astype(int),
                                       commands_input=input_batch, commands_lengths=input_lengths,
                                       situations_input=situation_batch,
                                       target_batch=target_batch,
                                       sos_idx=training_set.input_vocabulary.sos_idx,
                                       eos_idx=training_set.input_vocabulary.eos_idx)

            rewards = rollout.get_reward(samples,
                                         rollout_trails,
                                         input_batch,
                                         input_lengths,
                                         situation_batch,
                                         target_batch,
                                         training_set.input_vocabulary.sos_idx,
                                         training_set.input_vocabulary.eos_idx,
                                         discriminator)

            assert samples.shape == rewards.shape

            # calculate rewards
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if use_cuda:
                rewards = rewards.cuda()

            # get generator scores for sequence
            target_scores = generator.get_normalized_logits(commands_input=input_batch,
                                                            commands_lengths=input_lengths,
                                                            situations_input=situation_batch,
                                                            samples=samples,
                                                            sample_lengths=target_lengths,
                                                            sos_idx=training_set.input_vocabulary.sos_idx)

            del samples
            # calculate loss on the generated sequence given the rewards
            loss = generator.get_gan_loss(target_scores, target_batch, rewards)

            del rewards

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step(training_iteration)
            optimizer.zero_grad()
            generator.update_state(is_best=is_best)
            # Print current metrics.
            if training_iteration % print_every == 0:
                target_scores = target_scores.view(target_batch.shape[0], target_batch.shape[1], -1)
                accuracy, exact_match = generator.get_metrics(target_scores, target_batch)
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            % (training_iteration, loss, accuracy, exact_match, learning_rate))

            del target_scores, target_batch

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    generator.eval()
                    logger.info("Evaluating..")
                    accuracy, exact_match, target_accuracy = evaluate(
                        test_set.get_data_iterator(batch_size=1), model=generator,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"])
                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        generator.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        generator.save_checkpoint(file_name=file_name, is_best=is_best,
                                                  optimizer_state_dict=optimizer.state_dict())

            rollout.update_params()

            train_discriminator(training_set, discriminator, training_batch_size, generator, seed, epochs=1,
                                name="training_discriminator")
            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
            del loss

        torch.save(generator.state_dict(), os.path.join(output_directory, 'gen_{}_{}.ckpt'.format(training_iteration, seed)))
        torch.save(discriminator.state_dict(), os.path.join(output_directory, 'dis_{}_{}.ckpt'.format(training_iteration, seed)))

    logger.info("Finished training.")
