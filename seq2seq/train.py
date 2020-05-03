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

import pickle
def pretrain_discriminator(training_set, use_cuda, max_decoding_steps):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    total_vocabulary = set(list(training_set.input_vocabulary._word_to_idx.keys()) + list(training_set.target_vocabulary._word_to_idx.keys()))
    total_vocabulary_size = len(total_vocabulary)
    discriminator = Discriminator(300,  512, total_vocabulary_size, max_decoding_steps)
    batch_size = 16
    pretraining_dataset = pickle.load(open("new_dataset","rb"))
    val_pretraining_dataset = pickle.load(open("val_dataset", "rb"))
    _, val_inp, val_target = zip(*val_pretraining_dataset)
    epochs = 10
    dis_opt = optim.Adagrad(dis.parameters())
    d_step = 0
    for epoch in range(epochs):
        print('d-step epoch %d : ' % (epoch + 1), end='')
        total_loss = 0
        total_acc = 0
        for i in range(len(pretraining_dataset), batch_size):
            batch = pretraining_dataset[i: i + batch_size]
            input_command, target_command, label = zip(*batch.unzip)
            dis_opt.zero_grad()
            out = discriminator.batchClassify(target_command)
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, label)
            loss.backward()
            dis_opt.step()

            total_loss += loss.data.item()
            total_acc += torch.sum((out > 0.5) == (label  > 0.5)).data.item()

        total_loss /= (len(pretraining_dataset)/ batch_size)
        total_acc /= len(pretraining_dataset)

        val_pred = discriminator.batchClassify(val_inp)
        print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
            total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.))

    # Then we save the model weights
    torch.save(discriminator, open("trained_discriminator_weights"))


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, attention_type: str, k: int, max_training_examples=None, seed=42, **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()
    print(cfg)
    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies, k=k)
    training_set.read_dataset(max_examples=10,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    ##::test_set = GroundedScanDataset(data_path, data_directory, split="dev",
     #                              input_vocabulary_file=input_vocab_path,
      #                             target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0)
    #test_set.read_dataset(max_examples=10,
    #                      simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    #test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    model = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size,
                  num_cnn_channels=training_set.image_channels,
                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                  target_pad_idx=training_set.target_vocabulary.pad_idx,
                  target_eos_idx=training_set.target_vocabulary.eos_idx,
                  **cfg)
    total_vocabulary = set(list(training_set.input_vocabulary._word_to_idx.keys()) + list(training_set.target_vocabulary._word_to_idx.keys()))
    total_vocabulary_size = len(total_vocabulary)
    discriminator = Discriminator(300,  512, total_vocabulary_size, max_decoding_steps)

    model = model.cuda() if use_cuda else model
    discriminator = discriminator.cuda() if use_cuda else discriminator
    reward_func = lambda x: discriminator(x)
    rollout = Rollout(model, 0.8)
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    new_dataset = []
    # all_val_examples
    # PRETRAIN DISCRIMINATOR
   
    dis_opt = optim.Adagrad(discriminator.parameters())
    total_loss = 0
    total_acc = 0
    i = 0
    for epoch in range(10):
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
            batch_size=training_batch_size):
            i += 1
            target_scores, target_position_scores = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                          situations_input=situation_batch, target_batch=target_batch,
                                                          target_lengths=target_lengths)
            # and then we sample from the generator
            target_scores = F.log_softmax(target_scores, dim=-1).max(dim=-1)[1].detach()[:,1:]
            num = 16
            commands_input = input_batch
            commands_lengths = input_lengths
            situations_input = situation_batch
            sos_idx = training_set.input_vocabulary.sos_idx
            eos_idx = training_set.input_vocabulary.eos_idx
            data = target_scores
            neg_sample = model.sample(data, commands_input, commands_lengths,
                                      situations_input, target_batch, sos_idx,
                                      eos_idx, n=num)
            target_batch = target_batch.numpy().tolist()
            label = [[1] * len(target_batch)] + [[0] * len(neg_sample)]
            label = [x for y in label for x in y]
            target_batch.extend(neg_sample)
            # Now, let's randomly shuffle these up.
            exs = list(zip(target_batch, label))
            import random
            random.shuffle(exs)
            dis_opt.zero_grad()
            target_batch, label = zip(*exs)
            target_batch = torch.Tensor(target_batch)
            label = torch.Tensor(label)
            out = discriminator.batchClassify(target_batch) # POSITIVES FIRST
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, label)
            loss.backward()
            dis_opt.step()
    
            total_loss += loss.data.item()
            total_acc += torch.sum((out > 0.5) == (label > 0.5)).data.item()
            if i % 500 == 0: # eVERY 500 STEPS, PRINT out.
                print(' average_loss = %.4f, train_acc = %.4f' % (
                    total_loss, total_acc))
    torch.save(discriminator.state_dict(), "pretrained_discriminator.ckpt")
 
    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()

            # Forward pass.
            # * probabilities over target vocabulary outputted by the model
            target_scores, target_position_scores = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                          situations_input=situation_batch, target_batch=target_batch,
                                                          target_lengths=target_lengths)
            import pdb; pdb.set_trace()
            reward = rollout.get_reward(target_scores, 16, input_batch, input_lengths, situation_batch, target_batch, training_set.input_vocabulary.sos_idx, training_set.input_vocabulary.eos_idx, reward_func)
            
            loss = model.get_loss(target_scores, target_batch)
            if auxiliary_task:
                target_loss = model.get_auxiliary_loss(target_position_scores, target_positions)
            else:
                target_loss = 0
            loss += weight_target_loss * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model.get_metrics(target_scores, target_batch)
                if auxiliary_task:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")
                    accuracy, exact_match, target_accuracy = evaluate(
                        test_set.get_data_iterator(batch_size=1), model=model,
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
                        model.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")
