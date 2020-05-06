# Neural Baseline and GECA for Grounded SCAN

This repository contains a multi-modal neural sequence-to-sequence model with a CNN to parse a world state and joint attention over input instruction sequences and world states.
This model is detailed in the grounded SCAN paper, and graphically depicted in the image in ![model_image](https://raw.githubusercontent.com/groundedSCAN/multimodal_neural_gsCAN/master/documentation/model_bahdanau.png)

Go to __main__.py folder and run this command 

```python3 __main__.py --mode=train --data_directory="../demo_dataset" --output_directory='out' --attention_type=bahdanau --max_training_iterations=20000```

## TL;DR

Find all commands to reproduce the experiments in `all_experiments.sh` containing the used parameters and seeds.
For a detailed file with all parameters, seeds, and other logging per training run, see `documentation/training_logs/`

## Getting Started

Make a virtualenvironment that uses Python 3.7 or higher:

```virtualenv --python=/usr/bin/python3.7 <path/to/virtualenv>```

Activate the environment and install the requirements with a package manager:

```{ source <path/to/virtualenv>/bin/activate; python3.7 -m pip install -r requirements; }```

Note that this repository depends on the grounded SCAN implementation to load the dataset from a dataset.txt with the function `GroundedScan.load_dataset_from_file()`.
Before actually training models, unzip the data you want to use from [this repo](https://github.com/groundedSCAN/gSCAN_data) and put it in a folder `data`.

### Alternative way of loading data
In the folder `read_gscan` there is a separate `README.md` and code to read about how to prepare the data for a computational model in a way independent from the code in `GroundedScan`.


## Contents

Sequence to sequence models for Grounded SCAN.

### Training

To train a model on a grounded SCAN dataset with a simple situation representation, run:

    python3.7 -m seq2seq --mode=train --data_directory=<path/to/folder/with/dataset.txt/> --output_directory=<path/where/models/will/be/saved> --attention_type=bahdanau --max_training_iterations=200000

This will train a model and save the results in `output_directory`.

### Testing

To test with a model and obtain predictions, run the following command.

    python3.7 -m seq2seq --mode=test --data_directory=<path/to/folder/with/dataset.txt/> --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=<path/containing/trained/models> --resume_from_file=adverb_k_1_run_3/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=predict.json --max_decoding_steps=120

NB: the output .json file generated by this can be passed to the error analysis or execute commands mode in the dataset generation repo ([found here](https://github.com/groundedSCAN/groundedSCAN)). The repository at that link also contains a file `example_prediction.json` with 1 data example prediction as generated with the test mode of this repository.

## Important arguments

- `max_decoding_steps`: reflect max target length in data
- `k`: how many examples to add to the training set that contain 'cautiously' as an adverb
- `max_testing_examples`: testing is slow, because it is only implemented for batch size 1, so to speed up training use a small amount of testing examples for evaluation on the development set during training.


## Reproducing experiments from grounded SCAN paper

See file `all_experiments.sh` for all commands that were run to train the models for the grounded SCAN paper, as well as the commands to test trained models.


## Reproducability: Hyperparameters and Seeds

For all training logs of all models trained in the paper see `documentation/training_logs/..`.
These fills contain printed all hyperparameters, as well as the training and development performance over time and the seeds used in training.
