# GANs for gSCAN

This repository contains code for GAN training on the gSCAN dataset, and is based off of the code from the original [gSCAN directory](https://github.com/LauraRuis/groundedSCAN). 


## Getting Started

Make a virtualenvironment that uses Python 3.7 or higher:

```virtualenv --python=/usr/bin/python3.7 <path/to/virtualenv>```

Activate the environment and install the requirements with a package manager:

```{ source <path/to/virtualenv>/bin/activate; python3.7 -m pip install -r requirements; }```

Note that this repository depends on the grounded SCAN implementation to load the dataset from a dataset.txt with the function `GroundedScan.load_dataset_from_file()`.
Before actually training models, unzip the data you want to use from [this repo](https://github.com/groundedSCAN/gSCAN_data) and put it in a folder `data`.

### Training

To train a model on a grounded SCAN dataset with a simple situation representation, run:

    python3.7 -m seq2seq --mode=train --data_directory=<path/to/folder/with/dataset.txt/> --output_directory=<path/where/models/will/be/saved> --attention_type=bahdanau --max_training_iterations=200000

This will train a model and save the results in `output_directory`.

### Testing

To test with a model and obtain predictions, run the following command.

    python3.7 -m seq2seq --mode=test --data_directory=<path/to/folder/with/dataset.txt/> --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=<path/containing/trained/models> --resume_from_file=adverb_k_1_run_3/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=predict.json --max_decoding_steps=120

NB: the output .json file generated by this can be passed to the error analysis or execute commands mode in the dataset generation repo ([found here](https://github.com/groundedSCAN/groundedSCAN)). The repository at that link also contains a file `example_prediction.json` with 1 data example prediction as generated with the test mode of this repository.

