#!/bin/bash
#SBATCH --job-name=mlm
#SBATCH --output=/beegfs/yp913/GroundedScan/test.out
#SBATCH --time=3:40:00
#SBATCH --mem=45GB
#SBATCH --mail-user=yp913@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
source activate /beegfs/yp913/env37 
MODEL_PATH=$2
SPLIT=$1
echo "$MODEL_PATH"
python3.7 -m seq2seq --mode=test --data_directory=gSCAN_data/data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=outputs_pretrain  --output_file_name=predict.json --max_decoding_steps=120 --splits=$SPLIT  --resume_from_file=$MODEL_PATH
