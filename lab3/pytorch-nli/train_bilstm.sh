#!/bin/bash -l
#SBATCH -J SNLI_InferSent_4096D
#SBATCH -o out_InferSent_4096D_%J.txt
#SBATCH -e err_InferSent_4096D_%J.txt
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
# run command
module purge
module load gcc cuda pytorch
module list

srun python3 train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus snli \
  --encoder_type BiLSTMMaxPoolEncoder \
  --activation tanh \
  --optimizer sgd \
  --word_embedding glove.840B.300d \
  --embed_dim 300 \
  --fc_dim 512 \
  --hidden_dim 2048 \
  --layers 1 \
  --dropout 0 \
  --learning_rate 0.1 \
  --lr_patience 1 \
  --lr_decay 0.99 \
  --lr_reduction_factor 0.2 \
  --save_path results \
  --seed 1234

# This script will print some usage statistics to the
# end of file: output.txt
# Use that to improve your resource request estimate
# on later jobs.
used_slurm_resources.bash
