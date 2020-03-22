#!/bin/bash -l
#SBATCH -J InferSent_train
#SBATCH -o out_InferSent_train_%J.txt
#SBATCH -e err_InferSent_train_%J.txt
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --mem=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2001403
# run command

module purge
module load pytorch/1.0.1

srun python train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus snli \
  --encoder_type BiLSTMEncoder \
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
