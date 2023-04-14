#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=4:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

srun python train_stage1.py --arc resnet50 --exp-name OpenGprahAU-ResNet50_first_stage -b 64 -lr 0.00002  