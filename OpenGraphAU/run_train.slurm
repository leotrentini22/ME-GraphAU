#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=30:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

srun python train_stage1.py --arc resnet50 --resume results/OpenGprahAU-ResNet50_first_stage/bs_64_seed_0_lr_2e-05/best_model.pth --exp-name OpenGprahAU-ResNet50_first_stage -b 64 -lr 0.00002  