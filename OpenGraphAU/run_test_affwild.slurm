#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

srun python test.py  --resume results/OpenGprahAU-ResNet50_first_stage/bs_64_seed_0_lr_2e-05/best_model.pth --draw_text
deactivate
