#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

srun python test.py  --dataset RAFAU --resume /home/trentini/checkpoints/OpenGprahAU-ResNet50_first_stage.pth --draw_text
deactivate
