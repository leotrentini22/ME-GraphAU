#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

srun python demo.py --arc resnet50 --exp-name demo --resume /home/trentini/checkpoints/OpenGprahAU-ResNet50_first_stage.pth --input demo_imgs/1014.jpg
deactivate
