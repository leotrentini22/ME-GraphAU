#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

python train_stage1.py --dataset AffWild2 --arc resnet50 --exp-name resnet50_first_stage -b 64 -lr 0.0001 --fold 1 
deactivate
