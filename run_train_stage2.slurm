#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

python train_stage2.py --dataset AffWild2 --arc resnet50 --exp-name resnet50_second_stage  --resume results/resnet50_first_stage/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 
deactivate
