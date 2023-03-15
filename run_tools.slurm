#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

cd tool/
srun python AffWild2_image_label_process.py
srun python AffWild2_calculate_AU_class_weights.py
srun python AffWild2_deal_AU_relation.py
deactivate
