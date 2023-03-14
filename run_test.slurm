#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

module load gcc cuda cudnn mvapich2
source /ssoft/spack/external/tensorflow/tensorflow-1.3.0/tensorflow_venv_gpu/bi$
srun python test.py --dataset AffWild2 --arc swin_transformer_base --exp-name test_fold2  --resume results/swin_transformer_base_second_stage/bs_64_seed_0_lr_0.000001/xxxx_fold2.pth --fold 2
deactivate
