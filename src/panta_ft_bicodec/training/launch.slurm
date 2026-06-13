#!/bin/bash
#SBATCH --job-name=TrainingBicodec
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=./logs_slurm/first.out
#SBATCH --error=./logs_slurm/first.err
#SBATCH --account=fev@a100
#SBATCH --mail-type=BEGIN,FAIL,ABORT,END
#SBATCH --constraint=a100

module purge
module load arch/a100

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export PYTORCH_ALLOC_CONF=expandable_segments:True

source ../../../.venv/bin/activate
srun python -u training_ft_bicodec.py
