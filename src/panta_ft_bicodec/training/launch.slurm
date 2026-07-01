#!/bin/bash
#SBATCH --job-name=TrainingBicodec
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=./logs_slurm/test_1.out
#SBATCH --error=./logs_slurm/test_2.err
#SBATCH --account=oou@h100
#SBATCH --mail-type=BEGIN,FAIL,ABORT,END
#SBATCH --constraint=h100

module purge
module load arch/a100

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_ALLOC_CONF=expandable_segments:True

source ../../../.venv/bin/activate
srun python -u training_ft_bicodec.py
