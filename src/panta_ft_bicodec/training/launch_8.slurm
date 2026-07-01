#!/bin/bash
#SBATCH --job-name=TrainingBicodec
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./logs_slurm/run_8.out
#SBATCH --error=./logs_slurm/run_8.err
#SBATCH --account=oou@h100
#SBATCH --mail-type=BEGIN,FAIL,ABORT,END
#SBATCH --constraint=h100

module purge
module load arch/a100

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_ALLOC_CONF=expandable_segments:True

source ../../../.venv/bin/activate
srun python -u training_ft_bicodec.py --config config_8.yaml
