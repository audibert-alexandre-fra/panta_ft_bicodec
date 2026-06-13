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
#SBATCH --account=oou@a100
#SBATCH --mail-type=BEGIN,FAIL,ABORT,END
#SBATCH --constraint=a100

module purge
module load arch/a100

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_NET_PLUGIN=none  # désactive le plugin réseau
export NCCL_SHM_DISABLE=1        # désactive shared memory NCCL
export NCCL_P2P_DISABLE=1        # désactive peer-to-peer entre GPU
export NCCL_DEBUG=WARN

source ../../../.venv/bin/activate
srun python -u training_ft_bicodec.py
