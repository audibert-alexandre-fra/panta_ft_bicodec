#!/bin/bash
#SBATCH --job-name=ExtractTrain
#SBATCH --partition=archive         # partition qui peut accéder au STORE
#SBATCH --account=oou@cpu           # compte valide pour archive/cpu
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --output=mls.out

source ../../../.venv/bin/activate
python3 filter.py --data /lustre/fsn1/projects/rech/oou/uid39ok/mls_dataset/audio_processed --name mls_dataset
