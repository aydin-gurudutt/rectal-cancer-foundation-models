#!/bin/bash
#SBATCH --job-name=task3_conch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/task3_tissue/conch_slurm_%j.out
#SBATCH --error=outputs/task3_tissue/conch_slurm_%j.err

set -euo pipefail

cd /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env

module load Python/3.10.4-GCCcore-11.3.0
source venv_CLAM/bin/activate

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Starting CONCH linear-probe training"

python scripts/task3_train_conch.py

echo "Training completed"
