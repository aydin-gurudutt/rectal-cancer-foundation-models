#!/bin/bash
#SBATCH --job-name=task3_5models
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=asg162@case.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
cd /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env
module load Python/3.10.4-GCCcore-11.3.0
source venv_CLAM/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
STAGE="${1:-audit}"
echo "Job: ${SLURM_JOB_ID}, stage: ${STAGE}, node: $(hostname)"
python -u scripts/task3_five_model_compare.py --stage "$STAGE"
