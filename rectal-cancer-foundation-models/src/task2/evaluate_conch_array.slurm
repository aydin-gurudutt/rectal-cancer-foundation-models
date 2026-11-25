#!/bin/bash
#SBATCH --job-name=conch_eval_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=2-9%2
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/conch_eval_%A_%a.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/conch_eval_%A_%a.err

set -euo pipefail

BASE=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env

module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0
source "$BASE/venv_CLAM/bin/activate"

cd "$BASE/CLAM"

FOLD="$SLURM_ARRAY_TASK_ID"
python "$BASE/slurm/Task2/evaluate_conch_fold0.py" --fold "$FOLD"
