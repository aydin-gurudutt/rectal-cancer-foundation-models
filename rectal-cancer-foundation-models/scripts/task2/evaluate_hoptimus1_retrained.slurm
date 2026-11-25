#!/bin/bash
#SBATCH --job-name=eval_hoptimus1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/task2_clam/logs/eval_hoptimus1_%A_%a.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/task2_clam/logs/eval_hoptimus1_%A_%a.err

set -euo pipefail

BASE=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env

module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0
source "$BASE/venv_CLAM/bin/activate"

cd "$BASE/CLAM"

python "$BASE/slurm/Task2/evaluate_hoptimus1_retrained.py" \
    --fold "$SLURM_ARRAY_TASK_ID"
