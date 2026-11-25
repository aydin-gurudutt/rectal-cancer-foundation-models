#!/bin/bash
#SBATCH --job-name=conch_eval_f1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/conch_eval_f1_%j.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/conch_eval_f1_%j.err

set -euo pipefail

BASE=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env

module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0
source "$BASE/venv_CLAM/bin/activate"

cd "$BASE/CLAM"

python "$BASE/slurm/Task2/evaluate_conch_fold0.py" --fold 1
