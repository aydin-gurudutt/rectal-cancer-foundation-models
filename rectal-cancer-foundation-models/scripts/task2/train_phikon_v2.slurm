#!/bin/bash
#SBATCH --array=0-9%2
#SBATCH --job-name=clam_phikon_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/task2_clam/logs/train_phikon_v2_%A_%a.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/task2_clam/logs/train_phikon_v2_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

BASE=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env

module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0

source "$BASE/venv_CLAM/bin/activate"

cd "$BASE/CLAM"

FEATURE_DIR="$BASE/outputs/paper1_coords_512/20.0x_512px_0px_overlap/features_phikon_v2"


CHECKPOINT="$BASE/outputs/task2_clam/results/phikon_v2_s42/s_${SLURM_ARRAY_TASK_ID}_checkpoint.pt"
if [[ -e "$CHECKPOINT" ]]; then
    echo "Existing checkpoint: $CHECKPOINT" >&2
    exit 1
fi

python main.py \
    --task task_3_pcr_vs_nonpcr \
    --csv_path "$BASE/CLAM/dataset_csv/task2_pcr.csv" \
    --data_root_dir "$FEATURE_DIR" \
    --feature_format h5_flat \
    --embed_dim 1024 \
    --split_dir task2_pcr_10fold \
    --results_dir "$BASE/outputs/task2_clam/results" \
    --exp_code phikon_v2 \
    --model_type clam_mb \
    --model_size small \
    --drop_out 0.25 \
    --lr 0.0002 \
    --max_epochs 200 \
    --k 10 \
    --k_start ${SLURM_ARRAY_TASK_ID} \
    --k_end $((SLURM_ARRAY_TASK_ID + 1)) \
    --seed 42 \
    --bag_loss ce \
    --inst_loss svm \
    --early_stopping \
    --weighted_sample

echo "phikon_v2 fold ${SLURM_ARRAY_TASK_ID} completed."
