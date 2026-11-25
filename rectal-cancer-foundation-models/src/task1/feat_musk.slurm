#!/bin/bash
#SBATCH --job-name=muskfeat
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/muskfeat_%j.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/muskfeat_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6-02:00:00
#SBATCH --mail-user=asg162@case.edu
#SBATCH --mail-type=END,FAIL

unset PYTHONPATH
module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0
source /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/venv_trident/bin/activate
cd /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/trident

export WSI_DIR=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/SlidesToUse
export JOB_DIR=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/paper1_coords_512

python run_batch_of_slides.py \
  --task feat \
  --wsi_dir "$WSI_DIR" \
  --job_dir "$JOB_DIR" \
  --patch_encoder musk \
  --mag 20 \
  --patch_size 512