#!/bin/bash
#SBATCH --job-name=task1_dimred
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/task1_dimred_%j.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/task1_dimred_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --mail-user=asg162@case.edu
#SBATCH --mail-type=END,FAIL

unset PYTHONPATH
module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0
source /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/venv_trident/bin/activate

# one-time deps this venv likely doesn't have yet -- safe to leave in,
# pip skips anything already satisfied
pip install --quiet umap-learn scikit-learn matplotlib pandas h5py

cd /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/slurm/Benchmarking_Feature_Extraction

python task1_dimensionality_reduction.py \
  --job_dir /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/paper1_coords_512/20.0x_512px_0px_overlap \
  --metadata_csv /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/slurm/Benchmarking_Feature_Extraction/slide_metadata.csv \
  --out_dir /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/task1_results \
  --encoders conch_v15 virchow2 phikon_v2 hoptimus1 musk

echo "Task 1 dimensionality reduction complete."
