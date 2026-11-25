#!/bin/bash
#SBATCH --job-name=segcoords512
#SBATCH --output=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/segcoords512_%j.out
#SBATCH --error=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/logs/segcoords512_%j.err
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

# --- Real slides live nested: resection_wsis/<patient_id>/<slide>.svs ---
export SOURCE_DIR=/mnt/vstor/CSE_BME_CCIPD/home/sxv274/rectal_cancer_pathology/UH/2024_Dataset/resection_wsis
# --- Flat staging dir TRIDENT will actually read from ---
export WSI_DIR=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/SlidesToUse
export JOB_DIR=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/paper1_coords_512
# --- List of slide filenames that belong in this paper's cohort ---
export SLIDE_LIST=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/slurm/Benchmarking_Feature_Extraction/paper1_slide_list.txt

mkdir -p "$WSI_DIR"

echo "Building FILTERED flat symlink directory from nested source..."
matched=0
missing=0
while IFS= read -r wanted_name || [ -n "$wanted_name" ]; do
    [ -z "$wanted_name" ] && continue

    # case-insensitive search for this exact filename anywhere under SOURCE_DIR
    src=$(find "$SOURCE_DIR" -type f -iname "$wanted_name" | head -n 1)

    if [ -z "$src" ]; then
        echo "MISSING: '$wanted_name' not found under $SOURCE_DIR"
        missing=$((missing+1))
        continue
    fi

    link="$WSI_DIR/$(basename "$src")"
    if [ -e "$link" ]; then
        continue
    fi
    ln -s "$src" "$link"
    matched=$((matched+1))
done < "$SLIDE_LIST"

echo "Matched and linked: $matched"
echo "Missing (in list but not found on disk): $missing"
echo "Flat WSI dir now has: $(find "$WSI_DIR" -maxdepth 1 -name '*.svs' | wc -l) slides"

# --- Segmentation ---
python run_batch_of_slides.py \
  --task seg \
  --wsi_dir "$WSI_DIR" \
  --job_dir "$JOB_DIR" \
  --gpu 0 \
  --segmenter hest

if [ $? -ne 0 ]; then
    echo "ERROR: segmentation failed"
    exit 1
fi

# --- Patch coordinate extraction ---
python run_batch_of_slides.py \
  --task coords \
  --wsi_dir "$WSI_DIR" \
  --job_dir "$JOB_DIR" \
  --mag 20 \
  --patch_size 512 \
  --overlap 0

if [ $? -ne 0 ]; then
    echo "ERROR: coordinate extraction failed"
    exit 1
fi

echo "Segmentation + coords complete."