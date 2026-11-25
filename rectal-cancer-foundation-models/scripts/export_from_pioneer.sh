#!/usr/bin/env bash
# Run ONLY on Pioneer. Read-only with respect to research sources: copies selected files into a staging directory.
# No Git operations or model experiments. Review staged files before putting them in a repository.
set -euo pipefail
BASE=/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env
DEST="$BASE/github_handoff_staging"
mkdir -p "$DEST/src/task1" "$DEST/src/task2" "$DEST/src/task3" "$DEST/src/visualization" "$DEST/results/task2" "$DEST/results/task3"

copy_if_exists() {
  local in="$1" out="$2"
  if [[ -f "$in" ]]; then cp -p "$in" "$out"; echo "COPIED $in"; else echo "MISSING $in"; fi
}

# Original Task 1 script and wrapper paths must be confirmed; do not guess actual location.
for file in "$BASE"/slurm/Benchmarking_Feature_Extraction/*.slurm; do
  [[ -e "$file" ]] || continue
  copy_if_exists "$file" "$DEST/src/task1/$(basename "$file")"
done
for file in "$BASE"/slurm/Task2/*.py "$BASE"/slurm/Task2/*.slurm; do
  [[ -e "$file" ]] || continue
  copy_if_exists "$file" "$DEST/src/task2/$(basename "$file")"
done
for file in "$BASE"/scripts/task3_*.py "$BASE"/scripts/task3_*.slurm; do
  [[ -e "$file" ]] || continue
  copy_if_exists "$file" "$DEST/src/task3/$(basename "$file")"
done
for file in "$BASE"/task3_documentation/code/create_*.py; do
  [[ -e "$file" ]] || continue
  copy_if_exists "$file" "$DEST/src/visualization/$(basename "$file")"
done

# Export only aggregate non-patient-level metric files; never copy the OOF tables or row-level manifests.
copy_if_exists "$BASE/outputs/task2_clam/evaluation/five_model_comparison/five_model_summary.csv" "$DEST/results/task2/patient_level_model_summary.csv"
copy_if_exists "$BASE/outputs/task3_tissue/five_model_common_validation_summary.csv" "$DEST/results/task3/patch_validation_summary.csv"
copy_if_exists "$BASE/outputs/task3_tissue/conch_2000/five_model_common_validation_summary.csv" "$DEST/results/task3/conch_2000_summary.csv"
# JSON audit only summarizes sample counts but still requires disclosure review.
copy_if_exists "$BASE/outputs/task3_tissue/five_model_alignment_audit.json" "$DEST/results/task3/five_model_alignment_audit.json"

(cd "$DEST" && find . -type f ! -name 'SHA256SUMS.txt' -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
cat <<EOF
Staged files at: $DEST
STOP: inspect scripts, aggregate metrics, and paths for identifiers, credentials, usage rights and lab policy before transfer or Git commit.
DO NOT push any raw WSI, GeoJSON annotations, patient-level OOF prediction tables, checkpoint, H5 embeddings, or row-level manifests.
EOF
