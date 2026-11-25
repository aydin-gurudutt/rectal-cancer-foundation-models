# Rectal cancer pathology: foundation model benchmarking (Tasks 1–3)

**Research archive / documentation repository — preliminary results.**  This repository organizes code and aggregate results for image preprocessing and foundation-model feature extraction (Task 1a), exploratory embedding analysis, patient-level CLAM response prediction (Task 2), and SRA-guided patch tissue classification (Task 3).

> **Status:** The original custom Task 1–3 scripts and SLURM wrappers have been imported from the Pioneer export into `src/`. This does **not** include the complete CLAM/TRIDENT implementation, datasets, checkpoints, or a verified environment lockfile. The supervisor has paused additional foundation-model experiments. Lab authorization is required before sharing this repository.

## Research tasks and code locations

| Workstream | Input / method | Recovered code / expected location | Outputs |
| --- | --- | --- | --- |
| Task 1a | WSIs → TRIDENT HEST tissue masks → 20×, 512 px, zero-overlap coordinates → embeddings for five encoders | `src/task1/` **original exported wrappers and analysis script included** | H5 features (excluded from Git) |
| Exploratory Task 1 | Mean-pooled embeddings, PCA/UMAP, grouped kNN and silhouette | `src/task1/` **original exported wrappers and analysis script included** | PCA/UMAP and summary JSON (export pending) |
| Task 1b | Model architecture and dataset literature comparison | `docs/` — no verified complete literature matrix | To be supplied |
| Task 2 | 223 slides / 68 patients; shared ten patient-grouped folds; CLAM-MB | `src/task2/` **original exported CLAM wrappers/evaluators included** | [`results/task2/`](results/task2/) |
| Task 3 | SRA tile spatial alignment, provisional 3-class linear probes, patient-grouped partitions, manual overlay | `src/task3/` **original exported tissue-analysis scripts included**; recovered visualization scripts in `src/visualization/` | [`results/task3/`](results/task3/) |

Five encoders: CONCH v1.5, Virchow2, Phikon v2, H-Optimus-1, MUSK. These are encoder-plus-classifier comparisons, not head-to-head comparisons of identical native patch sizes.

## Documented results

### Task 2: pCR classification

Ten-fold grouped-CV patient-level summary is [`results/task2/patient_level_model_summary.csv`](results/task2/patient_level_model_summary.csv). The highest *pooled out-of-fold* patient AUC was **MUSK 0.8479** followed by **CONCH v1.5 0.8196**. These comparisons are internal and do not establish statistical superiority or external clinical performance. Individual patients may have multiple slides. Slide-wise predictions are aggregated to one patient prediction; **the exact aggregation function is still to be checked in the original evaluator**.

### Task 3: provisional SRA tissue categories

The previous manifest comprised **852,333 patches**, 179 matched slides and 56 provisional patient groups; after patient metadata verification, the partition contains 846,756 patches, 178 slides and 55 patients (38 train, 8 validation, 9 held-out test patients). The sampled balanced linear-probe comparison used 18,000 training and 6,000 validation patches, matched across all five models by slide ID and level-0 `(x,y)`. **MUSK** ranked highest on provisional SRA validation with **0.9208 balanced accuracy / 0.9204 macro-F1**. The CONCH 2,000-max-iteration check produced **0.904833 balanced accuracy / 0.904946 macro-F1** with no warning shown in its log. The nine-patient test set was not evaluated in the documented comparison.

Label classes `TUM`, `MUS`, and `STR` are SRA-derived **pseudo-labels**, not pathologist-confirmed ground truth. `CSTR` was excluded from the three-class probes pending an agreed biological label mapping. Report all results as preliminary.

A single-slide tumor-overlap check reported **0.8288 display-resolution rasterized IoU** versus **0.8253 in an earlier calculation**; the original derivation of the older result has not been reproduced. The example images and review candidates should not be treated as evidence of universal annotation accuracy.

## Repository structure

```text
src/
  task1/               # exported Task 1 scripts
  task2/               # exported Task 2 CLAM wrappers/evaluators
  task3/               # exported Task 3 scripts
  visualization/       # source recovered verbatim from executed terminal logs
scripts/
  export_from_pioneer.sh   # read-only copying of allowlisted source/results
  validate_results.py     # checks bundled aggregate CSVs
results/
  task2/              # aggregate patient-level metrics
  task3/              # aggregate patch validation metrics
figures/
  summary/            # charts derived from aggregate results
  methods/            # conceptual workflow and alignment diagram
docs/
  Methods_and_Figures_Handoff.md
  SOURCE_STATUS.md
  DATA_GOVERNANCE.md
  REPRODUCIBILITY.md
```

## Getting started and provenance

1. Review [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) and confirm your lab has approved private GitHub storage of even summary outputs.
2. Run `python3 scripts/validate_results.py`. This checks the included CSV structure and range, **not** model-training correctness.
3. Run `bash scripts/export_from_pioneer.sh` on Pioneer in the project directory to create an allowlisted staging copy of missing code and metrics. Inspect the staged files for protected data before adding them to Git.
4. Copy only the approved staged files into the corresponding `src/` or `results/` folders and update `docs/SOURCE_STATUS.md` with checksums, commit, and dates.
5. Create a **private repository** and push manually. Do not upload raw SVS/WSI, patient-level tables, checkpoints, embedding matrices, raw patient-linked paths, or secrets.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the step-by-step execution order and known verification gaps. No CI workflow runs clinical-data pipelines here.
