# Workflow and reproducibility map

## Source roots (Pioneer only)

Project: `/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env`. The actual source files have not all been uploaded to ChatGPT. The original scripts on Pioneer, not these narrative examples, are authoritative.

## Task 1a

1. Filter the approved 225-slide cohort from source metadata. One slide had an unsupported OpenSlide format; 224 processed slides remained.
2. Prepare symlinks into `SlidesToUse/`, preserving source image bytes.
3. Run TRIDENT HEST tissue segmentation.
4. Extract shared tissue patch coordinates at 20×, 512×512, zero overlap.
5. Extract per-patch H5 features for CONCH v1.5 (768-D), Virchow2 (2560-D), Phikon v2 (1024-D), H-Optimus-1 (1536-D), and MUSK (1024-D).
6. Export the *original* SLURM and TRIDENT scripts from Pioneer before using this repository to rerun.

## Exploratory Task 1

1. Load slide H5 `features`, average across patches for a slide vector.
2. Standardize vectors and generate PCA/UMAP; evaluate patient-grouped kNN and silhouettes.
3. **Known issue:** global feature scaling prior to grouped kNN CV is transductive leakage. Refit scalers in training folds if repeating the experiment, after supervisory approval.

## Task 2

1. Merge verified patient-level pCR response labels and group all slides by patient.
2. Use the same ten fixed patient-disjoint train/validation/test folds across the five encoders.
3. Train CLAM-MB (small) on the encoder-specific patch embeddings, retaining fold checkpoints outside Git.
4. Evaluate each fold at slide level; aggregate slide scores by patient with the **actual evaluator's rule, still to be inspected**.
5. Aggregate all patient out-of-fold predictions, report per-fold AUC/sensitivity/specificity/F1 mean ± SD as well as pooled 68-patient metrics. The CSV in `results/task2/` is a retyped aggregate from a terminal transcript, not the complete evaluator export.

## Task 3

1. Match SRA JSONs and feature H5s by slide ID; use spatial rectangle overlap, not matching array row indices. Require 80% SRA coverage of the feature patch and 80% dominant-class purity among covered area.
2. Preserve provenance (patient, slide, feature patch index and level-0 coordinates) in a restricted manifest, outside Git.
3. Verify patient identities and freeze 38/8/9 disjoint patient splits. One unmatched slide was excluded. The split seed was selected after checking test class proportions; disclose this design limitation.
4. Train/validate a balanced linear probe using 6,000 training and 2,000 validation patches per SRA class (TUM/MUS/STR); no test data used in training or audit.
5. Audit the common 24,000 sample coordinates across all five H5 sets; original job 4273013 confirmed all matches. Existing five-model training job 4273019 completed; later CONCH-only max-iteration check 4273066 completed.
6. Review existing disagreements and the one-slide tumor overlay, distinguishing approximate display raster IoU 0.8288 from historical 0.8253. The two image-generation programs in `src/visualization/` are copied from executed user-provided terminal content but cannot run without the authorized WSI and annotation files.
7. **Do not launch new model experiments** until the supervisor decides the Task 3 direction.

## Known missing items before declaring fully reproducible

- Actual Task 1, Task 2, Task 3 Python and SLURM scripts (except recovered visualization code).
- Task 2 evaluator patient aggregation function, classification threshold, and split-file provenance.
- Raw per-fold evaluator tables, per-class validation JSON, full source metadata and model checkpoints (keep restricted).
- Independent vector-based / original-resolution computation corresponding to historical IoU 0.8253.
- Clinical review of ambiguous MUS, STR, and CSTR definitions and the selected disagreement patches.
- Verified literature matrix for Task 1b.

No script in this repository should silently download slides or start a training job.
