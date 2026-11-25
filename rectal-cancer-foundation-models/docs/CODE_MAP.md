# Code map and execution context

| Stage | Code location | Purpose |
|---|---|---|
| Task 1a | `src/task1/segcoords512.slurm` and `feat_*.slurm` | Tissue segmentation, shared patch coordinates, five encoder feature extraction |
| Exploratory Task 1 | `src/task1/task1_dimensionality_reduction.py`, `task1_dimred.slurm` | Mean pooling, PCA/UMAP, grouped kNN |
| Task 2 | `src/task2/prepare_task2.py`, `task2_clam_*.slurm`, `train_*.slurm`, `evaluate_*.py`, `evaluate_*.slurm` | Grouped CLAM CV training and patient-level evaluation |
| Task 3 | `src/task3/task3_inventory.py`, `task3_build_labels.py`, `task3_create_splits.py`, `task3_train_conch.py`, `task3_five_model_compare.py` | SRA matching, splits, baseline and model comparisons |
| Figures | `src/visualization/*.py` | Existing patch contact sheet and SRA/manual annotation overlay |

**Important:** These scripts refer to paths and dependencies on CWRU Pioneer. The source tree alone is not an executable turnkey project. Do not run `sbatch` commands just to organize this repository. Keep patient-level data and histology images in approved storage.
