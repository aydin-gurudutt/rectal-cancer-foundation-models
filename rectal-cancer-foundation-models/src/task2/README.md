# Task 2 — original project scripts

CLAM cohort preparation, training/evaluation wrappers, fold evaluation scripts, and plotting.

These are copies of the scripts exported from Pioneer; they have **not been rewritten** or independently re-executed here.
Most paths are specific to the CWRU Pioneer HPC environment. Running the scripts requires their original data, dependencies, and (for CLAM/TRIDENT) the appropriate upstream or lab-modified installations.

## Included files

- `evaluate_conch_fold0.py`
- `evaluate_hoptimus1.py`
- `evaluate_hoptimus1_retrained.py`
- `evaluate_musk.py`
- `evaluate_phikon_v2.py`
- `evaluate_virchow2.py`
- `evaluate_virchow2_retrained.py`
- `generate_task2_figures.py`
- `prepare_task2.py`
- `evaluate_conch_array.slurm`
- `evaluate_conch_fold0.slurm`
- `evaluate_conch_fold1.slurm`
- `evaluate_hoptimus1.slurm`
- `evaluate_hoptimus1_retrained.slurm`
- `evaluate_musk.slurm`
- `evaluate_phikon_v2.slurm`
- `evaluate_virchow2.slurm`
- `evaluate_virchow2_retrained.slurm`
- `retrain_hoptimus1_fold8.slurm`
- `retrain_virchow2_fold4.slurm`
- `task2_clam_conch.slurm`
- `task2_clam_conch_array.slurm`
- `task2_clam_conch_fold1.slurm`
- `task2_clam_virchow2_fold0.slurm`
- `train_hoptimus1.slurm`
- `train_musk.slurm`
- `train_phikon_v2.slurm`
- `train_virchow2.slurm`

## Reproducibility notes

- Verify the exact upstream versions and local modifications of CLAM/TRIDENT before reproducing results.
- Verify patient grouping and label provenance; do not commit patient-level source tables or WSIs.
- The project supervisor has paused further model experiments.
