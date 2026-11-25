# Task 3 — original project scripts

SRA tile inventory and matching, patient splits, CONCH linear probe, and five-model comparison.

These are copies of the scripts exported from Pioneer; they have **not been rewritten** or independently re-executed here.
Most paths are specific to the CWRU Pioneer HPC environment. Running the scripts requires their original data, dependencies, and (for CLAM/TRIDENT) the appropriate upstream or lab-modified installations.

## Included files

- `task3_build_labels.py`
- `task3_conch_2000.py`
- `task3_create_splits.py`
- `task3_five_model_compare.py`
- `task3_inventory.py`
- `task3_train_conch.py`
- `task3_conch_2000.slurm`
- `task3_five_model_compare.slurm`
- `task3_train_conch.slurm`

## Reproducibility notes

- Verify the exact upstream versions and local modifications of CLAM/TRIDENT before reproducing results.
- Verify patient grouping and label provenance; do not commit patient-level source tables or WSIs.
- The project supervisor has paused further model experiments.
