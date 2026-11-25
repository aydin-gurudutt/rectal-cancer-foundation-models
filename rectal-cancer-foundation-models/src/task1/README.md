# Task 1 — original project scripts

TRIDENT tissue segmentation, feature-extraction SLURM wrappers, and dimensionality reduction.

These are copies of the scripts exported from Pioneer; they have **not been rewritten** or independently re-executed here.
Most paths are specific to the CWRU Pioneer HPC environment. Running the scripts requires their original data, dependencies, and (for CLAM/TRIDENT) the appropriate upstream or lab-modified installations.

## Included files

- `task1_dimensionality_reduction.py`
- `feat_conchv15.slurm`
- `feat_hoptimus1.slurm`
- `feat_musk.slurm`
- `feat_phikonv2.slurm`
- `feat_virchow2.slurm`
- `segcoords512.slurm`
- `task1_dimred.slurm`

## Reproducibility notes

- Verify the exact upstream versions and local modifications of CLAM/TRIDENT before reproducing results.
- Verify patient grouping and label provenance; do not commit patient-level source tables or WSIs.
- The project supervisor has paused further model experiments.
