# Source provenance and completeness

## Now included

- Original **custom** Task 1 (preprocessing, feature extraction, and dimensionality reduction), Task 2 (CLAM wrappers and evaluation), and Task 3 (SRA/linear-probe) source files were imported from the user-provided Pioneer export into `src/task1`, `src/task2`, and `src/task3`.
- Figure scripts in `src/visualization` were already present; identical exported copies were checked.
- Aggregate performance tables and approved summary figures are present in `results/` and `figures/`.

## Still not included or not verified

- The entire upstream/lab-modified `CLAM/` and `trident/` source trees and their pinned commits.
- Original patient- and slide-level datasets, SRA JSONs, large HDF5 feature matrices, histology WSIs, or model checkpoints (intentionally excluded).
- A tested reproducible environment lockfile and independent rerun from this GitHub snapshot.
- Task 1b literature matrix (not provided).
- The exact slide-to-patient probability aggregation implementation has not been independently audited.

## Research limitations

Task 2 is internal patient-grouped CV, not external validation. Task 3 labels are provisional SRA-derived categories, not independently reviewed pathology ground truth. The two reported tumor overlap values used different calculation contexts and should not be treated as identical methods. No new experiments are authorized by this repository organization.
