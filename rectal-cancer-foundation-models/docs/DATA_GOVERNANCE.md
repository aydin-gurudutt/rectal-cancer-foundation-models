# Data governance and sharing rules

This repository is intended to be **private and access-restricted**, pending the research group's approval. A private GitHub repository is not automatically an authorized repository for patient data or protected health information.

**Never commit:** raw histology slides (SVS), original or patient/slide-identifiable annotation files, label manifests with patient IDs, H5 embeddings/coordinates tied to patient IDs, patient/slide prediction CSVs, model checkpoints with unreviewed metadata, credentials/tokens, protected institutional paths or screenshots, or any restricted data. De-identification must be reviewed according to the institution's rules; replacing a patient name with a slide ID does not by itself make a record safe to share.

**Default included items:** code copied from the researcher's own project (after inspection and permission), synthetic/conceptual workflow diagrams, and approved aggregate metrics. Example pathology overlays/contact sheets are deliberately **not included** in the default repo archive; keep them in the approved secure project environment until explicit permission to share is confirmed.

Before pushing, inspect `git status --short` and `git diff --cached --name-only`, run `git ls-files`, and check all text files for identifiers and secrets. For robust checking, use institution-approved scanning; basic grep is not a compliance assessment. Securely share links only with authorized collaborators.

This repository is a methods/aggregate-results archive, **not** a public clinical dataset.
