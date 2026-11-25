# Methods and figures — supervisor handoff
**Rectal cancer pathology foundation-model benchmarking | recorded through November 25, 2026**

## Scientific questions
**Task 2:** Can frozen pretrained patch features, aggregated by a CLAM multi-instance learner, predict a patient's pathological complete response (pCR)? Ground truth is a **patient** response, applied to all that patient's slides, while the network predicts each **slide** separately. The evaluation merges slide predictions into one patient record; the exact implemented merge function and threshold require inspection of original code.

**Task 3:** How separable are provisional SRA tile categories TUM, MUS, and STR in foundation-model patch embeddings? Ground truth for this baseline is **SRA-derived pseudo-labels**, not independently adjudicated tissue pathology. `CSTR` is a distinct source class and was excluded from the three-class baseline.

## Step-by-step pipeline
1. **Slide selection:** 225 source slides identified. One corrupt/unreadable image excluded; 224 processed. TRIDENT tissue segmentation and 20×, 512px, zero-overlap coordinates are reused across five encoders.
2. **Feature extraction:** Each slide's H5 carries `(N,d)` embedding rows and `(N,2)` top-left coordinates. Dimensions: CONCH 768, Virchow2 2560, Phikon v2 1024, H-Optimus-1 1536, MUSK 1024. Matched spatial positions do not imply identical neural-network native resizing.
3. **Exploration:** Mean-pool patch features to slide vectors; plot PCA and UMAP; assess grouped kNN and silhouette values. Original exploratory standardization preceded CV and thus gives possible train/test distribution leakage.
4. **Task 2 grouped benchmark:** Prepare 223 labeled slides from 68 patients, reuse fixed 10-fold patient-separated splits, train CLAM per encoder, evaluate slides and collapse to one OOF record per patient. MUSK achieved pooled AUC 0.8479; CONCH 0.8196. This is internal CV, not external validation.
5. **SRA matching:** The SRA JSON contains class-labeled 512px rectangles. H5 patches come from an independently rooted grid. On each slide, intersect rectangles geometrically, require ≥80% total SRA area coverage and ≥80% dominant-class purity. Record per-patch `slide_id`, `patch_index`, `x`, `y`, `sra_class`, `coverage`, `purity` rather than duplicating tensors.
6. **Patient grouping for Task 3:** Raw manifest 852,333 patches on 179 slides. A single unmatched Task 2 metadata slide was excluded to form 846,756 patches/178 slides/55 patients; split 38/8/9 patients. The split seed was selected partly using source class distributions in all partitions, including reserved test; disclose this nuance.
7. **Fixed sampling & audit:** Sample balanced TUM/MUS/STR: 6k each train and 2k each validation (24k locations). Audit checks exact `slide_id,(x,y)` against five separate H5 files; all 24k matched. This establishes shared locations, **not** annotation truth.
8. **Provisional classifier:** StandardScaler + logistic regression per model. MUSK achieved balanced accuracy 0.9208 and macro-F1 0.9204 on the 6k pseudo-labeled validation patches. CONCH baseline max_iter=500 emitted a convergence warning; rerun at max_iter=2000 gave balanced accuracy 0.904833 and F1 0.904946. No further experiments authorized.
9. **Visual evidence:** (a) ten real tissue patches in the review sheet, from one patient, with source label and model predictions; (b) four-panel original-slide/SRA/manual/combined tumor overlay for `5619-7958-A10`, 1,137 SRA TUM tiles vs one manually annotated tumor MultiPolygon (13 component groups). Display-resolution IoU 0.8288. A previously reported 0.8253 was computed by a different/unverified method; it should not be represented as a newly replicated independent score.

## Figures included
- `figures/workflow_overview.png` — schematic methods flow for Task 2 versus Task 3.
- `figures/spatial_alignment_concept.png` — hypothetical relative grid shift; **illustrative only**.
- `figures/task2_patient_auc.png` — AUC reconstructed from verified fold-summary CSV output.
- `figures/task3_validation_comparison.png` — balanced accuracy and macro-F1 transcribed from result summary.
- `figures/task3_disagreement_contact_sheet.png` — actual ten H&E patches and labels; single patient.
- `figures/task3_sra_manual_tumor_overlay.png` — actual four-panel overlay; display IoU 0.8288.

## Scientific caveats and decisions pending
- Do not infer diagnosis, pathology-label correctness, or significance from the best preliminary validation score.
- Never use the nine-patient reserved Task 3 test partition before supervisor approval.
- Validate MUS, STR, and CSTR with the SRA annotation provider/pathologist.
- Inspect the Task 2 prediction aggregation function, patient probability threshold, and possible scanner/site effects.
- Package original scripts only after obtaining HPC source bytes and checking lab-approved access rules.
