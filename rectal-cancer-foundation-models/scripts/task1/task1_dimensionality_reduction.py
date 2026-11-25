"""
Task 1: Dimensionality reduction on foundation model features.

For each encoder (conch_v15, virchow2, phikon_v2, h_optimus_1, musk):
  1. Load patch-level features per slide, mean-pool to one vector per slide.
  2. Standardize features.
  3. Run PCA -> report variance explained by PC1/PC2, save scatter plot.
  4. Run UMAP (unsupervised, no labels used) -> save scatter plots colored by
     (a) pCR/non-pCR response status, and (b) patient ID.
  5. Repeat UMAP across a few n_neighbors/min_dist/seed combinations to check
     stability of any apparent clustering.
  6. Run patient-grouped kNN cross-validation and silhouette scores on the
     RAW standardized features (not the 2D projection) as a quantitative
     check that doesn't rely on eyeballing the plot.

Usage:
    python task1_dimensionality_reduction.py \
        --job_dir /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/paper1_coords_512 \
        --metadata_csv /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/slurm/Benchmarking_Feature_Extraction/slide_metadata.csv \
        --out_dir /mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env/outputs/task1_results \
        --encoders conch_v15 virchow2 phikon_v2 h_optimus_1 musk
"""

import argparse
import os
import glob
import json

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score

import umap  # pip install umap-learn


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def find_h5_for_slide(features_dir, slide_id):
    """Case-insensitive lookup, since some slide names on disk differ in
    case from the xlsx-derived metadata (e.g. '...A6-20X' vs '...a6-20x')."""
    exact = os.path.join(features_dir, f"{slide_id}.h5")
    if os.path.exists(exact):
        return exact
    candidates = glob.glob(os.path.join(features_dir, "*.h5"))
    target = slide_id.lower()
    for c in candidates:
        base = os.path.splitext(os.path.basename(c))[0]
        if base.lower() == target:
            return c
    return None


def load_slide_level_features(features_dir, metadata_df):
    """Mean-pool patch-level features to one vector per slide.
    Returns X (n_slides, dim), and the subset of metadata_df actually found."""
    feats = []
    keep_rows = []
    missing = []

    for _, row in metadata_df.iterrows():
        h5_path = find_h5_for_slide(features_dir, row["slide_id"])
        if h5_path is None:
            missing.append(row["slide_id"])
            continue
        with h5py.File(h5_path, "r") as f:
            patch_feats = f["features"][:]
        feats.append(patch_feats.mean(axis=0))
        keep_rows.append(row)

    if missing:
        print(f"  WARNING: {len(missing)} slides had no matching h5 file:")
        for m in missing:
            print(f"    missing: {m}")

    X = np.stack(feats, axis=0)
    meta = pd.DataFrame(keep_rows).reset_index(drop=True)
    return X, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter_colored(coords, labels, title, out_path, categorical=True, legend=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        unique_labels = sorted(pd.Series(labels).unique(), key=str)
        cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
        for i, lab in enumerate(unique_labels):
            mask = np.array(labels) == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.8,
                       color=cmap(i), label=str(lab))
        if legend and len(unique_labels) <= 25:
            ax.legend(markerscale=1.5, fontsize=7, bbox_to_anchor=(1.02, 1),
                      loc="upper left", ncol=1 if len(unique_labels) <= 15 else 2)
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=18)
        fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quantitative validation
# ---------------------------------------------------------------------------

def patient_grouped_knn_cv(X, y, groups, k=5, n_splits=5):
    n_splits = min(n_splits, pd.Series(groups).nunique())
    gkf = GroupKFold(n_splits=n_splits)
    accs, aucs = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        try:
            proba = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], proba))
        except ValueError:
            pass  # only one class present in this fold's test set
    return {
        "n_splits": n_splits,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
    }


# ---------------------------------------------------------------------------
# Main per-encoder pipeline
# ---------------------------------------------------------------------------

def run_encoder(encoder, job_dir, metadata_df, out_dir):
    print(f"\n=== {encoder} ===")
    features_dir = os.path.join(job_dir, f"features_{encoder}")
    if not os.path.isdir(features_dir):
        print(f"  SKIP: {features_dir} not found")
        return None

    enc_out = os.path.join(out_dir, encoder)
    os.makedirs(enc_out, exist_ok=True)

    X_raw, meta = load_slide_level_features(features_dir, metadata_df)
    print(f"  Loaded {X_raw.shape[0]} slides, feature dim = {X_raw.shape[1]}")

    y = meta["pCR"].astype(int).values
    patient_ids = meta["patient_id"].values
    response_label = np.where(y == 1, "pCR", "non-pCR")

    # Standardize
    X = StandardScaler().fit_transform(X_raw)

    # --- PCA ---
    pca = PCA(n_components=2, random_state=0)
    pca_coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")
    scatter_colored(pca_coords, response_label,
                     f"{encoder}: PCA (PC1 {var_explained[0]:.1%}, PC2 {var_explained[1]:.1%}) colored by response",
                     os.path.join(enc_out, "pca_by_response.png"))

    # --- Primary UMAP (default-ish settings) ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
    umap_coords = reducer.fit_transform(X)

    scatter_colored(umap_coords, response_label,
                     f"{encoder}: UMAP colored by response status",
                     os.path.join(enc_out, "umap_by_response.png"))
    scatter_colored(umap_coords, patient_ids,
                     f"{encoder}: UMAP colored by patient ID",
                     os.path.join(enc_out, "umap_by_patient.png"), legend=False)

    # --- Stability check across a few settings ---
    stability_settings = [
        {"n_neighbors": 10, "min_dist": 0.1, "random_state": 0},
        {"n_neighbors": 15, "min_dist": 0.1, "random_state": 1},
        {"n_neighbors": 30, "min_dist": 0.3, "random_state": 0},
        {"n_neighbors": 50, "min_dist": 0.5, "random_state": 2},
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, params in zip(axes.ravel(), stability_settings):
        r = umap.UMAP(**params)
        c = r.fit_transform(X)
        for lab, color in [("pCR", "tab:blue"), ("non-pCR", "tab:orange")]:
            mask = response_label == lab
            ax.scatter(c[mask, 0], c[mask, 1], s=10, alpha=0.7, label=lab, color=color)
        ax.set_title(f"n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}, seed={params['random_state']}")
        ax.legend(fontsize=7)
    fig.suptitle(f"{encoder}: UMAP stability across settings (colored by response)")
    fig.tight_layout()
    fig.savefig(os.path.join(enc_out, "umap_stability_grid.png"), dpi=150)
    plt.close(fig)

    # --- Quantitative validation on RAW standardized features ---
    knn_results = patient_grouped_knn_cv(X, y, patient_ids, k=5, n_splits=5)

    sil_response = silhouette_score(X, response_label) if len(set(response_label)) > 1 else None
    sil_patient = silhouette_score(X, patient_ids) if len(set(patient_ids)) > 1 else None

    summary = {
        "encoder": encoder,
        "n_slides": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "n_patients": int(pd.Series(patient_ids).nunique()),
        "pca_variance_explained_pc1": float(var_explained[0]),
        "pca_variance_explained_pc2": float(var_explained[1]),
        "patient_grouped_knn_cv": knn_results,
        "silhouette_response": sil_response,
        "silhouette_patient": sil_patient,
    }

    with open(os.path.join(enc_out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  kNN patient-grouped CV: acc={knn_results['mean_acc']:.3f} +/- {knn_results['std_acc']:.3f}, "
          f"AUC={knn_results['mean_auc']}")
    print(f"  Silhouette (response) = {sil_response}, Silhouette (patient) = {sil_patient}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_dir", required=True,
                     help="Directory containing features_<encoder>/ subfolders")
    ap.add_argument("--metadata_csv", required=True,
                     help="CSV with columns: slide_id, patient_id, pCR, RRS")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoders", nargs="+",
                     default=["conch_v15", "virchow2", "phikon_v2", "h_optimus_1", "musk"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metadata_df = pd.read_csv(args.metadata_csv)
    print(f"Loaded metadata for {len(metadata_df)} slides "
          f"({metadata_df['patient_id'].nunique()} patients)")

    all_summaries = []
    for encoder in args.encoders:
        s = run_encoder(encoder, args.job_dir, metadata_df, args.out_dir)
        if s is not None:
            all_summaries.append(s)

    with open(os.path.join(args.out_dir, "all_encoders_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nDone. Per-encoder plots + summary.json written under:", args.out_dir)


if __name__ == "__main__":
    main()
