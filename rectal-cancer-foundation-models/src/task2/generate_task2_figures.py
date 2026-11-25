from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

BASE = Path("/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env")
DATA = BASE / "outputs/task2_clam/five_model_comparison_final"
OUT = DATA / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "MUSK": "musk",
    "CONCH v1.5": "conch",
    "H-Optimus-1": "hoptimus1",
    "Virchow2": "virchow2",
    "Phikon v2": "phikon_v2"
}

summary = pd.read_csv(DATA / "five_model_summary.csv")
folds = pd.read_csv(DATA / "all_model_fold_metrics.csv")

assert len(summary) == 5
assert len(folds) == 50

predictions = {}
reference = None

for name, code in MODELS.items():
    df = pd.read_csv(
        DATA / f"{code}_patient_oof.csv",
        dtype={"case_id": str}
    )

    assert len(df) == 68, (name, len(df))
    assert df["case_id"].is_unique, name
    assert df["Y"].isin([0, 1]).all(), name
    assert df["Y_hat"].isin([0, 1]).all(), name
    assert df["pCR_probability"].between(0, 1).all(), name
    assert df["fold"].nunique() == 10, name

    identity = (
        df[["case_id", "Y", "fold"]]
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    if reference is None:
        reference = identity
    else:
        pd.testing.assert_frame_equal(reference, identity)

    predictions[name] = df

    row = summary.loc[summary["Model"] == name]
    assert len(row) == 1
    calculated_auc = roc_auc_score(
        df["Y"], df["pCR_probability"]
    )
    assert np.isclose(
        calculated_auc, row["Pooled_AUC"].iloc[0]
    ), f"AUC mismatch: {name}"

    model_folds = folds.loc[folds["model"] == name]
    assert len(model_folds) == 10, name
    assert set(model_folds["fold"]) == set(range(10)), name

print("Validation passed: 5 models, 50 folds, 68 matching patients.")

# --------------------------------------------------
# FIGURE 1: POOLED PATIENT-LEVEL ROC CURVES
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))

for name, df in predictions.items():
    fpr, tpr, _ = roc_curve(
        df["Y"], df["pCR_probability"]
    )
    auc_value = roc_auc_score(
        df["Y"], df["pCR_probability"]
    )

    ax.plot(
        fpr, tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Chance (AUC = 0.500)"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("CLAM: Patient-Level ROC Curves")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "01_pooled_patient_roc.png", dpi=300)
fig.savefig(OUT / "01_pooled_patient_roc.pdf")
plt.close(fig)

# --------------------------------------------------
# FIGURE 2: 10-FOLD AUC DISTRIBUTIONS
# --------------------------------------------------

labels = list(MODELS.keys())

auc_arrays = [
    folds.loc[folds["model"] == name]
    .sort_values("fold")["AUC"].to_numpy()
    for name in labels
]

fig, ax = plt.subplots(figsize=(10, 6))

positions = np.arange(1, len(labels) + 1)

ax.boxplot(
    auc_arrays,
    positions=positions,
    widths=0.5,
    showmeans=True
)

# Overlay fold-level points deterministically.
for pos, values in zip(positions, auc_arrays):
    offsets = np.linspace(-0.10, 0.10, len(values))
    ax.scatter(
        np.full(len(values), pos) + offsets,
        values,
        alpha=0.65,
        s=25
    )

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Patient-Level AUC")
ax.set_title("CLAM: AUC Across 10 Patient-Grouped Folds")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "02_fold_auc_boxplot.png", dpi=300)
fig.savefig(OUT / "02_fold_auc_boxplot.pdf")
plt.close(fig)

# --------------------------------------------------
# FIGURE 3: CONFUSION MATRICES FOR ALL MODELS
# --------------------------------------------------

for name, df in predictions.items():
    cm = confusion_matrix(
        df["Y"],
        df["Y_hat"],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(5.8, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-pCR", "pCR"]
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(f"{name}: Patient-Level Confusion Matrix")

    fig.tight_layout()
    filename = MODELS[name]
    fig.savefig(
        OUT / f"03_confusion_{filename}.png",
        dpi=300
    )
    fig.savefig(
        OUT / f"03_confusion_{filename}.pdf"
    )
    plt.close(fig)

# --------------------------------------------------
# FIGURE 4: MEAN AUC WITH STANDARD DEVIATION
# --------------------------------------------------

ordered = (
    summary.set_index("Model")
    .loc[labels]
)

fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(labels))

ax.bar(
    x,
    ordered["AUC_mean"],
    yerr=ordered["AUC_SD"],
    capsize=6,
    alpha=0.8
)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Mean Patient-Level AUC")
ax.set_ylim(0, 1.1)
ax.set_title("CLAM: Mean 10-Fold AUC ± SD")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "04_mean_auc_sd.png", dpi=300)
fig.savefig(OUT / "04_mean_auc_sd.pdf")
plt.close(fig)

# --------------------------------------------------
# EXPORT FINAL PERFORMANCE TABLE
# --------------------------------------------------

columns = [
    "Model", "Patients",
    "AUC_mean", "AUC_SD",
    "sensitivity_mean", "sensitivity_SD",
    "specificity_mean", "specificity_SD",
    "F1_mean", "F1_SD",
    "Pooled_AUC", "Pooled_sensitivity",
    "Pooled_specificity", "Pooled_F1",
    "TP", "TN", "FP", "FN"
]

summary[columns].to_csv(
    OUT / "final_performance_table.csv",
    index=False
)

print("\n=== FINAL PERFORMANCE SUMMARY ===")
print(summary[columns].round(4).to_string(index=False))

print("\n=== GENERATED FIGURES ===")
for path in sorted(OUT.iterdir()):
    print(path.name, path.stat().st_size, "bytes")

print("\nSaved figures to:", OUT)
