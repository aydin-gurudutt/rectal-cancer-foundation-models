
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE = Path("/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env")
CLAM = BASE / "CLAM"
META = BASE / "slurm/Benchmarking_Feature_Extraction/slide_metadata.csv"
FEATURE_ROOT = BASE / "outputs/paper1_coords_512/20.0x_512px_0px_overlap"

ENCODERS = [
    "conch_v15",
    "virchow2",
    "phikon_v2",
    "hoptimus1",
    "musk"
]

OUT = CLAM / "splits/task2_pcr_10fold"
OUT.mkdir(parents=True, exist_ok=True)

# Load actual project metadata
df = pd.read_csv(META, dtype={"slide_id": str, "patient_id": str})
df["slide_id"] = df["slide_id"].replace({"5650-7928-A6-20X": "5650-7928-A6-20x"})

assert df[["slide_id", "patient_id", "pCR"]].notna().all().all()
assert set(df["pCR"].unique()) == {0, 1}
assert df["slide_id"].is_unique
assert df.groupby("patient_id")["pCR"].nunique().eq(1).all()

# Verify identical slide coverage across encoders
for encoder in ENCODERS:
    folder = FEATURE_ROOT / f"features_{encoder}"
    existing = {p.stem for p in folder.glob("*.h5")}
    missing = [
        s for s in df["slide_id"]
        if s not in existing
    ]
    if missing:
        raise RuntimeError(f"{encoder}: missing slides {missing}")

# Create CLAM-compatible dataset CSV
labels = pd.DataFrame({
    "case_id": df["patient_id"],
    "slide_id": df["slide_id"],
    "label": df["pCR"].map({0: "non-pCR", 1: "pCR"})
})

labels.to_csv(CLAM / "dataset_csv/task2_pcr.csv", index=False)

# One label per patient
patients = (
    df.groupby("patient_id", as_index=False)["pCR"]
    .first()
)

patient_ids = patients["patient_id"].to_numpy()
patient_labels = patients["pCR"].to_numpy()

# Ten non-overlapping, stratified patient-level test folds
outer = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

all_test_patients = []

for fold, (trainval_idx, test_idx) in enumerate(
    outer.split(patient_ids, patient_labels)
):
    trainval_ids = patient_ids[trainval_idx]
    trainval_y = patient_labels[trainval_idx]
    test_ids = patient_ids[test_idx]

    # Inner validation: roughly 10% of remaining patients
    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=0.10,
        stratify=trainval_y,
        random_state=42 + fold
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set

    def get_slides(ids):
        return labels.loc[
            labels["case_id"].isin(ids), "slide_id"
        ].tolist()

    train_slides = get_slides(train_set)
    val_slides = get_slides(val_set)
    test_slides = get_slides(test_set)

    # Pad uneven columns, as expected by CLAM's split reader
    split_df = pd.DataFrame({
        "train": pd.Series(train_slides),
        "val": pd.Series(val_slides),
        "test": pd.Series(test_slides)
    })

    split_df.to_csv(OUT / f"splits_{fold}.csv", index=False)

    all_test_patients.extend(test_ids)

    print(
        f"Fold {fold}: "
        f"train={len(train_set)} patients, "
        f"val={len(val_set)} patients, "
        f"test={len(test_set)} patients"
    )

assert len(all_test_patients) == len(patient_ids)
assert len(set(all_test_patients)) == len(patient_ids)

print("\nSuccessfully created 10 patient-grouped folds.")
print(f"Total slides: {len(labels)}")
print(f"Total patients: {len(patient_ids)}")
print(f"Output directory: {OUT}")
