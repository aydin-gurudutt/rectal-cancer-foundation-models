import csv
import json
import random
from pathlib import Path
from collections import defaultdict, deque

import h5py
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)

ROOT = Path("outputs/task3_tissue")
FEATURES = Path(
    "outputs/paper1_coords_512/"
    "20.0x_512px_0px_overlap/features_conch_v15"
)

CLASSES = ["TUM", "MUS", "STR"]
SEED = 42
LIMITS = {"train": 6000, "val": 2000}

# Load verified slide-level patient splits
slide_info = {}

with (ROOT / "task3_slide_splits.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        slide_info[row["slide_id"]] = (
            row["case_id"], row["split"]
        )

# Organize candidate patches by split, class, patient
groups = defaultdict(list)

with (ROOT / "task3_patch_labels_provisional.csv").open(
    newline=""
) as f:
    for row in csv.DictReader(f):
        cls = row["sra_class"]
        slide = row["slide_id"]

        if cls not in CLASSES or slide not in slide_info:
            continue

        patient, split = slide_info[slide]

        if split not in LIMITS:
            continue

        assert row["patient_id"] == patient

        groups[(split, cls, patient)].append(
            (slide, int(row["patch_index"]))
        )

def select_patches(split, cls, limit):
    """Sample across patients, approximately evenly."""
    rng = random.Random(SEED)
    patients = sorted(
        patient for s, c, patient in groups
        if s == split and c == cls
    )

    queues = {}
    for patient in patients:
        patches = groups[(split, cls, patient)].copy()
        rng.shuffle(patches)
        queues[patient] = deque(patches)

    chosen = []
    active = patients.copy()

    while active and len(chosen) < limit:
        next_active = []

        for patient in active:
            if len(chosen) >= limit:
                break

            if queues[patient]:
                slide, idx = queues[patient].popleft()
                chosen.append((patient, slide, idx))

            if queues[patient]:
                next_active.append(patient)

        active = next_active

    return chosen

selections = {}

for split in LIMITS:
    for cls in CLASSES:
        selections[(split, cls)] = select_patches(
            split, cls, LIMITS[split]
        )
        print(
            f"{split} {cls}: "
            f"{len(selections[(split, cls)])} patches",
            flush=True
        )

# Feature loading by slide; preserve exact patch-index alignment
def build_matrix(split):
    selected = []
    for cls in CLASSES:
        selected.extend(
            (patient, slide, idx, cls)
            for patient, slide, idx
            in selections[(split, cls)]
        )

    by_slide = defaultdict(list)
    for pos, (_, slide, idx, _) in enumerate(selected):
        by_slide[slide].append((pos, idx))

    X = None
    y = np.array([CLASSES.index(r[3]) for r in selected])

    for slide, entries in sorted(by_slide.items()):
        path = FEATURES / f"{slide}.h5"

        with h5py.File(path, "r") as f:
            dataset = f["features"]

            indices = sorted(set(idx for _, idx in entries))
            vectors = dataset[indices]

            if X is None:
                X = np.empty(
                    (len(selected), vectors.shape[1]),
                    dtype=np.float32
                )

            index_to_row = {
                idx: i for i, idx in enumerate(indices)
            }

            for pos, idx in entries:
                X[pos] = vectors[index_to_row[idx]]

    if X is None:
        raise RuntimeError(f"No features selected for {split}")

    assert np.isfinite(X).all()
    return X, y, selected

print("\nLoading CONCH features...", flush=True)

X_train, y_train, train_rows = build_matrix("train")
X_val, y_val, val_rows = build_matrix("val")

print("Training matrix:", X_train.shape, flush=True)
print("Validation matrix:", X_val.shape, flush=True)

# Confirm patient independence
train_patients = {r[0] for r in train_rows}
val_patients = {r[0] for r in val_rows}

assert train_patients.isdisjoint(val_patients)

print("\nTraining linear probe...", flush=True)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=SEED
    )
)

model.fit(X_train, y_train)

print("\nEvaluating on validation patches...", flush=True)

pred = model.predict(X_val)

report = classification_report(
    y_val,
    pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0,
    output_dict=True
)

matrix = confusion_matrix(
    y_val, pred, labels=[0, 1, 2]
)

balanced_acc = balanced_accuracy_score(y_val, pred)

print("\n=== VALIDATION RESULTS ===")
print(
    classification_report(
        y_val,
        pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        zero_division=0
    )
)

print("Balanced accuracy:", round(balanced_acc, 4))
print("\nConfusion matrix (rows=true, columns=predicted)")
print("Class order:", CLASSES)
print(matrix)

model_path = ROOT / "conch_linear_probe_provisional.joblib"
joblib.dump(model, model_path)

results = {
    "classes": CLASSES,
    "train_samples": len(y_train),
    "val_samples": len(y_val),
    "train_patients": len(train_patients),
    "val_patients": len(val_patients),
    "balanced_accuracy": float(balanced_acc),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "seed": SEED,
    "label_source": "SRA_provisional",
    "test_evaluated": False
}

results_path = ROOT / "conch_linear_probe_val_metrics.json"

with results_path.open("w") as f:
    json.dump(results, f, indent=2)

sample_path = ROOT / "conch_linear_probe_sampled_patches.csv"

with sample_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "split", "case_id", "slide_id",
        "patch_index", "sra_class"
    ])

    for split, records in [
        ("train", train_rows),
        ("val", val_rows)
    ]:
        for patient, slide, idx, cls in records:
            writer.writerow([split, patient, slide, idx, cls])

print("\nSaved model:", model_path)
print("Saved metrics:", results_path)
print("Saved sampled patches:", sample_path)
print("Test set untouched.")
