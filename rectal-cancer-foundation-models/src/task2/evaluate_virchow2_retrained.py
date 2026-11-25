import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import h5py
import torch

from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score
)

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, choices=range(10))
args = parser.parse_args()

BASE = Path("/mnt/vstor/CSE_BME_CCIPD/home/asg162/Virtual_Env")
CLAM = BASE / "CLAM"
sys.path.insert(0, str(CLAM))

from models.model_clam import CLAM_MB

FEATURE_DIR = (
    BASE / "outputs/paper1_coords_512"
    / "20.0x_512px_0px_overlap/features_virchow2"
)
CHECKPOINT = (
    BASE / "outputs/task2_clam/retrain_results"
    / f"virchow2_s42/s_{args.fold}_checkpoint.pt"
)
SPLITS = CLAM / f"splits/task2_pcr_10fold/splits_{args.fold}.csv"
LABELS = CLAM / "dataset_csv/task2_pcr.csv"
OUT = BASE / f"outputs/task2_clam/evaluation_retrained/virchow2_fold{args.fold}"
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load metadata and exact fold-0 membership
meta = pd.read_csv(LABELS, dtype=str)
split = pd.read_csv(SPLITS, dtype=str)
label_map = {"non-pCR": 0, "pCR": 1}

assert meta["slide_id"].is_unique
assert set(meta["label"]) == set(label_map)

members = {}
for part in ("train", "val", "test"):
    ids = split[part].dropna().tolist()
    assert len(ids) == len(set(ids))
    members[part] = set(ids)

assert not (members["train"] & members["val"])
assert not (members["train"] & members["test"])
assert not (members["val"] & members["test"])
assert set.union(*members.values()) == set(meta["slide_id"])

for part in members:
    p = set(meta.loc[
        meta["slide_id"].isin(members[part]), "case_id"
    ])
    members[part] = p

assert not (members["train"] & members["val"])
assert not (members["train"] & members["test"])
assert not (members["val"] & members["test"])
print("Patient leakage checks passed.")

test_ids = split["test"].dropna().tolist()
test = meta.set_index("slide_id").loc[test_ids].reset_index()
test["Y"] = test["label"].map(label_map)

# Verify every test slide has readable 768-dimensional features
for slide_id in test_ids:
    path = FEATURE_DIR / f"{slide_id}.h5"
    assert path.is_file(), f"Missing: {path}"
    with h5py.File(path, "r") as f:
        shape = f["features"].shape
        assert len(shape) == 2 and shape[0] > 0
        assert shape[1] == 2560, (slide_id, shape)

print("Test feature files verified:", len(test_ids))

# Reconstruct the original CLAM model
model = CLAM_MB(
    dropout=0.25,
    n_classes=2,
    embed_dim=2560,
    size_arg="small"
)

state = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True
)
clean = {
    k.replace(".module", ""): v
    for k, v in state.items()
    if "instance_loss_fn" not in k
}
model.load_state_dict(clean, strict=True)
model.to(device)
model.eval()
print("Checkpoint loaded successfully.")

# Evaluate only the original held-out test set
rows = []
with torch.inference_mode():
    for _, item in test.iterrows():
        slide_id = item["slide_id"]
        with h5py.File(FEATURE_DIR / f"{slide_id}.h5", "r") as f:
            features = np.asarray(f["features"], dtype=np.float32)

        x = torch.from_numpy(features).to(device)
        _, probabilities, prediction, _, _ = model(x)

        rows.append({
            "slide_id": slide_id,
            "case_id": item["case_id"],
            "Y": int(item["Y"]),
            "Y_hat": int(prediction.item()),
            "pCR_probability": float(probabilities[0, 1].item())
        })
        print("Evaluated:", slide_id)

pred = pd.DataFrame(rows)
pred.to_csv(OUT / "slide_predictions.csv", index=False)

def metrics(df):
    y = df["Y"].to_numpy(dtype=int)
    yh = df["Y_hat"].to_numpy(dtype=int)
    scores = df["pCR_probability"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y, yh, labels=[0, 1]
    ).ravel()

    return {
        "n": len(df),
        "AUC": (
            roc_auc_score(y, scores)
            if len(np.unique(y)) == 2 else float("nan")
        ),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "F1": f1_score(y, yh, zero_division=0),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn)
    }

# Aggregate slide probabilities within each patient
patient = pred.groupby("case_id", as_index=False).agg(
    Y=("Y", "first"),
    pCR_probability=("pCR_probability", "mean"),
    n_slides=("slide_id", "size")
)
patient["Y_hat"] = (
    patient["pCR_probability"] >= 0.5
).astype(int)
patient.to_csv(OUT / "patient_predictions.csv", index=False)

result = pd.DataFrame([
    {"level": "slide", **metrics(pred)},
    {"level": "patient", **metrics(patient)}
])
result.to_csv(OUT / f"fold{args.fold}_metrics.csv", index=False)

print(f"\nFOLD {args.fold} RESULTS")
print(result.to_string(index=False))
print("\nSaved to:", OUT)
