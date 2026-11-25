import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("outputs/task3_tissue")
MANIFEST = ROOT / "task3_patch_labels_provisional.csv"
TASK2 = Path("CLAM/dataset_csv/task2_pcr.csv")

# Authoritative Task 2 slide-to-patient mapping
slide_to_patient = {}
with TASK2.open(newline="") as f:
    for row in csv.DictReader(f):
        slide_to_patient[row["slide_id"]] = row["case_id"]

# Aggregate provisional tissue labels per patient
patient_counts = defaultdict(Counter)
patient_slides = defaultdict(set)
unmatched_slides = set()

with MANIFEST.open(newline="") as f:
    for row in csv.DictReader(f):
        slide = row["slide_id"]

        if slide not in slide_to_patient:
            unmatched_slides.add(slide)
            continue

        patient = slide_to_patient[slide]
        patient_counts[patient][row["sra_class"]] += 1
        patient_slides[patient].add(slide)

patients = sorted(patient_counts)
n = len(patients)

if n < 10:
    raise RuntimeError("Too few patients for 3-way splitting")

# Approximately 70/15/15 by patient count
n_train = round(n * 0.70)
n_val = round(n * 0.15)

classes = ["TUM", "MUS", "STR", "CSTR"]
total_counts = Counter()
for counts in patient_counts.values():
    total_counts.update(counts)

total = sum(total_counts.values())
global_dist = {
    cls: total_counts[cls] / total for cls in classes
}

best_score = float("inf")
best_split = None

for seed in range(1000):
    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    groups = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train+n_val],
        "test": shuffled[n_train+n_val:],
    }

    score = 0.0
    valid = True

    for split, members in groups.items():
        counts = Counter()

        for patient in members:
            counts.update(patient_counts[patient])

        split_total = sum(counts.values())

        if split_total == 0:
            valid = False
            break

        # Require tumor and all candidate tissue classes
        if any(counts[cls] == 0 for cls in classes):
            valid = False
            break

        for cls in classes:
            proportion = counts[cls] / split_total
            score += abs(proportion - global_dist[cls])

    if valid and score < best_score:
        best_score = score
        best_split = (seed, groups)

if best_split is None:
    raise RuntimeError("No valid split found")

seed, groups = best_split

patient_output = ROOT / "task3_patient_splits.csv"
slide_output = ROOT / "task3_slide_splits.csv"

with patient_output.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "split"])

    for split, members in groups.items():
        for patient in sorted(members):
            writer.writerow([patient, split])

with slide_output.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "slide_id", "split"])

    for split, members in groups.items():
        for patient in sorted(members):
            for slide in sorted(patient_slides[patient]):
                writer.writerow([patient, slide, split])

print("=== PATIENT-GROUPED SPLIT ===")
print("Selected seed:", seed)
print("Score:", round(best_score, 5))
print("Total patients:", n)
print("Excluded unmatched slides:", sorted(unmatched_slides))

for split, members in groups.items():
    counts = Counter()
    n_slides = 0

    for patient in members:
        counts.update(patient_counts[patient])
        n_slides += len(patient_slides[patient])

    print(f"\n{split.upper()}")
    print("Patients:", len(members))
    print("Slides:", n_slides)
    print("Patches:", sum(counts.values()))

    for cls in classes:
        print(f"{cls}: {counts[cls]:,}")

# Confirm no patient overlap between partitions
sets = [set(members) for members in groups.values()]
assert not (sets[0] & sets[1])
assert not (sets[0] & sets[2])
assert not (sets[1] & sets[2])

print("\nPASS: No patient overlap between splits")
print("Saved:", patient_output)
print("Saved:", slide_output)
