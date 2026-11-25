import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from math import floor

import h5py

ROOT = Path("/mnt/vstor/CSE_BME_CCIPD/home/asg162")
FEATURES = (
    ROOT / "Virtual_Env/outputs/paper1_coords_512/"
    "20.0x_512px_0px_overlap/features_conch_v15"
)
SRA = ROOT / "SRA_wantedjsons"
OUT = ROOT / "Virtual_Env/outputs/task3_tissue"
OUT.mkdir(parents=True, exist_ok=True)

KEEP = {"TUM", "MUS", "STR", "CSTR"}
COVERAGE_MIN = 0.80
PURITY_MIN = 0.80

output_path = OUT / "task3_patch_labels_provisional.csv"
counts = Counter()
slides_done = 0

with output_path.open("w", newline="") as output:
    writer = csv.writer(output)
    writer.writerow([
        "patient_id", "slide_id", "patch_index",
        "x", "y", "sra_class", "coverage",
        "purity", "label_source"
    ])

    for h5_path in sorted(FEATURES.glob("*.h5")):
        slide = h5_path.stem
        sra_path = SRA / (
            f"{slide}.svs_classification_srma_detection.json"
        )

        if not sra_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            size = int(f["coords"].attrs["patch_size_level0"])

        with sra_path.open() as f:
            sra = json.load(f)

        tiles = {}
        for item in sra.values():
            c = item.get("coords")
            if not c:
                continue
            xs, ys = c
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            if x1 - x0 != size or y1 - y0 != size:
                continue
            key = (x0, y0)
            if key in tiles:
                raise ValueError(f"Duplicate SRA tile: {slide} {key}")
            tiles[key] = (x1, y1, item.get("class"))

        if not tiles:
            continue

        x_offsets = {x % size for x, y in tiles}
        y_offsets = {y % size for x, y in tiles}

        if len(x_offsets) != 1 or len(y_offsets) != 1:
            print(f"SKIP nonuniform grid: {slide}")
            continue

        ox = next(iter(x_offsets))
        oy = next(iter(y_offsets))
        patient_id = "-".join(slide.split("-")[:2])

        for i, (x, y) in enumerate(coords):
            x, y = int(x), int(y)
            x1, y1 = x + size, y + size
            areas = defaultdict(float)

            for ix in range(
                floor((x - ox) / size),
                floor((x1 - 1 - ox) / size) + 1
            ):
                for iy in range(
                    floor((y - oy) / size),
                    floor((y1 - 1 - oy) / size) + 1
                ):
                    sx = ox + ix * size
                    sy = oy + iy * size

                    tile = tiles.get((sx, sy))
                    if tile is None:
                        continue

                    tx1, ty1, cls = tile
                    area = (
                        max(0, min(x1, tx1) - max(x, sx))
                        * max(0, min(y1, ty1) - max(y, sy))
                    )
                    areas[cls] += area

            total = sum(areas.values())
            if total == 0:
                continue

            coverage = total / (size * size)
            dominant = max(areas, key=areas.get)
            purity = areas[dominant] / total

            if coverage < COVERAGE_MIN:
                continue
            if purity < PURITY_MIN:
                continue
            if dominant not in KEEP:
                continue

            writer.writerow([
                patient_id, slide, i, x, y,
                dominant, round(coverage, 5),
                round(purity, 5), "SRA_provisional"
            ])
            counts[dominant] += 1

        slides_done += 1

print("Slides processed:", slides_done)
print("Candidate patches:", sum(counts.values()))

for cls in sorted(KEEP):
    print(cls, counts[cls])

print("Saved:", output_path)
