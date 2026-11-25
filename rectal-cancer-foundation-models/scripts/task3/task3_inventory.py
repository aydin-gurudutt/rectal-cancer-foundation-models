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
OUTPUT = ROOT / "Virtual_Env/outputs/task3_tissue"
OUTPUT.mkdir(parents=True, exist_ok=True)

COVERAGE_THRESHOLD = 0.80
PURITY_THRESHOLD = 0.80

rows = []

for h5_path in sorted(FEATURES.glob("*.h5")):
    slide = h5_path.stem
    sra_path = SRA / f"{slide}.svs_classification_srma_detection.json"

    if not sra_path.exists():
        continue

    try:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            size = int(f["coords"].attrs["patch_size_level0"])

        with open(sra_path) as f:
            data = json.load(f)

        tiles = {}
        for item in data.values():
            c = item.get("coords")
            if not c:
                continue

            x, y = c
            x0, x1 = int(min(x)), int(max(x))
            y0, y1 = int(min(y)), int(max(y))
            tiles[(x0, y0)] = (x1, y1, item.get("class"))

        if not tiles:
            print(f"[SKIP] {slide}: no valid tiles")
            continue

        ox_set = {x % size for x, y in tiles}
        oy_set = {y % size for x, y in tiles}

        if len(ox_set) != 1 or len(oy_set) != 1:
            print(f"[SKIP] {slide}: nonuniform SRA grid")
            continue

        ox, oy = next(iter(ox_set)), next(iter(oy_set))
        counts = Counter()

        for x, y in coords:
            x, y = int(x), int(y)
            x1, y1 = x + size, y + size
            areas = defaultdict(float)

            ix0 = floor((x - ox) / size)
            ix1 = floor((x1 - 1 - ox) / size)
            iy0 = floor((y - oy) / size)
            iy1 = floor((y1 - 1 - oy) / size)

            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    sx = ox + ix * size
                    sy = oy + iy * size

                    tile = tiles.get((sx, sy))
                    if tile is None:
                        continue

                    tx1, ty1, cls = tile
                    overlap = (
                        max(0, min(x1, tx1) - max(x, sx))
                        * max(0, min(y1, ty1) - max(y, sy))
                    )
                    areas[cls] += overlap

            total = sum(areas.values())
            coverage = total / (size ** 2)

            if not areas or coverage < COVERAGE_THRESHOLD:
                counts["INSUFFICIENT_COVERAGE"] += 1
                continue

            dominant = max(areas, key=areas.get)
            purity = areas[dominant] / total

            if purity < PURITY_THRESHOLD:
                counts["MIXED"] += 1
                continue

            counts[dominant] += 1

        row = {
            "slide": slide,
            "total_patches": len(coords),
            "TUM": counts["TUM"],
            "MUS": counts["MUS"],
            "STR": counts["STR"],
            "CSTR": counts["CSTR"],
            "ADI": counts["ADI"],
            "MUC": counts["MUC"],
            "MIXED": counts["MIXED"],
            "INSUFFICIENT_COVERAGE": counts["INSUFFICIENT_COVERAGE"],
        }
        rows.append(row)

        print(
            f"[OK] {slide}: "
            f"TUM={row['TUM']} "
            f"MUS={row['MUS']} "
            f"STR={row['STR']} "
            f"CSTR={row['CSTR']}"
        )

    except Exception as e:
        print(f"[ERROR] {slide}: {e}")

if rows:
    output_csv = OUTPUT / "task3_slide_inventory.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== COHORT SUMMARY ===")
    print("Slides processed:", len(rows))

    for cls in ("TUM", "MUS", "STR", "CSTR"):
        total = sum(row[cls] for row in rows)
        slides = sum(row[cls] > 0 for row in rows)
        print(f"{cls}: {total} patches across {slides} slides")

    print("\nTOP 15 SLIDES BY TUMOR PATCHES")
    for row in sorted(rows, key=lambda r: r["TUM"], reverse=True)[:15]:
        print(
            row["slide"],
            "TUM:", row["TUM"],
            "MUS:", row["MUS"],
            "STR:", row["STR"],
            "CSTR:", row["CSTR"]
        )

    print("\nSaved:", output_csv)
else:
    print("No slides processed successfully.")
