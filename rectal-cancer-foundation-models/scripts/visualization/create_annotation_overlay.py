import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import openslide

BASE = Path("/mnt/vstor/CSE_BME_CCIPD/home/asg162")
OUT = Path("task3_documentation/figures")
OUT.mkdir(parents=True, exist_ok=True)

slide_path = BASE / "Virtual_Env/valid_resection_WSIs/5619-7958__5619-7958-A10.svs"
manual_path = BASE / "Annotations/5619-7958__5619-7958-A10.geojson"
sra_path = BASE / "SRA_wantedjsons/5619-7958-A10.svs_classification_srma_detection.json"

manual = json.loads(manual_path.read_text())
sra = json.loads(sra_path.read_text())

tumor_features = [
    f for f in manual["features"]
    if f.get("properties", {}).get("classification", {}).get("name", "").lower() == "tumor"
]
tumor_tiles = [v for v in sra.values() if v.get("class") == "TUM"]

assert tumor_features, "No manual tumor annotation"
assert tumor_tiles, "No SRA tumor tiles"

# Collect level-0 tumor bounds from both sources.
xs, ys = [], []
for f in tumor_features:
    assert f["geometry"]["type"] == "MultiPolygon"
    for polygon in f["geometry"]["coordinates"]:
        for ring in polygon:
            xs.extend(p[0] for p in ring)
            ys.extend(p[1] for p in ring)

for tile in tumor_tiles:
    xs.extend(tile["coords"][0])
    ys.extend(tile["coords"][1])

with openslide.OpenSlide(str(slide_path)) as slide:
    W, H = slide.dimensions

    x0, x1 = max(0, min(xs)), min(W, max(xs))
    y0, y1 = max(0, min(ys)), min(H, max(ys))
    pad = 0.08 * max(x1 - x0, y1 - y0)

    x0 = max(0, math.floor(x0 - pad))
    y0 = max(0, math.floor(y0 - pad))
    x1 = min(W, math.ceil(x1 + pad))
    y1 = min(H, math.ceil(y1 + pad))

    # Read a cropped original-slide region at an appropriate pyramid level.
    width0, height0 = x1 - x0, y1 - y0
    target_max = 1400
    requested_ds = max(width0, height0) / target_max
    level = slide.get_best_level_for_downsample(max(1, requested_ds))
    ds = float(slide.level_downsamples[level])

    read_w = math.ceil(width0 / ds)
    read_h = math.ceil(height0 / ds)
    original = slide.read_region(
        (x0, y0), level, (read_w, read_h)
    ).convert("RGB")

# The coordinate transform maps level-0 annotations to the crop.
scale_x = original.width / width0
scale_y = original.height / height0

def transform(x, y):
    return ((x - x0) * scale_x, (y - y0) * scale_y)

def make_manual_mask():
    mask = Image.new("L", original.size, 0)
    draw = ImageDraw.Draw(mask)
    for feature in tumor_features:
        for polygon in feature["geometry"]["coordinates"]:
            if not polygon:
                continue
            # Exterior ring followed by any interior holes.
            draw.polygon(
                [transform(*p) for p in polygon[0]], fill=255
            )
            for hole in polygon[1:]:
                draw.polygon(
                    [transform(*p) for p in hole], fill=0
                )
    return mask

def make_sra_mask():
    mask = Image.new("L", original.size, 0)
    draw = ImageDraw.Draw(mask)
    for tile in tumor_tiles:
        xy = tile["coords"]
        corners = list(zip(xy[0], xy[1]))
        draw.polygon([transform(x, y) for x, y in corners], fill=255)
    return mask

manual_mask = make_manual_mask()
sra_mask = make_sra_mask()

def overlay(base, mask, color):
    result = base.copy()
    layer = Image.new("RGB", base.size, color)
    alpha = mask.point(lambda p: int(p * 0.45))
    result.paste(layer, (0, 0), alpha)
    return result

sra_view = overlay(original, sra_mask, (0, 180, 220))
manual_view = overlay(original, manual_mask, (255, 180, 0))
combined_view = overlay(sra_view, manual_mask, (255, 180, 0))

panels = [
    ("A. Original H&E slide region", original),
    ("B. SRA tumor tiles (cyan)", sra_view),
    ("C. Manual tumor polygons (orange)", manual_view),
    ("D. Combined overlay", combined_view),
]

margin, header, gap = 24, 48, 20
pw, ph = original.size
sheet = Image.new(
    "RGB",
    (2 * pw + 2 * margin + gap,
     2 * (ph + header) + 2 * margin + gap),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, (title, panel) in enumerate(panels):
    col, row = i % 2, i // 2
    px = margin + col * (pw + gap)
    py = margin + row * (ph + header + gap)
    draw.text((px, py + 10), title, fill="black")
    sheet.paste(panel, (px, py + header))

output = OUT / "task3_sra_manual_tumor_overlay.png"
sheet.save(output)

a = np.asarray(manual_mask) > 0
b = np.asarray(sra_mask) > 0
intersection = np.logical_and(a, b).sum()
union = np.logical_or(a, b).sum()

print("Saved:", output)
print("Slide crop level-0 bounds:", (x0, y0, x1, y1))
print("Pyramid level:", level)
print("Manual tumor features:", len(tumor_features))
print("SRA tumor tiles:", len(tumor_tiles))
print("Rasterized IoU at display resolution:",
      round(float(intersection / union), 4) if union else "undefined")
print("Note: display-resolution IoU is approximate, not exact polygon IoU.")
