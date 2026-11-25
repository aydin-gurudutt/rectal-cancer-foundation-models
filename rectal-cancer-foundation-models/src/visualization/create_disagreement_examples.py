import csv
from pathlib import Path
from PIL import Image, ImageDraw
import h5py
import openslide

ROOT = Path("outputs/task3_tissue")
SLIDES = Path("SlidesToUse")
FEATURES = Path(
    "outputs/paper1_coords_512/"
    "20.0x_512px_0px_overlap/features_conch_v15"
)
OUT = Path("task3_documentation/figures")
OUT.mkdir(parents=True, exist_ok=True)

with (ROOT / "muscle_stroma_review_candidates.csv").open() as f:
    rows = list(csv.DictReader(f))[:10]

thumb_size = 256
row_height = 300
sheet = Image.new("RGB", (780, row_height * len(rows)), "white")
draw = ImageDraw.Draw(sheet)

for i, row in enumerate(rows):
    slide_id = row["slide_id"]
    index = int(row["conch_patch_index"])
    x, y = int(row["x"]), int(row["y"])

    with h5py.File(FEATURES / f"{slide_id}.h5", "r") as h5:
        actual = tuple(map(int, h5["coords"][index]))
        assert actual == (x, y), (slide_id, index, actual, (x, y))

        attrs = h5["coords"].attrs
        assert float(attrs["target_magnification"]) == 20.0
        patch_level0 = float(attrs["patch_size_level0"])
        assert patch_level0 == 512.0

    with openslide.OpenSlide(str(SLIDES / f"{slide_id}.svs")) as slide:
        assert abs(float(slide.properties["openslide.objective-power"]) - 20) < 0.1
        assert x >= 0 and y >= 0
        assert x + 512 <= slide.dimensions[0]
        assert y + 512 <= slide.dimensions[1]
        patch = slide.read_region((x, y), 0, (512, 512)).convert("RGB")

    patch.save(OUT / f"disagreement_{i+1:02d}_{slide_id}_{index}.png")
    sheet.paste(patch.resize((thumb_size, thumb_size)), (12, i * row_height + 12))

    predictions = [
        row[f"{model}_prediction"]
        for model in ["conch_v15", "virchow2",
                      "phikon_v2", "hoptimus1", "musk"]
    ]

    label = (
        f"Slide: {slide_id}\n"
        f"Patch: {index}  Coordinates: ({x}, {y})\n"
        f"SRA label: {row['sra_class']}\n"
        f"Model predictions: {', '.join(predictions)}\n"
        f"Models disagreeing: {row['number_models_disagreeing']}/5"
    )
    draw.multiline_text((285, i * row_height + 25), label,
                        fill="black", spacing=9)

sheet.save(OUT / "task3_disagreement_contact_sheet.png")
print("Saved:", OUT / "task3_disagreement_contact_sheet.png")
print("Extracted patches:", len(rows))
