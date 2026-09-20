import os
import shutil
from pathlib import Path

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# --- Step 1: download all four source datasets ---
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project1 = rf.workspace("potholes-r7qcn").project("pothole-jujbl")
dataset1 = project1.version(1).download("yolo26")

project2 = rf.workspace("jaydens-workspace-a3lmc").project("potholes-krmk8")
dataset2 = project2.version(2).download("yolo26")

# object-detection, multi-class crack taxonomy + a "Repair" class that
# isn't road damage
project3 = rf.workspace("mcmaster-vkq5k").project("pavement-crack-detection-r4n7n")
dataset3 = project3.version(2).download("yolo26")

# instance-segmentation (polygon labels, not boxes) — converted below
project4 = rf.workspace("smark-z7pr6").project("road-potholes-and-cracks-jmxtp")
dataset4 = project4.version(6).download("yolo26")

# --- Step 2: map every source dataset's own classes onto one shared taxonomy ---
TARGET_CLASSES = ["pothole", "crack"]
POTHOLE, CRACK = 0, 1


def load_source_names(dataset):
    data_yaml = yaml.safe_load((Path(dataset.location) / "data.yaml").read_text())
    return data_yaml["names"]


def single_class_map(dataset):
    """dataset1/dataset2 are known single-class pothole datasets — map
    whatever that one class is called straight to POTHOLE rather than
    matching by name (avoids silently dropping every box if the source
    project's label text differs from "pothole")."""
    names = load_source_names(dataset)
    assert len(names) == 1, f"expected a single-class dataset, got {names}"
    return {names[0]: POTHOLE}


# None = drop that class's boxes entirely. "Repair" is a road patch, not
# damage, and would just be noise for a pothole/crack detector.
DATASETS = [
    (dataset1, single_class_map(dataset1)),
    (dataset2, single_class_map(dataset2)),
    (dataset3, {
        "Pothole": POTHOLE,
        "Transverse crack": CRACK,
        "Longitudinal crack": CRACK,
        "Oblique crack": CRACK,
        "Alligator crack": CRACK,
        "Block crack": CRACK,
        "Repair": None,
    }),
    (dataset4, {"pothole": POTHOLE, "Crack": CRACK}),
]


def polygon_to_bbox(coords):
    """Segmentation labels store a polygon (x1 y1 x2 y2 ...); we only need
    a bounding box, so take the polygon's min/max extent."""
    xs, ys = coords[0::2], coords[1::2]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    return (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min


def remap_label_file(src_path, dst_path, source_names, class_map):
    out_lines = []
    for line in src_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        target = class_map.get(source_names[int(parts[0])])
        if target is None:
            continue  # dropped class
        coords = [float(x) for x in parts[1:]]
        if len(coords) > 4:
            cx, cy, w, h = polygon_to_bbox(coords)
        else:
            cx, cy, w, h = coords
        out_lines.append(f"{target} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


# --- Step 3: merge everything into one combined folder with unified labels ---
combined = Path("combined_dataset")
for split in ["train", "valid", "test"]:
    (combined / split / "images").mkdir(parents=True, exist_ok=True)
    (combined / split / "labels").mkdir(parents=True, exist_ok=True)
    for ds, class_map in DATASETS:
        source_names = load_source_names(ds)
        ds_path = Path(ds.location)
        img_dir, lbl_dir = ds_path / split / "images", ds_path / split / "labels"
        if not img_dir.exists():
            continue
        for img_file in img_dir.iterdir():
            shutil.copy(img_file, combined / split / "images" / f"{ds_path.name}_{img_file.name}")
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            dst_lbl = combined / split / "labels" / f"{ds_path.name}_{img_file.stem}.txt"
            if lbl_file.exists():
                remap_label_file(lbl_file, dst_lbl, source_names, class_map)
            else:
                dst_lbl.write_text("")  # background image, no boxes

(combined / "data.yaml").write_text(f"""
train: {combined.resolve()}/train/images
val: {combined.resolve()}/valid/images
nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}
""")

# --- Step 4: train on the merged dataset, continuing from the last checkpoint
# if the project's volume already has one (20 more epochs on top of the prior
# 50 = 70 total) rather than retraining from the base weights every job ---
checkpoint_dir = Path(os.environ.get("CHECKPOINT_PATH", "/b10/workspace/checkpoints"))
checkpoint_dir.mkdir(parents=True, exist_ok=True)
previous_weights = checkpoint_dir / "last.pt"
model = YOLO(str(previous_weights)) if previous_weights.exists() else YOLO("yolo26n.pt")
results = model.train(data=str(combined / "data.yaml"), epochs=20, imgsz=640, device=0)

# --- Step 5: persist weights to the checkpoint volume so they survive job teardown ---
weights_dir = Path(results.save_dir) / "weights"
for weight_file in ["best.pt", "last.pt"]:
    src = weights_dir / weight_file
    if src.exists():
        shutil.copy2(src, checkpoint_dir / weight_file)
        print(f"Saved {weight_file} to {checkpoint_dir / weight_file}")
