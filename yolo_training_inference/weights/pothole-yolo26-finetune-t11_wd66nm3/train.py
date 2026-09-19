import os
import shutil
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# --- Step 1: download both datasets ---
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project1 = rf.workspace("potholes-r7qcn").project("pothole-jujbl")
dataset1 = project1.version(1).download("yolo26")

project2 = rf.workspace("jaydens-workspace-a3lmc").project("potholes-krmk8")
dataset2 = project2.version(2).download("yolo26")

# --- Step 3: merge both into one combined dataset folder ---
combined = Path("combined_dataset")
for split in ["train", "valid", "test"]:
    (combined / split / "images").mkdir(parents=True, exist_ok=True)
    (combined / split / "labels").mkdir(parents=True, exist_ok=True)
    for ds in [Path(dataset1.location), Path(dataset2.location)]:
        for kind in ["images", "labels"]:
            src_dir = ds / split / kind
            if not src_dir.exists():
                continue
            for f in src_dir.iterdir():
                shutil.copy(f, combined / split / kind / f"{ds.name}_{f.name}")

(combined / "data.yaml").write_text(f"""
train: {combined.resolve()}/train/images
val: {combined.resolve()}/valid/images
nc: 1
names: ['pothole']
""")

# --- Step 4: train on the merged dataset ---
model = YOLO("yolo26n.pt")
model.train(data=str(combined / "data.yaml"), epochs=50, imgsz=640, device=0)