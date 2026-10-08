import os
import random
import shutil
from pathlib import Path

SOURCE = Path(r"C:\Users\Gajanan Ranmal\OneDrive\Desktop\Divider_Dataset\New_Dataset\divider_crops")

OUTPUT = Path(r"C:\Users\Gajanan Ranmal\OneDrive\Desktop\Divider_Dataset\divider_detection")

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Create folders
train_images = OUTPUT / "images" / "train"
val_images = OUTPUT / "images" / "val"
train_labels = OUTPUT / "labels" / "train"
val_labels = OUTPUT / "labels" / "val"

for folder in [train_images, val_images, train_labels, val_labels]:
    folder.mkdir(parents=True, exist_ok=True)

# Find images
images = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    images.extend(SOURCE.glob(ext))

images = sorted(images)

print(f"Found {len(images)} images")

if len(images) == 0:
    raise RuntimeError("No images found!")

# Shuffle
random.shuffle(images)

# Split
split = int(len(images) * TRAIN_RATIO)

train_set = images[:split]
val_set = images[split:]

print(f"Training images : {len(train_set)}")
print(f"Validation images: {len(val_set)}")


def process_image(src, image_dir, label_dir):
    dst = image_dir / src.name

    shutil.copy2(src, dst)

    # One divider occupies the entire crop.
    # YOLO format:
    # class x_center y_center width height
    label_file = label_dir / (src.stem + ".txt")

    with open(label_file, "w") as f:
        f.write("0 0.5 0.5 1.0 1.0\n")


for img in train_set:
    process_image(img, train_images, train_labels)

for img in val_set:
    process_image(img, val_images, val_labels)


# Create data.yaml
yaml_content = f"""path: {OUTPUT.as_posix()}
train: images/train
val: images/val

names:
  0: divider
"""

with open(OUTPUT / "data.yaml", "w") as f:
    f.write(yaml_content)

print("\n===================================")
print("DATASET READY")
print("===================================")
print(f"Dataset: {OUTPUT}")
print(f"Train:   {len(train_set)}")
print(f"Val:     {len(val_set)}")
print(f"YAML:    {OUTPUT / 'data.yaml'}")