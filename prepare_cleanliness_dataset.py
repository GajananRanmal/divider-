import os
import shutil
import random

SOURCE = r"C:\Users\Gajanan Ranmal\OneDrive\Desktop\Divider_Dataset\New_Dataset\labeled"

DEST = r"C:\Users\Gajanan Ranmal\OneDrive\Desktop\SWACH\datasets\divider_cleanliness"

CLASSES = ["clean", "medium", "dirty"]

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

for cls in CLASSES:

    source_folder = os.path.join(SOURCE, cls)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_folder = os.path.join(DEST, "train", cls)
    val_folder = os.path.join(DEST, "val", cls)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(val_folder, image)
        )

    print(f"\n{cls.upper()}")
    print(f"Total : {len(images)}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")

print("\n================================")
print("DATASET PREPARATION COMPLETE")
print("================================")