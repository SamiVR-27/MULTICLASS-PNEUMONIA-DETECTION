import os
import shutil
import random

SOURCE_DIR = "dataset_balanced_1000"
TARGET_DIR = "dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["BACTERIAL", "COVID", "NORMAL", "VIRAL"]

for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)

    train_dir = os.path.join(TARGET_DIR, "train", cls)
    val_dir = os.path.join(TARGET_DIR, "val", cls)
    test_dir = os.path.join(TARGET_DIR, "test", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = os.listdir(src)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(train_dir, img))

    for img in val_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(val_dir, img))

    for img in test_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(test_dir, img))

    print(f"✅ {cls} → Train:{len(train_imgs)} Val:{len(val_imgs)} Test:{len(test_imgs)}")

print("\n🎉 Dataset splitting completed successfully!")
