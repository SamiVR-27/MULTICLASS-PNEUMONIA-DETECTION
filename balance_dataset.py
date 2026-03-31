import os
import cv2
import random

SOURCE_DIR ="dataset"   # where all raw data exists
TARGET_DIR = "dataset_balanced_1000"
TARGET = 1000
IMG_SIZE = 224

CLASSES = ["BACTERIAL", "COVID", "NORMAL", "VIRAL"]

os.makedirs(TARGET_DIR, exist_ok=True)

for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)
    dst = os.path.join(TARGET_DIR, cls)
    os.makedirs(dst, exist_ok=True)

    images = os.listdir(src)
    random.shuffle(images)

    selected = images[:TARGET]

    for img_name in selected:
        img = cv2.imread(os.path.join(src, img_name))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(dst, img_name), img)

    print(f"✅ {cls} balanced to {TARGET} images")
