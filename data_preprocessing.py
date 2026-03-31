import os
import cv2
import numpy as np
from sklearn.utils import shuffle

# ✅ USE FULL DATASET (before split)
DATASET_PATH = "dataset_balanced_1000"

# ✅ CLASS NAMES (MATCH EXACTLY)
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL", "COVID"]

IMG_SIZE = 224


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray)

    filtered = cv2.medianBlur(equalized, 5)

    normalized = filtered / 255.0

    return normalized


def load_and_preprocess_dataset():
    images = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_path):
            print(f"⚠️ Folder not found: {class_path}")
            continue

        for file_name in os.listdir(class_path):
            img_path = os.path.join(class_path, file_name)
            processed_img = preprocess_image(img_path)

            if processed_img is not None:
                images.append(processed_img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # ✅ SHUFFLE DATA
    images, labels = shuffle(images, labels, random_state=42)

    return images, labels


if __name__ == "__main__":
    X, y = load_and_preprocess_dataset()

    print("✅ DATA PREPROCESSING COMPLETED")
    print("Total Images :", X.shape[0])
    print("Image Shape  :", X.shape[1:])
    print("Labels Shape :", y.shape)
