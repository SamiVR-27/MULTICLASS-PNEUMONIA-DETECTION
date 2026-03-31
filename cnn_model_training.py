import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI errors

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# DATASET PATH
# -------------------------------
DATASET_DIR = "dataset_balanced_1000"   # Make sure this is your balanced dataset
IMG_SIZE = 224
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL", "COVID"]

# -------------------------------
# LOAD IMAGES
# -------------------------------
X = []
y = []

for label, cls in enumerate(CLASSES):
    class_path = os.path.join(DATASET_DIR, cls)

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

print("✅ Dataset Loaded:", X.shape)

# -------------------------------
# ONE HOT ENCODING
# -------------------------------
y_cat = to_categorical(y, num_classes=4)

# -------------------------------
# TRAIN-TEST SPLIT (FIXED)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT FIX
)

# -------------------------------
# CLASS WEIGHTS (IMPORTANT)
# -------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

datagen.fit(X_train)

# -------------------------------
# CNN MODEL (IMPROVED ARCHITECTURE)
# -------------------------------
cnn_model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')
])

# -------------------------------
# COMPILE MODEL
# -------------------------------
cnn_model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    verbose=1
)

# -------------------------------
# TRAINING
# -------------------------------
history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=15,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# -------------------------------
# EVALUATION
# -------------------------------
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print("\n✅ CNN TEST ACCURACY:", accuracy)

# -------------------------------
# SAVE MODEL
# -------------------------------
os.makedirs("models", exist_ok=True)
cnn_model.save("models/cnn_model.h5")
print("✅ CNN model saved as models/cnn_model.h5")

