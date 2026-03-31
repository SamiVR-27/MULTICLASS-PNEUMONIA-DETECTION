import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# -------------------------------
# DATASET PATH
# -------------------------------
DATASET_DIR = "dataset_split/train"   # ✅ IMPORTANT FIX

IMG_SIZE = 224
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL", "COVID"]

# -------------------------------
# LOAD IMAGES
# -------------------------------
X = []
y = []

for label, cls in enumerate(CLASSES):
    class_path = os.path.join(DATASET_DIR, cls)

    if not os.path.exists(class_path):
        print("❌ Missing:", class_path)
        continue

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=4)

print("✅ Dataset Loaded:", X.shape)

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# AUGMENTATION (IMPROVED)
# -------------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# -------------------------------
# LOAD VGG16
# -------------------------------
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ❌ Freeze first layers
for layer in base_model.layers[:-12]:
    layer.trainable = False

for layer in base_model.layers[-12:]:
    layer.trainable = True
# -------------------------------
# CUSTOM HEAD (IMPROVED)
# -------------------------------
x = base_model.output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)

vgg16_model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# COMPILE
# -------------------------------
vgg16_model.compile(
    optimizer=Adam(learning_rate=0.00001),  # 🔽 lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# CALLBACKS
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# -------------------------------
# TRAINING
# -------------------------------
history = vgg16_model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=12,   # 🔥 Sweet spot
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# -------------------------------
# EVALUATION
# -------------------------------
loss, accuracy = vgg16_model.evaluate(X_test, y_test)
print("\n✅ VGG16 TEST ACCURACY:", accuracy)

# -------------------------------
# SAVE MODEL
# -------------------------------
os.makedirs("models", exist_ok=True)

vgg16_model.save("models/vgg16_model.keras")  # ✅ modern format
print("✅ VGG16 model saved")