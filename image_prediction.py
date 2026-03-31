import numpy as np
import cv2
import joblib
import json

from tensorflow.keras.models import load_model

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL", "COVID"]
IMG_SIZE = 224

# -------------------------------
# LOAD MODELS
# -------------------------------
svm_model = joblib.load("models/svm_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

cnn_model = load_model("models/cnn_model.h5")
vgg16_model = load_model("models/vgg16_model.keras")

# -------------------------------
# LOAD REAL METRICS
# -------------------------------
with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

MODEL_ACCURACY = metrics["accuracy"]
MODEL_F1_SCORE = metrics["f1_score"]

print("✅ All models + real metrics loaded")


# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    original = img.copy()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_norm = gray / 255.0
    cnn_input = gray_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    vgg_input = img / 255.0
    vgg_input = np.reshape(vgg_input, (1, IMG_SIZE, IMG_SIZE, 3))

    return original, gray_norm, cnn_input, vgg_input


# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(gray_img):
    import pywt
    from skimage.feature import graycomatrix, graycoprops

    coeffs = pywt.dwt2(gray_img, 'haar')
    LL, (LH, HL, HH) = coeffs

    dwt = []
    for band in [LL, LH, HL, HH]:
        dwt.append(np.mean(band))
        dwt.append(np.std(band))

    img_uint8 = (gray_img * 255).astype(np.uint8)

    glcm = graycomatrix(img_uint8, [1], [0], 256, True, True)

    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ]

    features = np.array(glcm_features + dwt).reshape(1, -1)
    return scaler.transform(features)


# -------------------------------
# HEATMAP
# -------------------------------
def generate_heatmap(gray_img):
    img = (gray_img * 255).astype("uint8")
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    edges = cv2.Canny(blur, 50, 150)
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return heatmap


# -------------------------------
# PREDICTION
# -------------------------------
def predict_for_flask(image_path):

    original, gray, cnn_input, vgg_input = preprocess_image(image_path)

    features = extract_features(gray)

    svm_probs = svm_model.predict_proba(features)[0]
    rf_probs = rf_model.predict_proba(features)[0]
    ml_probs = (svm_probs + rf_probs) / 2

    cnn_probs = cnn_model.predict(cnn_input)[0]
    vgg_probs = vgg16_model.predict(vgg_input)[0]

    final_probs = (
        0.6 * vgg_probs +
        0.3 * cnn_probs +
        0.1 * ml_probs
    )

    svm_class = CLASSES[np.argmax(svm_probs)]
    rf_class = CLASSES[np.argmax(rf_probs)]
    cnn_class = CLASSES[np.argmax(cnn_probs)]
    vgg_class = CLASSES[np.argmax(vgg_probs)]

    final_class = CLASSES[np.argmax(final_probs)]
    confidence = np.max(final_probs) * 100

    # Heatmap overlay
    heatmap = generate_heatmap(gray)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)

    # Label
    label = f"{final_class} ({confidence:.2f}%)"

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    cv2.rectangle(overlay, (10, 10), (10 + w + 10, 10 + h + 20), (0, 0, 0), -1)

    cv2.putText(
        overlay,
        label,
        (15, 10 + h + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return {
        "final_class": final_class,
        "confidence": round(confidence, 2),

        # ✅ REAL VALUES FROM FILE
        "f1_score": round(MODEL_F1_SCORE, 2),
        "accuracy": round(MODEL_ACCURACY * 100, 2),

        "svm": svm_class,
        "rf": rf_class,
        "cnn": cnn_class,
        "vgg16": vgg_class,

        "labeled_image": overlay
    }