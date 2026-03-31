import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from glcm_feature_extraction import extract_glcm_dataset
from dwt_feature_extraction import extract_dwt_dataset

# -------------------------------
# Create models directory
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# Load Features
# -------------------------------
X_glcm, y = extract_glcm_dataset()
X_dwt, _ = extract_dwt_dataset()

# Combine GLCM + DWT
X = np.hstack((X_glcm, X_dwt))
print("✅ Combined Feature Shape:", X.shape)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
print("✅ Feature scaler saved")

class_names = ["NORMAL", "BACTERIAL", "VIRAL", "COVID"]

# ===============================
# 🔹 SVM MODEL
# ===============================
svm_model = SVC(
    kernel="rbf",
    C=15,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, y_pred_svm)

print("\n🔹 SVM RESULTS")
print("Accuracy:", svm_acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

joblib.dump(svm_model, "models/svm_model.pkl")
print("✅ SVM model saved")

# ----- SVM Confusion Matrix -----
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6,5))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("models/svm_confusion_matrix.png")
plt.close()

# ===============================
# 🔹 RANDOM FOREST MODEL
# ===============================
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)

print("\n🔹 RANDOM FOREST RESULTS")
print("Accuracy:", rf_acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

joblib.dump(rf_model, "models/rf_model.pkl")
print("✅ Random Forest model saved")

# ----- RF Confusion Matrix -----
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("models/rf_confusion_matrix.png")
plt.close()

# ===============================
# 📊 CLASSIFICATION REPORT
# ===============================
print("\n📊 CLASSIFICATION REPORT (Random Forest)")
print(classification_report(y_test, y_pred_rf))

# ===============================
# 🔥 SAVE REAL METRICS
# ===============================
f1 = f1_score(y_test, y_pred_rf, average='weighted')

metrics = {
    "accuracy": float(rf_acc),
    "f1_score": float(f1)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

print("\n✅ REAL METRICS SAVED:", metrics)

print("\n✅ Confusion matrix graphs saved inside 'models' folder.")