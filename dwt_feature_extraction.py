import numpy as np
import pywt
from data_preprocessing import load_and_preprocess_dataset

def extract_dwt_features(image):
    """
    Extract DWT features from a single image
    """
    # Apply DWT
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    features = []

    # Extract statistical features from each sub-band
    for band in [LL, LH, HL, HH]:
        features.append(np.mean(band))
        features.append(np.std(band))

    return features


def extract_dwt_dataset():
    """
    Apply DWT feature extraction to the entire dataset
    """
    X_images, y = load_and_preprocess_dataset()

    features = []

    for img in X_images:
        dwt_features = extract_dwt_features(img)
        features.append(dwt_features)

    return np.array(features), y


if __name__ == "__main__":
    X_dwt, y = extract_dwt_dataset()

    print("✅ DWT FEATURE EXTRACTION COMPLETED")
    print("Feature Vector Shape:", X_dwt.shape)
    print("Labels Shape:", y.shape)
