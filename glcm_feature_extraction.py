import numpy as np
from skimage.feature import graycomatrix, graycoprops
from data_preprocessing import load_and_preprocess_dataset

def extract_glcm_features(image):
    """
    Extract GLCM texture features from a single image
    """
    # Convert normalized image to uint8
    image = (image * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract texture properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return [contrast, correlation, energy, homogeneity]


def extract_glcm_dataset():
    """
    Apply GLCM feature extraction to the entire dataset
    """
    X_images, y = load_and_preprocess_dataset()

    features = []

    for img in X_images:
        glcm_features = extract_glcm_features(img)
        features.append(glcm_features)

    return np.array(features), y


if __name__ == "__main__":
    X_glcm, y = extract_glcm_dataset()

    print("✅ GLCM FEATURE EXTRACTION COMPLETED")
    print("Feature Vector Shape:", X_glcm.shape)
    print("Labels Shape:", y.shape)
