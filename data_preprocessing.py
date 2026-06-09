# data_preprocessing.py

import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from data_collection import collect_data


def preprocess_images(image_list, size=(224, 224)):
    """
    Preprocesses images by resizing and normalizing.

    Parameters:
        image_list (list): List of images to preprocess.
        size (tuple): Desired image size.

    Returns:
        np.array: Array of processed images (float32).
    """
    processed_images = []
    for img in image_list:
        # Resize and normalize images
        resized_img = cv2.resize(img, size)
        normalized_img = resized_img.astype("float32") / 255.0  # force float32
        processed_images.append(normalized_img)

    arr = np.array(processed_images, dtype="float32")  # force float32 array
    print(f"✅ Preprocessed images: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def preprocess_data(
    images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists
):
    """
    Preprocesses data and splits it into training and testing sets.

    Returns:
        Data splits and encoders/scalers.
    """
    # Image preprocessing
    X_images = preprocess_images(images_list, size=(224, 224))

    # Encode soil types
    soil_encoder = LabelEncoder()
    y_soil = soil_encoder.fit_transform(soil_types)

    # Prepare NPK and pH values
    X_npk_ph = np.array(
        list(zip(N_values, P_values, K_values, pH_values)), dtype="float32"
    )

    # Scale NPK and pH values
    scaler = StandardScaler()
    X_npk_ph_scaled = scaler.fit_transform(X_npk_ph)

    # Encode crop recommendations
    crop_encoder = MultiLabelBinarizer()
    y_crops = crop_encoder.fit_transform(crop_lists)

    # Combine features for crop recommendation model
    X_features = np.hstack((X_npk_ph_scaled, y_soil.reshape(-1, 1))).astype("float32")

    # Split data for soil type prediction
    X_train_img, X_test_img, y_train_soil, y_test_soil = train_test_split(
        X_images, y_soil, test_size=0.2, random_state=42, stratify=y_soil
    )

    # Split data for crop recommendation
    X_train_feat, X_test_feat, y_train_crops, y_test_crops = train_test_split(
        X_features, y_crops, test_size=0.2, random_state=42
    )

    print("✅ Preprocessing complete. Data is ready for training.")
    print(f"  Soil images train: {X_train_img.shape}, test: {X_test_img.shape}")
    print(f"  Crop features train: {X_train_feat.shape}, test: {X_test_feat.shape}")

    return (
        X_train_img,
        X_test_img,
        y_train_soil,
        y_test_soil,
        X_train_feat,
        X_test_feat,
        y_train_crops,
        y_test_crops,
        soil_encoder,
        crop_encoder,
        scaler,
    )


if __name__ == "__main__":
    # Collect the data
    images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists = (
        collect_data()
    )

    # Preprocess the data
    data = preprocess_data(
        images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists
    )
