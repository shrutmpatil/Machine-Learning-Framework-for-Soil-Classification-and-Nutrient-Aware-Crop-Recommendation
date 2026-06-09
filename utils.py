# utils.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


def save_encoders_and_scaler(soil_encoder, crop_encoder, scaler):
    """
    Saves the soil encoder, crop encoder, and scaler to disk.
    """
    joblib.dump(soil_encoder, "soil_encoder.pkl")
    joblib.dump(crop_encoder, "crop_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Encoders and scaler saved successfully.")


def load_encoders_and_scaler():
    """
    Loads the soil encoder, crop encoder, and scaler from disk.
    """
    try:
        soil_encoder = joblib.load("soil_encoder.pkl")
        crop_encoder = joblib.load("crop_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        print("✅ Encoders and scaler loaded successfully.")
        return soil_encoder, crop_encoder, scaler
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error: {e}. Ensure 'soil_encoder.pkl', 'crop_encoder.pkl', and 'scaler.pkl' exist."
        )


def preprocess_images(images, size=(224, 224)):
    """
    Preprocess soil images (resize + normalize for VGG16).
    """
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        processed_images.append(img_array)
    return np.vstack(processed_images)


def plot_training_history(history):
    """
    Plots training and validation accuracy/loss curves.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()
