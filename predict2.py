# terminal_predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from utils import load_encoders_and_scaler
import tkinter as tk
from tkinter import filedialog


# ----------------------------
# Load models once
# ----------------------------
def load_models():
    soil_model = load_model("soil_model.h5")
    crop_model = joblib.load("crop_model.pkl")
    soil_encoder, crop_encoder, scaler = load_encoders_and_scaler()
    return soil_model, crop_model, soil_encoder, crop_encoder, scaler


# ----------------------------
# Predict Soil Type
# ----------------------------
def predict_soil_type(image_path, soil_model, soil_encoder, size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found.")

    img_resized = cv2.resize(img, size) / 255.0
    img_resized = np.expand_dims(img_resized.astype(np.float32), axis=0)

    pred = soil_model.predict(img_resized, verbose=0)
    predicted_class = np.argmax(pred, axis=1)
    return soil_encoder.inverse_transform(predicted_class)[0]


# ----------------------------
# Recommend Crops
# ----------------------------
def recommend_crops(
    N, P, K, pH, soil_type, crop_model, soil_encoder, crop_encoder, scaler
):
    soil_encoded = soil_encoder.transform([soil_type])[0]
    npk_ph_scaled = scaler.transform([[N, P, K, pH]])
    features = np.hstack((npk_ph_scaled, [[soil_encoded]])).astype(np.float32)

    prediction = crop_model.predict(features)
    return crop_encoder.inverse_transform(prediction)[0]


# ----------------------------
# Select File (terminal-friendly)
# ----------------------------
def select_image():
    root = tk.Tk()
    root.withdraw()  # hide tkinter main window
    file_path = filedialog.askopenfilename(
        title="Select Soil Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return file_path


# ----------------------------
# Main Runner
# ----------------------------
if __name__ == "__main__":
    soil_model, crop_model, soil_encoder, crop_encoder, scaler = load_models()

    print("\n🌱 Soil & Crop Recommendation Tool 🌱\n")

    # Browse image
    image_path = select_image()
    if not image_path:
        print("❌ No image selected. Exiting.")
        exit()

    try:
        soil_type = predict_soil_type(image_path, soil_model, soil_encoder)
        print(f"\n🧾 Predicted Soil Type: {soil_type}")
    except FileNotFoundError as e:
        print(e)
        exit()

    # Input values
    N = float(input("Enter Nitrogen (N): "))
    P = float(input("Enter Phosphorus (P): "))
    K = float(input("Enter Potassium (K): "))
    pH = float(input("Enter pH value: "))

    # Predict crop
    crop = recommend_crops(
        N, P, K, pH, soil_type, crop_model, soil_encoder, crop_encoder, scaler
    )
    print(f"\n✅ Recommended Crop: {crop}")
