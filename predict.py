# predict.py
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils import load_encoders_and_scaler, preprocess_images


def load_models():
    """
    Loads the trained soil model, crop model, and encoders/scaler.
    """
    soil_model = load_model("soil_model.h5")
    crop_model = joblib.load("crop_model.pkl")
    soil_encoder, crop_encoder, scaler = load_encoders_and_scaler()
    return soil_model, crop_model, soil_encoder, crop_encoder, scaler


def capture_image_from_camera():
    """
    Captures an image from the camera (press 's' to capture).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return None

    print("📸 Press 's' to capture image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1)
        if key == ord("s"):  # save on 's'
            print("✅ Image captured!")
            img = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return img


def predict_soil_type(image, soil_model, soil_encoder):
    """
    Predicts soil type from an image.
    """
    img_processed = preprocess_images([image], size=(224, 224))
    prediction = soil_model.predict(img_processed)
    predicted_class = np.argmax(prediction, axis=1)
    soil_type = soil_encoder.inverse_transform(predicted_class)
    return soil_type[0]


def recommend_crops(
    N, P, K, pH, soil_type_label, crop_model, soil_encoder, crop_encoder, scaler
):
    """
    Recommends crops based on NPK, pH, and soil type.
    """
    soil_type_encoded = soil_encoder.transform([soil_type_label])[0]
    npk_ph_scaled = scaler.transform([[N, P, K, pH]])
    input_features = np.hstack((npk_ph_scaled, [[soil_type_encoded]]))
    crop_prediction = crop_model.predict(input_features)
    recommended_crop = crop_encoder.inverse_transform(crop_prediction)
    return recommended_crop[0]


if __name__ == "__main__":
    soil_model, crop_model, soil_encoder, crop_encoder, scaler = load_models()

    # Capture soil image
    image = capture_image_from_camera()
    if image is None:
        exit()

    predicted_soil = predict_soil_type(image, soil_model, soil_encoder)
    print(f"🌍 Predicted Soil Type: {predicted_soil}")

    # Ask user for inputs
    input_N = float(input("Enter Nitrogen (N): "))
    input_P = float(input("Enter Phosphorus (P): "))
    input_K = float(input("Enter Potassium (K): "))
    input_pH = float(input("Enter pH: "))

    # Recommend crops
    recommended_crop = recommend_crops(
        input_N,
        input_P,
        input_K,
        input_pH,
        predicted_soil,
        crop_model,
        soil_encoder,
        crop_encoder,
        scaler,
    )
    print(f"🌱 Recommended Crop: {recommended_crop}")
