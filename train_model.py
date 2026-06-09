# train_models.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle


def train_soil_model(X_train_img, y_train_soil, X_test_img, y_test_soil, num_classes):
    """
    Trains the soil type prediction model using transfer learning with VGG16.
    """
    # Ensure correct dtype for TF
    X_train_img = X_train_img.astype("float32")
    X_test_img = X_test_img.astype("float32")

    # Use Transfer Learning with VGG16
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model layers

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    soil_model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    soil_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Train the model with data augmentation
    history = soil_model.fit(
        datagen.flow(X_train_img, y_train_soil, batch_size=32),
        validation_data=(X_test_img, y_test_soil),
        steps_per_epoch=len(X_train_img) // 32,
        epochs=10,
    )

    # Save the model
    soil_model.save("soil_model.h5")
    print("✅ Soil type model saved as 'soil_model.h5'")

    return soil_model, history


def train_crop_model(X_train_feat, y_train_crops):
    """
    Trains the crop recommendation model using Random Forest Classifier.
    """
    crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
    crop_model.fit(X_train_feat, y_train_crops)

    joblib.dump(crop_model, "crop_model.pkl")
    print("✅ Crop recommendation model saved as 'crop_model.pkl'")

    return crop_model


if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from data_collection import collect_data
    from utils import save_encoders_and_scaler

    # Collect & preprocess
    images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists = (
        collect_data()
    )
    data = preprocess_data(
        images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists
    )
    (
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
    ) = data

    # Train soil type model
    num_classes = len(soil_encoder.classes_)
    soil_model, history_soil = train_soil_model(
        X_train_img, y_train_soil, X_test_img, y_test_soil, num_classes
    )

    # Train crop recommendation model
    crop_model = train_crop_model(X_train_feat, y_train_crops)

    # Save encoders & scaler
    save_encoders_and_scaler(soil_encoder, crop_encoder, scaler)
