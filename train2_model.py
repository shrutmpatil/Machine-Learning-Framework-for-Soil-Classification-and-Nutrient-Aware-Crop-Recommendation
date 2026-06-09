from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import joblib
import numpy as np
import tensorflow as tf

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy("mixed_float16")


def train_soil_model(X_train_img, y_train_soil, X_test_img, y_test_soil, num_classes):
    """
    Trains the soil type prediction model using transfer learning with MobileNetV2.
    Optimized for high accuracy (>92%) with class weights for imbalance.
    """
    # Ensure correct dtype and normalization
    X_train_img = X_train_img.astype("float32") / 255.0
    X_test_img = X_test_img.astype("float32") / 255.0

    # Base model
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False  # freeze convolutional layers

    # Enhanced custom head for better feature extraction
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    soil_model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with lower learning rate
    soil_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Stronger data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        fill_mode="nearest",
    )

    # Compute class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_soil), y=y_train_soil
    )
    class_weights = dict(enumerate(class_weights))

    # Callbacks with learning rate scheduling
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint("best_soil_model.keras", save_best_only=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1
        ),
        # Cyclic learning rate (optional, uncomment if needed)
        # tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * (0.5 ** (epoch // 5))),
    ]

    # Initial training
    history = soil_model.fit(
        datagen.flow(X_train_img, y_train_soil, batch_size=32),
        validation_data=(X_test_img, y_test_soil),
        steps_per_epoch=len(X_train_img) // 32,
        epochs=10,  # reduced for efficiency
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning: unfreeze more layers for better accuracy
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    soil_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = soil_model.fit(
        datagen.flow(X_train_img, y_train_soil, batch_size=32),
        validation_data=(X_test_img, y_test_soil),
        steps_per_epoch=len(X_train_img) // 32,
        epochs=15,  # extended for better adaptation
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save models
    soil_model.save("soil_model.keras")
    soil_model.save("soil_model.h5")
    print("✅ Soil type model saved as 'soil_model.keras' and 'soil_model.h5'")

    # Evaluate with classification report
    y_pred = soil_model.predict(X_test_img)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("🌍 Classification Report:")
    print(
        classification_report(
            y_test_soil, y_pred_classes, target_names=soil_encoder.classes_
        )
    )

    return soil_model, history


def train_crop_model(X_train_feat, y_train_crops, X_test_feat, y_test_crops):
    """
    Trains the crop recommendation model using Random Forest.
    """
    crop_model = RandomForestClassifier(
        n_estimators=300,  # balanced for efficiency
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    crop_model.fit(X_train_feat, y_train_crops)

    train_acc = crop_model.score(X_train_feat, y_train_crops)
    test_acc = crop_model.score(X_test_feat, y_test_crops)
    print(f"🌱 Crop Train Acc: {train_acc:.2f}")
    print(f"🌱 Crop Test Acc: {test_acc:.2f}")

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

    # Train soil model
    num_classes = len(soil_encoder.classes_)
    soil_model, history_soil = train_soil_model(
        X_train_img, y_train_soil, X_test_img, y_test_soil, num_classes
    )

    # Evaluate soil model
    _, test_acc = soil_model.evaluate(X_test_img, y_test_soil, verbose=0)
    print(f"🌍 Soil Model Test Accuracy: {test_acc:.2f}")

    # Train crop model
    crop_model = train_crop_model(
        X_train_feat, y_train_crops, X_test_feat, y_test_crops
    )

    # Save encoders & scaler
    save_encoders_and_scaler(soil_encoder, crop_encoder, scaler)
