# evaluate_confidence.py

import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample


# ------------------- CONFIDENCE INTERVAL FUNCTION -------------------
def bootstrap_confidence_interval(
    y_true, y_pred, metric, n_bootstraps=1000, alpha=0.05
):
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric(y_true[indices], y_pred[indices])
        scores.append(score)

    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), (lower, upper)


def print_confidence_intervals(y_true, y_pred, model_name="Model"):
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted"
        ),
        "Recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted"
        ),
        "F1-score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    }

    print(f"\n📌 Confidence Intervals (95%) - {model_name}:")
    for name, metric in metrics.items():
        mean_val, (low, high) = bootstrap_confidence_interval(y_true, y_pred, metric)
        print(f"{name}: {mean_val:.3f}, 95% CI = [{low:.3f}, {high:.3f}]")


# ------------------- MAIN -------------------
if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from data_collection import collect_data
    from utils import load_encoders_and_scaler

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

    # -------- Soil Model Evaluation --------
    soil_model = tf.keras.models.load_model("soil_model.h5")
    y_pred_probs_soil = soil_model.predict(X_test_img)
    y_pred_soil = np.argmax(y_pred_probs_soil, axis=1)

    print_confidence_intervals(y_test_soil, y_pred_soil, model_name="Soil Model")

    # -------- Crop Model Evaluation --------
    crop_model = joblib.load("crop_model.pkl")
    y_pred_crops = crop_model.predict(X_test_feat)

    print_confidence_intervals(y_test_crops, y_pred_crops, model_name="Crop Model")
