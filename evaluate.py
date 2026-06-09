# evaluation_pipeline.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from predict import load_models
from data_collection import collect_data
from data_preprocessing import preprocess_data


def evaluate_soil_model(soil_model, X_test_img, y_test_soil, soil_encoder):
    # Evaluate soil model performance with detailed metrics and confusion matrix.
    print("\n--- Soil Type Classification Evaluation ---")
    y_pred_probs = soil_model.predict(X_test_img)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall metrics
    print("\nClassification Report (Soil Types):")
    print(
        classification_report(y_test_soil, y_pred, target_names=soil_encoder.classes_)
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test_soil, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=soil_encoder.classes_,
        yticklabels=soil_encoder.classes_,
    )
    plt.title("Soil Type Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def evaluate_crop_model(crop_model, X_test_feat, y_test_crops, crop_encoder):
    # Evaluate crop recommendation model performance.
    print("\n--- Crop Recommendation Evaluation ---")
    y_pred_bin = crop_model.predict(X_test_feat)

    # Classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_crops, y_pred_bin, average="micro"
    )
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # Feature importance plot
    if hasattr(crop_model, "feature_importances_"):
        importances = crop_model.feature_importances_
        feature_names = ["N", "P", "K", "pH", "Soil Type Encoded"]
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
        plt.title("Crop Model Feature Importance")
        plt.show()


if __name__ == "__main__":
    # Load models and encoders
    soil_model, crop_model, soil_encoder, crop_encoder, scaler = load_models()

    # Load and preprocess data
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
        _,
        _,
        _,
    ) = data

    # Evaluate both models
    evaluate_soil_model(soil_model, X_test_img, y_test_soil, soil_encoder)
    evaluate_crop_model(crop_model, X_test_feat, y_test_crops, crop_encoder)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load dataset
file_path = r"C:\\Users\\Durgesh\\Desktop\\TIH FINAL (IMP)\\synthetic_crop_13crops_500each(npk with image).csv"  # change if needed
df = pd.read_csv(file_path)


print("\n--- Basic Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

# 1. Duplicate rows check
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# 2. Constant values check
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Columns with constant value: {constant_cols}")

# 3. Correlation check
correlation = df.corr(numeric_only=True)
print("\n--- Correlation Matrix ---")
print(correlation)

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. Distribution check for each numeric column
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 5. Outlier detection using Z-score
outliers = {}
for col in numeric_cols:
    z_scores = abs(zscore(df[col]))
    outliers[col] = (z_scores > 3).sum()
print("\n--- Outlier Counts ---")
for col, count in outliers.items():
    print(f"{col}: {count} outliers")

# 6. Check unrealistic patterns
print("\nMin/Max values for each column:")
print(df[numeric_cols].agg(["min", "max"]))
