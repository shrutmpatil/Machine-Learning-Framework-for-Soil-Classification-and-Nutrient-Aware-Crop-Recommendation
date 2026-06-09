import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Data from your tables
# -------------------------------

# Phosphate Data
phosphate_data = {
    "Sample": [f"S{i}" for i in range(1, 12)],  # S1, S2, S3...
    "Lab (kg/ha)": [
        13.20,
        11.23,
        8.50,
        17.00,
        9.75,
        10.00,
        9.50,
        12.50,
        16.00,
        14.23,
        11.20,
    ],
    "Sensor (kg/ha)": [
        12.90,
        11.00,
        8.45,
        16.87,
        9.51,
        9.99,
        9.54,
        12.42,
        6.99,
        6.23,
        4.80,
    ],
}

# Nitrate Data
nitrate_data = {
    "Sample": [f"S{i}" for i in range(1, 12)],
    "Lab (kg/ha)": [224, 200, 193, 225, 180, 228, 196, 234, 235, 211, 166],
    "Sensor (kg/ha)": [222, 197, 197, 224, 179, 225, 213, 233, 235, 210, 162],
}


# Potassium Data
potassium_data = {
    "Sample": [f"S{i}" for i in range(1, 12)],
    "Lab (kg/ha)": [265, 302, 278, 256, 356, 297, 354, 251, 304, 255, 268],
    "Sensor (kg/ha)": [263, 305, 281, 250, 351, 276, 354, 249, 300, 254, 265],
}


# -------------------------------
# Function for Joint Bar Graph
# -------------------------------
def plot_joint_bar(data, title, ylim=None):
    df = pd.DataFrame(data)
    x = np.arange(len(df["Sample"]))  # sample indices
    width = 0.35  # bar width

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["Lab (kg/ha)"], width, label="Lab Analysis")
    plt.bar(x + width / 2, df["Sensor (kg/ha)"], width, label="Sensor Reading")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Sample No.", fontsize=12)
    plt.ylabel("kg/ha", fontsize=12)
    plt.xticks(x, df["Sample"], rotation=45)  # nice labels S1, S2, ...
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()


# -------------------------------
# Generate Joint Bar Graphs
# -------------------------------
plot_joint_bar(phosphate_data, "Phosphate: Lab vs Sensor", ylim=(0, 20))
plot_joint_bar(nitrate_data, "Nitrate: Lab vs Sensor", ylim=(0, 250))
plot_joint_bar(potassium_data, "Potassium: Lab vs Sensor", ylim=(0, 400))
