from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
import random
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for testing


# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect("agrisense.db")
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      time TEXT,
                      result TEXT)"""
        )
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")


init_db()

# Mock soil types and crop recommendation model
soil_types = ["Sandy", "Clay", "Loamy", "Silty"]
crops = ["Wheat", "Rice", "Maize", "Soybean", "Barley"]

# Train a simple RandomForest model
X_train = np.array(
    [
        [50, 20, 30, 6.5, 0],
        [80, 40, 50, 7.0, 1],
        [60, 30, 40, 6.8, 2],
        [70, 25, 35, 6.2, 3],
    ]
)
y_train = ["Wheat", "Rice", "Maize", "Soybean"]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


@app.route("/recognize-soil", methods=["POST"])
def recognize_soil():
    print("Received request for /recognize-soil")
    if "image" not in request.files:
        print("No image provided")
        return jsonify({"error": "No image provided"}), 400
    soil_type = random.choice(soil_types)  # Mock recognition
    print(f"Recognized soil type: {soil_type}")
    return jsonify({"type": soil_type})


@app.route("/sensor-data", methods=["GET"])
def get_sensor_data():
    print("Received request for /sensor-data")
    data = {
        "nitrogen": random.randint(40, 100),
        "phosphorus": random.randint(20, 60),
        "potassium": random.randint(30, 70),
        "ph": round(random.uniform(5.5, 7.5), 1),
    }
    print(f"Sensor data: {data}")
    return jsonify(data)


@app.route("/recommend", methods=["POST"])
def recommend_crop():
    print("Received request for /recommend")
    data = request.get_json()
    print(f"Request data: {data}")
    soil_type = data.get("soil_type")
    if not soil_type or soil_type == "--":
        print("Invalid soil type")
        return jsonify({"error": "Invalid soil type"}), 400

    try:
        soil_index = soil_types.index(soil_type)
    except ValueError:
        print(f"Unknown soil type: {soil_type}")
        return jsonify({"error": "Unknown soil type"}), 400

    try:
        nitrogen = float(data.get("nitrogen", 0))
        phosphorus = float(data.get("phosphorus", 0))
        potassium = float(data.get("potassium", 0))
        ph = float(data.get("ph", 0))
    except (ValueError, TypeError):
        print("Invalid input data")
        return jsonify({"error": "Invalid input data"}), 400

    X = np.array([[nitrogen, phosphorus, potassium, ph, soil_index]])
    predicted_crop = clf.predict(X)[0]
    print(f"Predicted crop: {predicted_crop}")

    # Save to history
    try:
        conn = sqlite3.connect("agrisense.db")
        c = conn.cursor()
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = f"Recommended {predicted_crop} for soil type {soil_type} (N: {nitrogen}, P: {phosphorus}, K: {potassium}, pH: {ph})"
        c.execute("INSERT INTO history (time, result) VALUES (?, ?)", (time, result))
        conn.commit()
        conn.close()
        print("History saved")
    except Exception as e:
        print(f"Failed to save history: {e}")

    return jsonify({"crop": predicted_crop})


@app.route("/history", methods=["GET"])
def get_history():
    print("Received request for /history")
    try:
        conn = sqlite3.connect("agrisense.db")
        c = conn.cursor()
        c.execute("SELECT time, result FROM history ORDER BY time DESC")
        history = [{"time": row[0], "result": row[1]} for row in c.fetchall()]
        conn.close()
        print(f"History retrieved: {history}")
        return jsonify(history)
    except Exception as e:
        print(f"Failed to retrieve history: {e}")
        return jsonify({"error": "Failed to load history"}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
