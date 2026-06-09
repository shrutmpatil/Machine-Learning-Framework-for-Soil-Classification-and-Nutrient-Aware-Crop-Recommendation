# 🌱 AgriSense — Hyper-Local Two-Stage ML Framework for Soil Classification & Crop Recommendation

> **Research Paper:** *A Hyper-Local Two-Stage Machine Learning Framework for Soil Classification and Nutrient-Aware Crop Recommendation* — Published/Submitted to IEEE, 2025.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/datasets/0c3cf64c453e3435ea061649ca64f1567add80c818925f031a14403c5131c821)

---

## 📌 Overview

AgriSense is a full-stack precision agriculture system built around an IEEE-published two-stage multimodal ML pipeline that combines:

1. **Stage 1 — Soil Type Classification:** VGG16/MobileNetV2 CNN (transfer learning) on RGB soil images → classifies into Medium Black, Reddish Brown, or Saline soil.
2. **Stage 2 — Crop Recommendation:** Random Forest classifier using NPK + pH sensor values + predicted soil type → multi-label crop suitability predictions for 18 crop categories.

The system targets **hyper-local deployment** in the Palghar district of Maharashtra, India, and is designed for field use with low-cost NPK sensors, smartphones, and resource-constrained hardware.

**Key metrics (from paper):**
- Soil Classification Accuracy: **83.2%** (macro F1 = 0.83; 95% CI: [81.4, 84.9])
- Crop Recommendation Accuracy: **98.1%** (F1 = 0.982; 95% CI: [0.975, 0.988])
- Integrated Pipeline Effective F1: **~0.95** (accounting for error propagation)

---

## 👥 Authors

| Name | Department | Institute | Email |
|---|---|---|---|
| Sahil Rajadhyaksha | Electronics & Telecommunication | VIT Mumbai | sahil.rajadhyaksha@vit.edu.in |
| Durgesh Dere | Information Technology | VIT Mumbai | durgesh.dere@vit.edu.in |
| Preksha Koli | Biomedical Engineering | VIT Mumbai | preksha.koli@vit.edu.in |
| **Shrut Patil** | Electronics & Computer Science | VIT Mumbai | shrut.patil@vit.edu.in |
| Sanskruti Sankhe | Biomedical Engineering | VIT Mumbai | sanskruti.sankhe@vit.edu.in |
| Dr. Sheetal Mapare *(Guide)* | Electronics & Telecommunication | VIT Mumbai | sheetal.mapare@vit.edu.in |

*Supported by TIH-IoT, IIT Bombay under the Chanakya Fellowship.*

---

## 🗂️ Repository Structure

```
AgriSense/
│
├── 📄 app.py                    # Flask app (v1) — inline HTML template, soil + crop prediction
├── 📄 app2.py                   # Flask app (v2) — full login system, ESP32 API, weather, DB history
├── 📄 Webpage/
│   ├── app.py                   # Standalone Flask backend for Webpage UI
│   └── index.html               # Vanilla JS + Tailwind + Leaflet dashboard (no build step)
│
├── 📁 Frontend/                 # React + Vite frontend (modern SPA)
│   ├── src/
│   │   ├── App.jsx
│   │   ├── Home.jsx             # Main dashboard component
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── 📁 templates/                # Jinja2 HTML templates for app2.py (Bootstrap 4)
│   ├── base.html
│   ├── dashboard.html           # Live weather + GPS location
│   ├── predict.html             # Manual/Auto (ESP32) mode prediction UI
│   ├── history.html             # Prediction history + Excel export
│   ├── login.html
│   └── register.html
│
├── 📄 train_model.py            # VGG16-based soil model training
├── 📄 train2_model.py           # MobileNetV2-based soil model training (optimized)
├── 📄 data_collection.py        # Loads images + CSV (Palghar dataset)
├── 📄 data_preprocessing.py     # Image resize/normalize, label encoding, train/test split
├── 📄 predict.py                # CLI prediction with camera capture
├── 📄 predict2.py               # CLI prediction with file picker (tkinter)
├── 📄 evaluate.py               # Full evaluation: classification report, confusion matrix
├── 📄 confidence.py             # 95% CI bootstrap evaluation for all metrics
├── 📄 utils.py                  # save/load encoders, scaler, image preprocessing
├── 📄 DATAVISUAL.py             # Lab vs Sensor comparison bar charts (phosphate, nitrate, K)
│
├── 📄 soil_model.h5             # Trained soil CNN (not tracked — see .gitignore)
├── 📄 soil_encoder.pkl          # LabelEncoder for soil types
├── 📄 crop_encoder.pkl          # MultiLabelBinarizer for crop labels
├── 📄 scaler.pkl                # StandardScaler for NPK+pH features
├── 📄 crop_model.pkl            # Trained RandomForestClassifier
│
└── 📁 uploads/                  # Runtime image uploads (gitignored except .gitkeep)
```

---

## 🧠 ML Architecture

### Stage 1 — Soil Classification (CNN)

```
Input: RGB Soil Image (224×224×3)
       ↓
VGG16 / MobileNetV2 (ImageNet weights, frozen backbone)
       ↓
GlobalAveragePooling2D
       ↓
Dense(512, ReLU) → BatchNorm → Dropout(0.5)
       ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
       ↓
Dense(num_classes, Softmax)  → {Medium Black, Reddish Brown, Saline}
```

- Optimizer: Adam (lr=1e-4, fine-tune: 1e-5)
- Augmentation: rotation ±30°, zoom ±20%, flips, brightness [0.7–1.3]
- Regularization: Dropout, EarlyStopping, ReduceLROnPlateau

### Stage 2 — Crop Recommendation (Random Forest)

```
Input: [N_scaled, P_scaled, K_scaled, pH_scaled, soil_type_encoded]
       ↓
RandomForestClassifier(n_estimators=200–300, random_state=42)
       ↓
Multi-label crop predictions (18 crop categories)
```

**Feature Importance (from paper):**
| Feature | Importance |
|---|---|
| Potassium (K) | 0.25 |
| Nitrogen (N) | 0.24 |
| Phosphorus (P) | 0.24 |
| Soil Type (CNN output) | 0.17 |
| pH | 0.08 |

---

## 📊 Dataset

**SMART SOIL ANALYZER – VIT** (Self-curated, Palghar district, Maharashtra)

- **18,000 samples** total:
  - 9,000 RGB soil images (80% in-situ field collection, 20% public/Google images)
  - 9,000 sensor readings (NPK + pH + crop labels)
- **3 soil classes:** Medium Black Soil, Reddish Brown Soil, Saline Soil
- **18 crop categories:** cereals, pulses, fruits, cash crops
- **80/20 stratified train/test split**

📥 **Download:** [Kaggle Dataset](https://www.kaggle.com/datasets/0c3cf64c453e3435ea061649ca64f1567add80c818925f031a14403c5131c821)

Place the CSV at the path set in `data_collection.py`:
```python
data_file = r"path/to/synthetic_crop_13crops_500each(npk with image).csv"
image_dir = r"path/to/RESIZED_IMAGES"
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.9+
Node.js 20+ (for React frontend)
```

### 1. Clone & Install Python Dependencies

```bash
git clone https://github.com/your-username/agrisense.git
cd agrisense

pip install flask flask-cors flask-sqlalchemy flask-login \
    tensorflow scikit-learn joblib pillow pandas openpyxl \
    numpy requests werkzeug
```

### 2. Train the Models

```bash
# Download dataset first, update paths in data_collection.py, then:
python train_model.py        # VGG16 (standard)
# OR
python train2_model.py       # MobileNetV2 (optimized, mixed precision)
```

This produces: `soil_model.h5`, `soil_encoder.pkl`, `crop_encoder.pkl`, `scaler.pkl`, `crop_model.pkl`

### 3. Run the Flask App

**Option A — Full-featured app (recommended):**
```bash
python app2.py
# Open http://localhost:5000
```

**Option B — Minimal single-file app:**
```bash
python app.py
# Open http://localhost:5000
```

**Option C — Webpage standalone UI:**
```bash
cd Webpage
python app.py
# Open Webpage/index.html directly in browser
```

### 4. Run the React Frontend (Development)

```bash
cd Frontend
npm install
npm run dev
# Open http://localhost:5173
```

> **Note:** The React frontend (`Frontend/`) communicates with the Flask backend at `http://localhost:5000`. Ensure `app2.py` (or `Webpage/app.py`) is running before starting the React app.

---

## 🔑 API Keys Configuration

The following API keys need to be set before the app is fully functional:

### OpenWeatherMap (Weather data)

Used in `app2.py` and `Webpage/index.html`.

**`app2.py` (line ~47):**
```python
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
```

**`Webpage/index.html` (line ~184):**
```javascript
const apiKey = 'YOUR_OPENWEATHERMAP_API_KEY';
```

**`Frontend/src/Home.jsx` (line ~28):**
```javascript
`...&appid=YOUR_OPENWEATHERMAP_API_KEY`
```

Get a free key at: [https://openweathermap.org/api](https://openweathermap.org/api)

---

### ThingSpeak (ESP32 IoT sensor data — React frontend)

Used in `Frontend/src/Home.jsx`:
```javascript
`https://api.thingspeak.com/channels/YOUR_CHANNEL_ID/feeds/last.json?api_key=YOUR_READ_API_KEY`
```

Set up a free channel at: [https://thingspeak.com](https://thingspeak.com)

---

### NewsAPI (Agriculture news — Webpage UI)

Used in `Webpage/index.html` (line ~218):
```javascript
const apiKey = 'YOUR_NEWSAPI_KEY';
```

Get a free key at: [https://newsapi.org](https://newsapi.org)

---

## 🔌 ESP32 IoT Integration

`app2.py` exposes a REST API for real-time sensor data from an ESP32 or any IoT device:

**POST** `/api/sensor_data`
```json
{
  "nitrogen": 85.5,
  "phosphorus": 30.2,
  "potassium": 45.0,
  "ph": 6.8
}
```

**GET** `/api/get_sensor_data`
```json
{
  "connected": true,
  "data": { "nitrogen": 85.5, "phosphorus": 30.2, "potassium": 45.0, "ph": 6.8 }
}
```

Data is considered "live" for 30 seconds after last receipt. In the prediction UI, select **Auto (ESP32)** mode to use live sensor readings automatically.

---

## 📡 Application Features

### `app2.py` (Full Flask App)

| Feature | Details |
|---|---|
| User authentication | Register/Login with hashed passwords (Flask-Login + Werkzeug) |
| Soil image upload | Camera capture or file upload, EXIF orientation correction |
| Soil type prediction | VGG16/MobileNetV2 CNN inference |
| Crop recommendation | Random Forest with soil type + NPK pipeline |
| Manual & Auto mode | Type values manually or pull live from ESP32 |
| Weather & GPS | IP-based fallback + browser Geolocation API + OpenWeatherMap |
| Live map | Google Maps embed with GPS coordinates |
| Prediction history | SQLite-backed per-user history, downloadable as Excel |
| ESP32 API | POST/GET endpoints for real-time sensor integration |
| Mock model fallback | Runs without trained models for UI development/demo |

### React Frontend (`Frontend/`)

| Feature | Details |
|---|---|
| Dashboard | Live sensor cards, weather widget |
| Sensor Data tab | Real-time ThingSpeak feed |
| Crop Recommendation | Manual NPK form + one-click sensor import |
| History | Per-session prediction log |
| Farm Location | Placeholder (extendable with Leaflet) |
| Agriculture News | Placeholder (extendable with NewsAPI) |

### Webpage UI (`Webpage/`)

| Feature | Details |
|---|---|
| Weather | City-based OpenWeatherMap lookup |
| Soil photo | File upload or live camera capture |
| Manual/Sensor toggle | Radio-button data source selection |
| Recommendation | Flask `/recommend` with Chart.js NPK bar graph |
| History | Appended list from Flask `/history` |
| Village map | Leaflet.js OpenStreetMap embed |
| Agri news | NewsAPI feed |

---

## 📈 Model Evaluation

Run full evaluation (classification report + confusion matrix + feature importance):
```bash
python evaluate.py
```

Run bootstrap 95% confidence intervals:
```bash
python confidence.py
```

Run sensor data visualization (Lab vs Sensor bar charts):
```bash
python DATAVISUAL.py
```

---

## 📋 Results Summary (from IEEE Paper)

### Soil Classification (CNN — VGG16)

| Soil Type | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Medium Black Soil | 0.82 | 0.77 | 0.79 | 600 |
| Reddish Brown Soil | 0.75 | 0.88 | 0.81 | 600 |
| Saline Soil | **0.96** | 0.85 | **0.90** | 600 |
| **Overall Accuracy** | — | — | **0.83** | 1800 |

### Crop Recommendation (Random Forest)

| Metric | Value |
|---|---|
| Accuracy | **98.1%** |
| F1-Score | **0.982** |
| 95% CI | [0.975, 0.988] |

### Integrated Pipeline

| Metric | Value |
|---|---|
| Effective F1 (with error propagation) | **~0.95** |
| NPK Regression RMSE | N=6.2, P=5.7, K=4.9 |

---

## 🗃️ Database Schema (app2.py)

**Users table:**
```sql
id, username (unique), email (unique), password (hashed)
```

**PredictionHistory table:**
```sql
id, nitrogen, phosphorus, potassium, ph,
soil_type, crop, image_path, date, user_id (FK)
```

SQLite database stored at `instance/site.db`. Auto-created on first run.

---

## 🧪 CLI Tools

```bash
# Camera-based prediction (OpenCV)
python predict.py

# File picker prediction (tkinter dialog)
python predict2.py
```

Both tools load the saved models and run the full two-stage pipeline in the terminal.

---

## ⚙️ Configuration Reference

| Variable | File | Default | Description |
|---|---|---|---|
| `WEATHER_API_KEY` | `app2.py:47` | `"202d5..."` | OpenWeatherMap key |
| `SECRET_KEY` | `app2.py:36` | `"secretkey123"` | Flask session secret — **change in production** |
| `SQLALCHEMY_DATABASE_URI` | `app2.py:37` | `sqlite:///site.db` | Database URI |
| `UPLOAD_FOLDER` | `app2.py:38` | `static/uploads` | Image upload path |
| `data_file` | `data_collection.py:7` | Windows path | CSV dataset path |
| `image_dir` | `data_collection.py:8` | Windows path | Image directory path |
| `USE_HTTPS` | `app2.py` (main block) | `False` | Toggle SSL (set `True` for phone access via ngrok) |

---

## 🔬 Research Context

This system addresses three core gaps identified in existing literature:

1. **Modality gap** — Prior work uses either soil images OR nutrient data, not both. This framework fuses both in a single pipeline.
2. **Generalization gap** — Existing datasets cover large geographic regions; this work uses a hyper-local Palghar-specific dataset of 18,000 samples.
3. **Deployment gap** — Complex spectroscopy or vision systems require expensive hardware; this system runs on a smartphone with a ₹500 NPK sensor.

The two-stage pipeline allows accurate soil classification (83% accuracy) to cascade into accurate crop recommendation (98.1% F1), with only a ~3% drop in the integrated pipeline F1, demonstrating robustness of the design.

---

## 📚 Citation

If you use this codebase or dataset in your research, please cite:

```bibtex
@inproceedings{agrisense2025,
  title     = {A Hyper-Local Two-Stage Machine Learning Framework for Soil Classification
               and Nutrient-Aware Crop Recommendation},
  author    = {Rajadhyaksha, Sahil and Dere, Durgesh and Koli, Preksha and
               Patil, Shrut and Sankhe, Sanskruti and Mapare, Sheetal},
  booktitle = {IEEE Conference Proceedings},
  year      = {2025},
  note      = {979-8-3315-5371-5/25/\$31.00 ©2025 IEEE}

}
```
Paper citation:
```
S. Rajadhyaksha, P. Koli, S. Sankhe, D. Dere, S. Patil and S. Mapare, "A Hyper-Local Two-Stage Machine Learning Framework for Soil Classification and NutrientAware Crop Recommendation," 2025 18th International Conference on Sensing Technology (ICST), Utsunomiya, Japan, 2025, pp. 1-6, doi: 10.1109/ICST66402.2025.11512485.
keywords: {Crops;Modeling;Soil;Printing;Nutrients;Convolutional neural networks;Machine learning;Random forests;Accuracy;Image sensors;Precision Agriculture;Crop Recommendation;Image processing;NPK Analysis;CNN;Random Forest},
```
Dataset citation:
```
D. Dere, "SMART SOIL ANALYAZER – VIT: A Soil-Dataset with NPK, pH, ImageData for
Precision Farming and Crop Prediction," Kaggle, 2025.
Available: https://www.kaggle.com/datasets/0c3cf64c453e3435ea061649ca64f1567add80c818925f031a14403c5131c821
```

---

## 🤝 Acknowledgements

- **Vidyalankar Institute of Technology (VIT), Mumbai** — institutional support and infrastructure
- **TIH-IoT, IIT Bombay** — funding under the Chanakya Fellowship
- **Dr. Sheetal Mapare** — project guide and technical advisor
- **Soham Pal** — support and guidance throughout the research journey

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

```
MIT License — Copyright (c) 2026 Shrut Mahendra Patil
```

---

## 🐛 Known Issues & Notes

- `soil_model.h5` is excluded from git (see `.gitignore`). Train locally or download pre-trained weights separately.
- `app2.py` includes a **mock model fallback** — if `.h5`/`.pkl` files are absent, it uses deterministic mock classes for UI development without requiring a GPU.
- The React frontend uses `lucide`'s `createIcons()` for icon rendering; ensure `lucide-react` is installed via `npm install`.
- Windows path separators in `data_collection.py` will need updating for Linux/macOS deployments.
- For phone access on the same Wi-Fi network, set `USE_HTTPS = False` in `app2.py` and open `http://<LAN_IP>:5000` on your phone.
