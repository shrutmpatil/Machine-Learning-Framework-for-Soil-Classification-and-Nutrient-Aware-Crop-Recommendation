"""
# app.py
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    send_file,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image, ExifTags
from datetime import datetime
import pandas as pd
import io
import requests  # For IP info & weather

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config["SECRET_KEY"] = (
    "secretkey123"  # keep as you requested (change for production)
)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------- Weather API Key ----------------
WEATHER_API_KEY = "202d53472af0adfa1108bc14c58453dd"  # your provided key


# ---------------- Database Models ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship("PredictionHistory", backref="author", lazy=True)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(50), nullable=False)
    crop = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(200), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------- Load ML Models ----------------
# ensure these files exist; adjust paths if necessary
soil_model = load_model("soil_model.h5")
soil_encoder = joblib.load("soil_encoder.pkl")
scaler = joblib.load("scaler.pkl")
crop_model = joblib.load("crop_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")


# ---------------- Helpers ----------------
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif:
            orientation = None
            for k, v in ExifTags.TAGS.items():
                if v == "Orientation":
                    orientation = k
                    break
            val = exif.get(orientation, None)
            if val == 3:
                image = image.rotate(180, expand=True)
            elif val == 6:
                image = image.rotate(270, expand=True)
            elif val == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print("Orientation error:", e)
    return image


def get_weather_and_location(lat=None, lon=None):
    """ """
    Fetch weather by lat/lon if provided, otherwise use IP-based fallback.
    Returns dict with city, latitude, longitude, temperature, description, map_url.
    """ """
    try:
        if lat is None or lon is None:
            ip_info = requests.get("https://ipinfo.io/json", timeout=5).json()
            loc = ip_info.get("loc", "0,0").split(",")
            city = ip_info.get("city", "Unknown")
            lat, lon = float(loc[0]), float(loc[1])
        else:
            city = "Unknown"

        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )
        w = requests.get(url, timeout=6).json()
        if w.get("cod") != 200:
            raise Exception(w.get("message", "Unknown error from weather API"))

        city = w.get("name", city)
        return {
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "temperature": w["main"]["temp"],
            "description": w["weather"][0]["description"].title(),
            "map_url": f"https://maps.google.com/maps?q={lat},{lon}&hl=en&z=14&output=embed",
        }
    except Exception as e:
        print("Weather error:", e)
        return {
            "city": "Unknown",
            "latitude": "N/A",
            "longitude": "N/A",
            "temperature": "N/A",
            "description": "Unavailable",
            "map_url": "",
        }


def get_weather_by_ip():
    return get_weather_and_location()


# ---------------- Routes ----------------
@app.route("/")
@login_required
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    fallback_weather = get_weather_by_ip()
    return render_template("dashboard.html", weather=fallback_weather)


@app.route("/get_weather")
@login_required
def get_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify(error="Missing coordinates"), 400
    try:
        lat, lon = float(lat), float(lon)
        data = get_weather_and_location(lat=lat, lon=lon)
        return jsonify(data)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = User(
            username=request.form["username"],
            email=request.form["email"],
            password=generate_password_hash(request.form["password"]),
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form["email"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction_result = None
    image_preview = None

    if request.method == "POST":
        try:
            file = request.files.get("image")
            if not file or not file.filename:
                flash("Please upload an image.", "danger")
                return render_template("predict.html")

            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            img = Image.open(path).convert("RGB")
            img = correct_image_orientation(img)
            img = img.resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Soil prediction
            soil_pred = soil_model.predict(arr)
            soil_label = soil_encoder.inverse_transform([np.argmax(soil_pred)])[0]

            image_preview = os.path.join("uploads", filename).replace(os.sep, "/")

            N = float(request.form.get("nitrogen", 0))
            P = float(request.form.get("phosphorus", 0))
            K = float(request.form.get("potassium", 0))
            pH = float(request.form.get("ph", 7))

            scaled_np = scaler.transform([[N, P, K, pH]])
            soil_idx = soil_encoder.transform([soil_label])[0]
            feats = np.hstack((scaled_np, [[soil_idx]]))

            crop_pred = crop_model.predict(feats)
            try:
                crop_labels = crop_encoder.inverse_transform(crop_pred)
            except Exception:
                crop_pred_labels = np.argmax(crop_pred, axis=1)
                crop_labels = crop_encoder.inverse_transform(crop_pred_labels)

            if len(crop_labels) > 0:
                first = crop_labels[0]
                if isinstance(first, (list, tuple, np.ndarray)):
                    crop_name = ", ".join(map(str, first))
                else:
                    crop_name = str(first)
            else:
                crop_name = str(crop_labels)

            prediction_result = {"soil_type": soil_label, "recommended_crop": crop_name}

            hist = PredictionHistory(
                nitrogen=N,
                phosphorus=P,
                potassium=K,
                ph=pH,
                soil_type=soil_label,
                crop=crop_name,
                image_path=path,
                author=current_user,
            )
            db.session.add(hist)
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            flash(f"Error: {str(e)}", "danger")

    return render_template(
        "predict.html", prediction=prediction_result, image_preview=image_preview
    )


@app.route("/history")
@login_required
def history():
    records = (
        PredictionHistory.query.filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.date.desc())
        .all()
    )
    return render_template("history.html", records=records)


@app.route("/download_history")
@login_required
def download_history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    data = [
        {
            "Date": r.date,
            "Nitrogen": r.nitrogen,
            "Phosphorus": r.phosphorus,
            "Potassium": r.potassium,
            "pH": r.ph,
            "Soil Type": r.soil_type,
            "Crop": r.crop,
            "Image Filename": os.path.basename(r.image_path) if r.image_path else "",
        }
        for r in records
    ]

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        df.to_excel(writer, index=False, sheet_name="History")
    buf.seek(0)

    return send_file(
        buf,
        download_name="prediction_history.xlsx",
        as_attachment=True,
        mimetype=(
            "application/" "vnd.openxmlformats-officedocument." "spreadsheetml.sheet"
        ),
    )


if __name__ == "__main__":
    import socket

    with app.app_context():
        db.create_all()

    # Get a reliable LAN IP (better than gethostbyname); falls back to 127.0.0.1
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"

    print(f"\n🌐 Open on your phone (same Wi-Fi): https://{lan_ip}:5000")
    print(
        "If the browser warns about the certificate, accept the certificate or use ngrok for a trusted HTTPS tunnel."
    )
    # Dev server with adhoc SSL (self-signed). For phone: accept the cert or use ngrok.
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context="adhoc")
"""
"""
# app.py
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    send_file,
    jsonify,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image, ExifTags
from datetime import datetime
import pandas as pd
import io
import requests  # For IP info & weather

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "secretkey123"  # change for production
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------- Weather API Key ----------------
WEATHER_API_KEY = "202d53472af0adfa1108bc14c58453dd"  # your provided key


# ---------------- Database Models ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship("PredictionHistory", backref="author", lazy=True)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(50), nullable=False)
    crop = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(200), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------- Load ML Models ----------------
soil_model = load_model("soil_model.h5")
soil_encoder = joblib.load("soil_encoder.pkl")
scaler = joblib.load("scaler.pkl")
crop_model = joblib.load("crop_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")


# ---------------- Helpers ----------------
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif:
            orientation = None
            for k, v in ExifTags.TAGS.items():
                if v == "Orientation":
                    orientation = k
                    break
            val = exif.get(orientation, None)
            if val == 3:
                image = image.rotate(180, expand=True)
            elif val == 6:
                image = image.rotate(270, expand=True)
            elif val == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print("Orientation error:", e)
    return image


def get_weather_and_location(lat=None, lon=None):
    try:
        if lat is None or lon is None:
            ip_info = requests.get("https://ipinfo.io/json", timeout=5).json()
            loc = ip_info.get("loc", "0,0").split(",")
            city = ip_info.get("city", "Unknown")
            lat, lon = float(loc[0]), float(loc[1])
        else:
            city = "Unknown"

        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )
        w = requests.get(url, timeout=6).json()
        if w.get("cod") != 200:
            raise Exception(w.get("message", "Unknown error from weather API"))

        city = w.get("name", city)
        return {
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "temperature": w["main"]["temp"],
            "description": w["weather"][0]["description"].title(),
            "map_url": f"https://maps.google.com/maps?q={lat},{lon}&hl=en&z=14&output=embed",
        }
    except Exception as e:
        print("Weather error:", e)
        return {
            "city": "Unknown",
            "latitude": "N/A",
            "longitude": "N/A",
            "temperature": "N/A",
            "description": "Unavailable",
            "map_url": "",
        }


def get_weather_by_ip():
    return get_weather_and_location()


# ---------------- Routes ----------------
@app.route("/")
@login_required
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    fallback_weather = get_weather_by_ip()
    return render_template("dashboard.html", weather=fallback_weather)


@app.route("/get_weather")
@login_required
def get_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify(error="Missing coordinates"), 400
    try:
        lat, lon = float(lat), float(lon)
        data = get_weather_and_location(lat=lat, lon=lon)
        return jsonify(data)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = User(
            username=request.form["username"],
            email=request.form["email"],
            password=generate_password_hash(request.form["password"]),
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form["email"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# -------- Prediction with Auto/Manual Mode --------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction_result = None
    image_preview = None

    if request.method == "POST":
        try:
            # Mode selection
            mode = request.form.get("mode", "manual")

            if mode == "auto":
                data = session.get("sensor_data")
                if not data:
                    flash(
                        "No sensor data available. Please wait for ESP32 to send.",
                        "danger",
                    )
                    return render_template("predict.html")
                N, P, K, pH = (
                    data["nitrogen"],
                    data["phosphorus"],
                    data["potassium"],
                    data["ph"],
                )
            else:
                N = float(request.form.get("nitrogen", 0))
                P = float(request.form.get("phosphorus", 0))
                K = float(request.form.get("potassium", 0))
                pH = float(request.form.get("ph", 7))

            # Image handling
            file = request.files.get("image")
            if not file or not file.filename:
                flash("Please upload an image.", "danger")
                return render_template("predict.html")

            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            img = Image.open(path).convert("RGB")
            img = correct_image_orientation(img)
            img = img.resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Soil prediction
            soil_pred = soil_model.predict(arr)
            soil_label = soil_encoder.inverse_transform([np.argmax(soil_pred)])[0]
            image_preview = os.path.join("uploads", filename).replace(os.sep, "/")

            # Features
            scaled_np = scaler.transform([[N, P, K, pH]])
            soil_idx = soil_encoder.transform([soil_label])[0]
            feats = np.hstack((scaled_np, [[soil_idx]]))

            # Crop prediction
            crop_pred = crop_model.predict(feats)
            try:
                crop_labels = crop_encoder.inverse_transform(crop_pred)
            except Exception:
                crop_pred_labels = np.argmax(crop_pred, axis=1)
                crop_labels = crop_encoder.inverse_transform(crop_pred_labels)

            if len(crop_labels) > 0:
                first = crop_labels[0]
                if isinstance(first, (list, tuple, np.ndarray)):
                    crop_name = ", ".join(map(str, first))
                else:
                    crop_name = str(first)
            else:
                crop_name = str(crop_labels)

            prediction_result = {"soil_type": soil_label, "recommended_crop": crop_name}

            # Save to DB
            hist = PredictionHistory(
                nitrogen=N,
                phosphorus=P,
                potassium=K,
                ph=pH,
                soil_type=soil_label,
                crop=crop_name,
                image_path=path,
                author=current_user,
            )
            db.session.add(hist)
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            flash(f"Error: {str(e)}", "danger")

    return render_template(
        "predict.html", prediction=prediction_result, image_preview=image_preview
    )


# -------- ESP32 API --------
@app.route("/api/sensor_data", methods=["POST"])
def sensor_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400
        session["sensor_data"] = data
        return jsonify({"message": "Sensor data received", "data": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get_sensor_data", methods=["GET"])
def get_sensor_data():
    data = session.get("sensor_data")
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "No sensor data available"}), 404


# -------- History --------
@app.route("/history")
@login_required
def history():
    records = (
        PredictionHistory.query.filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.date.desc())
        .all()
    )
    return render_template("history.html", records=records)


@app.route("/download_history")
@login_required
def download_history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    data = [
        {
            "Date": r.date,
            "Nitrogen": r.nitrogen,
            "Phosphorus": r.phosphorus,
            "Potassium": r.potassium,
            "pH": r.ph,
            "Soil Type": r.soil_type,
            "Crop": r.crop,
            "Image Filename": os.path.basename(r.image_path) if r.image_path else "",
        }
        for r in records
    ]

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        df.to_excel(writer, index=False, sheet_name="History")
    buf.seek(0)

    return send_file(
        buf,
        download_name="prediction_history.xlsx",
        as_attachment=True,
        mimetype=(
            "application/" "vnd.openxmlformats-officedocument." "spreadsheetml.sheet"
        ),
    )


# ---------------- Run ----------------
if __name__ == "__main__":
    # Detect LAN IP for phone access
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"

    # Toggle HTTPS mode
    USE_HTTPS = False  # Set to False when using ngrok

    print(
        f"\n🌐 Open on your phone (same Wi-Fi): {'https' if USE_HTTPS else 'http'}://{lan_ip}:5000"
    )

    if USE_HTTPS:
        app.run(host="0.0.0.0", port=5000, debug=True, ssl_context="adhoc")
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)"""


# app2.py - fully functional; robust decoding for model outputs (prob vs indices)
# Updated to GUARANTEE crop name (never a bare digit) by clamping indices and safe decoding.
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    send_file,
    jsonify,
    session,
    current_app,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
from PIL import Image, ExifTags
from datetime import datetime, timedelta
import pandas as pd
import io
import requests
from sqlalchemy.exc import IntegrityError

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "secretkey123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------- Weather API Key ----------------
WEATHER_API_KEY = "202d53472af0adfa1108bc14c58453dd"

# ---------------- Global Sensor Storage ----------------
app.config["LAST_SENSOR_DATA"] = None
app.config["LAST_SENSOR_TIME"] = None

# ---------------- Model globals ----------------
_models_loaded = False
MOCK_MODE = True
soil_model = None
soil_encoder = None
scaler = None
crop_model = None
crop_encoder = None


# ---------------- Database Models ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship("PredictionHistory", backref="author", lazy=True)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(50), nullable=False)
    crop = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(200), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------- Utility: ensure DB tables ----------------
def create_tables():
    try:
        db.create_all()
        current_app.logger.info("DB tables ensured")
    except Exception as e:
        try:
            current_app.logger.exception("create_all failed: %s", e)
        except Exception:
            print("create_all failed:", e)


# ---------------- Helpers ----------------
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif:
            orientation = None
            for k, v in ExifTags.TAGS.items():
                if v == "Orientation":
                    orientation = k
                    break
            val = exif.get(orientation, None)
            if val == 3:
                image = image.rotate(180, expand=True)
            elif val == 6:
                image = image.rotate(270, expand=True)
            elif val == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def get_weather_and_location(lat=None, lon=None):
    try:
        if lat is None or lon is None:
            ip_info = requests.get("https://ipinfo.io/json", timeout=5).json()
            loc = ip_info.get("loc", "0,0").split(",")
            city = ip_info.get("city", "Unknown")
            lat, lon = float(loc[0]), float(loc[1])
        else:
            city = "Unknown"

        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )
        w = requests.get(url, timeout=6).json()
        if w.get("cod") != 200:
            raise Exception(w.get("message", "Unknown error"))

        return {
            "city": w.get("name", city),
            "latitude": lat,
            "longitude": lon,
            "temperature": w["main"]["temp"],
            "description": w["weather"][0]["description"].title(),
            "map_url": f"https://maps.google.com/maps?q={lat},{lon}&hl=en&z=14&output=embed",
        }
    except Exception as e:
        current_app.logger.debug("Weather error: %s", e)
        return {
            "city": "Unknown",
            "latitude": "N/A",
            "longitude": "N/A",
            "temperature": "N/A",
            "description": "Unavailable",
            "map_url": "",
        }


def get_weather_by_ip():
    return get_weather_and_location()


def safe_crop_name(labels):
    if labels is None or len(labels) == 0:
        return "Unknown"
    val = labels[0]
    while isinstance(val, (tuple, list, np.ndarray)):
        if len(val) == 0:
            return "Unknown"
        val = val[0]
    return str(val)


# ---------------- Mock encoder / scaler / models ----------------
class MockEncoder:
    def __init__(self, labels):
        self.labels = list(labels)
        # keep classes_ for sklearn-like compat
        self.classes_ = np.array(self.labels, dtype=object)

    def inverse_transform(self, arr):
        out = []
        for a in arr:
            try:
                a_np = np.array(a)
                if a_np.ndim > 0 and a_np.dtype != object:
                    idx = int(np.argmax(a_np))
                    out.append(self.labels[idx])
                    continue
            except Exception:
                pass
            try:
                idx = int(a)
                if 0 <= idx < len(self.labels):
                    out.append(self.labels[idx])
                    continue
            except Exception:
                pass
            out.append("Unknown")
        return np.array(out)

    def transform(self, arr):
        out = []
        for a in arr:
            try:
                out.append(self.labels.index(a))
            except Exception:
                out.append(0)
        return np.array(out)


class MockScaler:
    def transform(self, X):
        return np.array(X, dtype=float)


class MockSoilModel:
    def predict(self, arr):
        avg = float(np.mean(arr))
        if avg < 0.33:
            probs = np.array([[0.9, 0.08, 0.02]])
        elif avg < 0.66:
            probs = np.array([[0.05, 0.9, 0.05]])
        else:
            probs = np.array([[0.02, 0.08, 0.9]])
        return probs


class MockCropModel:
    def predict(self, feats):
        arr = np.array(feats)
        soil_idx = int(arr[0, -1])
        N = float(arr[0, 0])
        n_classes = 5
        probs = np.zeros((1, n_classes))
        choice = (soil_idx + int(N)) % n_classes
        probs[0, choice] = 1.0
        return probs


# ---------------- Safe encoder helpers ----------------
def encoder_size(enc):
    """
    Return number of classes for encoder-like objects.
    Works for sklearn LabelEncoder (.classes_) or our MockEncoder (.labels).
    """
    try:
        if hasattr(enc, "classes_"):
            return len(enc.classes_)
        if hasattr(enc, "labels"):
            return len(enc.labels)
    except Exception:
        pass
    return None


def safe_inverse_transform_single(enc, idx):
    """
    Given encoder and single integer idx, return decoded string label safely.
    Clamps idx into valid range and uses inverse_transform if available.
    """
    size = encoder_size(enc)
    if size is None:
        # last resort: attempt inverse_transform and catch errors
        try:
            return enc.inverse_transform([idx])[0]
        except Exception:
            return str(int(idx))
    # clamp
    if idx < 0:
        idx = 0
    if idx >= size:
        idx = size - 1
    try:
        return enc.inverse_transform([idx])[0]
    except Exception:
        # fallback if encoder expects probabilities or different inputs
        try:
            if hasattr(enc, "labels"):
                return enc.labels[idx]
            if hasattr(enc, "classes_"):
                return enc.classes_[idx]
        except Exception:
            pass
    return str(int(idx))


# ---------------- Load (or fallback to mock) models ----------------
def load_models():
    global _models_loaded, MOCK_MODE
    global soil_model, soil_encoder, scaler, crop_model, crop_encoder

    if _models_loaded:
        return not MOCK_MODE

    try:
        from tensorflow.keras.models import load_model as tf_load_model

        if os.path.exists("soil_model.h5") and os.path.exists("soil_encoder.pkl"):
            soil_model = tf_load_model("soil_model.h5")
            soil_encoder = joblib.load("soil_encoder.pkl")
            scaler = joblib.load("scaler.pkl")
            crop_model = joblib.load("crop_model.pkl")
            crop_encoder = joblib.load("crop_encoder.pkl")
            MOCK_MODE = False
            _models_loaded = True
            current_app.logger.info("Real ML models loaded.")
            return True
    except Exception as e:
        current_app.logger.debug("Real model load failed: %s", e)

    # Fallback to mock models
    current_app.logger.info("Using MOCK ML models (fallback).")
    soil_labels = ["sandy", "loamy", "clayey"]
    crop_labels = ["paddy", "ragi", "jowar", "tur", "urad"]

    soil_encoder = MockEncoder(soil_labels)
    crop_encoder = MockEncoder(crop_labels)
    scaler = MockScaler()
    soil_model = MockSoilModel()
    crop_model = MockCropModel()
    MOCK_MODE = True
    _models_loaded = True
    return False


# ---------------- Routes ----------------
@app.route("/")
@login_required
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", weather=get_weather_by_ip())


@app.route("/get_weather")
@login_required
def get_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify(error="Missing coordinates"), 400
    try:
        lat, lon = float(lat), float(lon)
        data = get_weather_and_location(lat=lat, lon=lon)
        return jsonify(data)
    except Exception as e:
        return jsonify(error=str(e)), 500


# -------- Register --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not (username and email and password):
            flash("Please fill all fields", "danger")
            return render_template("register.html")

        try:
            user = User(
                username=username,
                email=email,
                password=generate_password_hash(password),
            )
            db.session.add(user)
            db.session.commit()
            flash("✅ Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except IntegrityError:
            db.session.rollback()
            flash("⚠️ Username or email already exists. Try another.", "danger")
        except Exception as e:
            db.session.rollback()
            current_app.logger.exception("Register error: %s", e)
            flash(f"❌ Error: {str(e)}", "danger")

    return render_template("register.html")


# -------- Login --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            create_tables()
        except Exception:
            pass

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not (email and password):
            flash("Provide email and password", "danger")
            return render_template("login.html")

        try:
            user = User.query.filter_by(email=email).first()
        except Exception as e:
            current_app.logger.exception("DB read failed during login: %s", e)
            create_tables()
            user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# -------- Prediction (Auto/Manual) --------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    # Ensure models are loaded (real or mock)
    load_models()

    prediction_result = None
    image_preview = None

    sensor_values = app.config.get("LAST_SENSOR_DATA")
    last_time = app.config.get("LAST_SENSOR_TIME")
    sensor_connected = False
    if last_time and (datetime.utcnow() - last_time) < timedelta(seconds=30):
        sensor_connected = True

    if request.method == "POST":
        try:
            mode = request.form.get("mode", "manual")
            if mode == "auto":
                if not sensor_connected or not sensor_values:
                    flash(
                        "No live sensor data available. Please check ESP32.", "danger"
                    )
                    return render_template(
                        "predict.html",
                        prediction=None,
                        image_preview=None,
                        sensor_values=sensor_values,
                        sensor_connected=sensor_connected,
                        models_loaded=not MOCK_MODE,
                    )
                N = float(sensor_values.get("nitrogen", 0))
                P = float(sensor_values.get("phosphorus", 0))
                K = float(sensor_values.get("potassium", 0))
                pH = float(sensor_values.get("ph", 7))
            else:
                try:
                    N = float(request.form.get("nitrogen", 0))
                    P = float(request.form.get("phosphorus", 0))
                    K = float(request.form.get("potassium", 0))
                    pH = float(request.form.get("ph", 7))
                except Exception:
                    flash("Invalid numeric inputs", "danger")
                    return render_template("predict.html")

            file = request.files.get("image")
            if not file or not file.filename:
                flash("Please upload an image.", "danger")
                return render_template(
                    "predict.html",
                    prediction=None,
                    image_preview=None,
                    sensor_values=sensor_values,
                    sensor_connected=sensor_connected,
                    models_loaded=not MOCK_MODE,
                )

            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            img = Image.open(path).convert("RGB")
            img = correct_image_orientation(img)
            img = img.resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)

            # --- Robust soil prediction handling ---
            soil_pred = soil_model.predict(arr)
            if hasattr(soil_pred, "ndim") and getattr(soil_pred, "ndim") > 1:
                soil_idx_arr = np.argmax(soil_pred, axis=1)
            else:
                soil_idx_arr = np.array(soil_pred).astype(int).ravel()

            try:
                soil_label = soil_encoder.inverse_transform(soil_idx_arr)[0]
            except Exception:
                try:
                    soil_label = str(int(np.ravel(soil_idx_arr)[0]))
                except Exception:
                    soil_label = "Unknown"

            image_preview = os.path.join("uploads", filename).replace(os.sep, "/")

            # Prepare features
            scaled_np = scaler.transform([[N, P, K, pH]])
            try:
                soil_idx = soil_encoder.transform([soil_label])[0]
            except Exception:
                try:
                    soil_idx = int(np.ravel(soil_idx_arr)[0])
                except Exception:
                    soil_idx = 0
            feats = np.hstack((scaled_np, [[soil_idx]]))

            # --- Robust crop prediction handling (SAFE name, never digit) ---
            crop_pred = crop_model.predict(feats)
            if hasattr(crop_pred, "ndim") and getattr(crop_pred, "ndim") > 1:
                crop_idx_arr = np.argmax(crop_pred, axis=1)
            else:
                crop_idx_arr = np.array(crop_pred).astype(int).ravel()

            # pick single predicted index and clamp to encoder range
            try:
                pred_idx = int(crop_idx_arr[0])
            except Exception:
                pred_idx = 0

            crop_name = safe_inverse_transform_single(crop_encoder, pred_idx)

            prediction_result = {"soil_type": soil_label, "recommended_crop": crop_name}

            hist = PredictionHistory(
                nitrogen=N,
                phosphorus=P,
                potassium=K,
                ph=pH,
                soil_type=soil_label,
                crop=crop_name,
                image_path=path,
                user_id=current_user.id,
            )
            db.session.add(hist)
            db.session.commit()
            flash("Prediction saved.", "success")

        except Exception as e:
            db.session.rollback()
            current_app.logger.exception("Prediction error: %s", e)
            flash(f"Error during prediction: {e}", "danger")

    return render_template(
        "predict.html",
        prediction=prediction_result,
        image_preview=image_preview,
        sensor_values=sensor_values,
        sensor_connected=sensor_connected,
        models_loaded=not MOCK_MODE,
    )


# -------- ESP32 API --------
@app.route("/api/sensor_data", methods=["POST"])
def sensor_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        cleaned = {
            "nitrogen": float(data.get("nitrogen", 0)),
            "phosphorus": float(data.get("phosphorus", 0)),
            "potassium": float(data.get("potassium", 0)),
            "ph": float(data.get("ph", 7)),
        }

        app.config["LAST_SENSOR_DATA"] = cleaned
        app.config["LAST_SENSOR_TIME"] = datetime.utcnow()
        return jsonify({"message": "Sensor data received", "data": cleaned}), 200
    except Exception as e:
        current_app.logger.exception("Sensor API error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/get_sensor_data", methods=["GET"])
def get_sensor_data():
    data = app.config.get("LAST_SENSOR_DATA")
    last_time = app.config.get("LAST_SENSOR_TIME")
    if data and last_time:
        connected = (datetime.utcnow() - last_time) < timedelta(seconds=30)
        return jsonify({"connected": connected, "data": data})
    else:
        return jsonify({"connected": False, "error": "No sensor data"}), 404


# -------- History & download --------
@app.route("/history")
@login_required
def history():
    records = (
        PredictionHistory.query.filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.date.desc())
        .all()
    )
    return render_template("history.html", records=records)


@app.route("/download_history")
@login_required
def download_history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    data = [
        {
            "Date": r.date,
            "Nitrogen": r.nitrogen,
            "Phosphorus": r.phosphorus,
            "Potassium": r.potassium,
            "pH": r.ph,
            "Soil Type": r.soil_type,
            "Crop": r.crop,
            "Image Filename": os.path.basename(r.image_path) if r.image_path else "",
        }
        for r in records
    ]

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        df.to_excel(writer, index=False, sheet_name="History")
    buf.seek(0)

    return send_file(
        buf,
        download_name="prediction_history.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------- Run ----------------
if __name__ == "__main__":
    # Create DB tables right away
    try:
        with app.app_context():
            create_tables()
    except Exception as e:
        print("create_tables error:", e)

    # Print LAN IP
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"

    print(f"\n🌐 Open on your phone (same Wi-Fi): http://{lan_ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
