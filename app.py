"""from flask import Flask, request, render_template_string
import numpy as np
import joblib
from PIL import Image, ExifTags
from tensorflow.keras.models import load_model
import os
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Load Models ----------------
soil_model = load_model("soil_model.h5")
soil_encoder = joblib.load("soil_encoder.pkl")
scaler = joblib.load("scaler.pkl")
crop_model = joblib.load("crop_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")


# ---------------- Helper Function ----------------
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print("Orientation correction error:", e)
    return image


# ---------------- HTML Template ----------------
html_template = """ """""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Soil & Crop Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 10px; background-color: #f7f7f7; font-size: 14px; }
        .card { padding: 15px; margin-top: 15px; border-radius: 10px; box-shadow: 0px 0px 8px rgba(0,0,0,0.1);}
        video, canvas, img { max-width: 100%; border-radius: 10px; margin-bottom: 10px; }
        .btn-custom { margin: 5px 0; }
    </style>
</head>
<body>
<div class="container">
    <h3 class="text-center">Soil & Crop Predictor</h3>
    <div class="card">
        <form method="POST" enctype="multipart/form-data">
            <div class="mb-2 text-center">
                <video id="video" autoplay playsinline width="250"></video><br>
                <button type="button" class="btn btn-primary btn-sm btn-custom" id="captureBtn">Capture Image</button>
            </div>

            <div class="mb-2 text-center">
                <label>Or choose image from folder:</label><br>
                <input type="file" accept="image/*" id="fileInput" class="form-control form-control-sm">
            </div>

            <canvas id="canvas" style="display:none;"></canvas>
            <input type="hidden" name="image" id="capturedImage">

            <div class="mb-2 text-center">
                <img id="preview" src="{{ image_preview or '' }}" alt="Preview">
            </div>

            <div class="row mb-2">
                <div class="col">
                    <input type="text" name="nitrogen" placeholder="Nitrogen (N)" class="form-control form-control-sm" required>
                </div>
                <div class="col">
                    <input type="text" name="phosphorus" placeholder="Phosphorus (P)" class="form-control form-control-sm" required>
                </div>
            </div>
            <div class="row mb-2">
                <div class="col">
                    <input type="text" name="potassium" placeholder="Potassium (K)" class="form-control form-control-sm" required>
                </div>
                <div class="col">
                    <input type="text" name="ph" placeholder="pH" class="form-control form-control-sm" required>
                </div>
            </div>
            <button type="submit" class="btn btn-success btn-sm btn-block w-100">Predict</button>
        </form>
    </div>

    {% if prediction %}
    <div class="card mt-2">
        <h5>Prediction Result:</h5>
        {% if prediction.error %}
            <p class="text-danger">Error: {{ prediction.error }}</p>
        {% else %}
            <p><strong>Soil Type:</strong> {{ prediction.soil_type }}</p>
            <p><strong>Recommended Crop:</strong> {{ prediction.recommended_crop }}</p>
            {% if image_preview %}
                <img src="{{ image_preview }}" alt="Uploaded Image">
            {% endif %}
        {% endif %}
    </div>
    {% endif %}
</div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const capturedImageInput = document.getElementById('capturedImage');
const previewImg = document.getElementById('preview');
const fileInput = document.getElementById('fileInput');

// Camera
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { console.error("Camera error:", err); });

// Capture button
captureBtn.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg');
    capturedImageInput.value = dataURL;
    previewImg.src = dataURL;
});

// File picker
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
        capturedImageInput.value = ev.target.result;
        previewImg.src = ev.target.result;
    }
    reader.readAsDataURL(file);
});
</script>
</body>
</html>
""" """""


# ---------------- Flask Route ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    image_preview = None
    if request.method == "POST":
        try:
            img_data = request.form.get("image")
            if img_data:
                img_str = img_data.split(",")[1]
                img_bytes = io.BytesIO(base64.b64decode(img_str))
                img = Image.open(img_bytes).convert("RGB")
                img = correct_image_orientation(img)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                soil_pred = soil_model.predict(img_array)
                soil_label_str = soil_encoder.inverse_transform([np.argmax(soil_pred)])[
                    0
                ]
                soil_encoded = soil_encoder.transform([soil_label_str])[0]

                # Save the image to static folder for preview
                save_path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
                img.save(save_path)
                image_preview = save_path
            else:
                return render_template_string(
                    html_template, prediction={"error": "No image selected."}
                )

            # NPK + pH
            N = float(request.form["nitrogen"].replace(",", "."))
            P = float(request.form["phosphorus"].replace(",", "."))
            K = float(request.form["potassium"].replace(",", "."))
            pH = float(request.form["ph"].replace(",", "."))

            nutrients = np.array([[N, P, K, pH]])
            nutrients_scaled = scaler.transform(nutrients)
            features = np.hstack((nutrients_scaled, [[soil_encoded]]))

            crop_pred = crop_model.predict(features)
            crop_name = crop_encoder.inverse_transform(crop_pred)[0]

            prediction_result = {
                "soil_type": soil_label_str,
                "recommended_crop": crop_name,
            }

        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template_string(
        html_template, prediction=prediction_result, image_preview=image_preview
    )


# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)"""


from flask import Flask, request, render_template_string
import numpy as np
import joblib
from PIL import Image, ExifTags
from tensorflow.keras.models import load_model
import os
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Load Models ----------------
soil_model = load_model("soil_model.h5")
soil_encoder = joblib.load("soil_encoder.pkl")
scaler = joblib.load("scaler.pkl")
crop_model = joblib.load("crop_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")


# ---------------- Helper Function ----------------
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print("Orientation correction error:", e)
    return image


# ---------------- HTML Template ----------------
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Soil & Crop Predictor</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
    body { padding: 10px; background: #f0f0f0; font-size: 13px; }
    .card { padding: 10px; border-radius: 10px; margin-top: 10px; box-shadow: 0 0 5px rgba(0,0,0,0.1);}
    video, canvas { max-width: 100%; border-radius: 8px; margin-bottom: 5px; }
    img#preview, .output-image {
        max-width: 150px;
        max-height: 150px;
        border-radius: 5px;
        margin: 5px auto;
        display: block;
    }
    .btn-sm { font-size: 0.75rem; padding: 2px 5px; }
    input { font-size: 12px; padding: 3px; }
</style>
</head>
<body>
<div class="container">
<div class="card">
    <h5 class="text-center mb-2">Soil & Crop Predictor</h5>
    <form method="POST" enctype="multipart/form-data">
        <div class="text-center mb-2">
            <video id="video" autoplay playsinline width="200"></video><br>
            <button type="button" id="captureBtn" class="btn btn-primary btn-sm mt-1">Capture</button>
        </div>
        <div class="mb-2 text-center">
            <input type="file" accept="image/*" id="fileInput" class="form-control form-control-sm">
        </div>
        <canvas id="canvas" style="display:none;"></canvas>
        <input type="hidden" name="image" id="capturedImage">
        <div class="text-center mb-2">
            <img id="preview" src="{{ image_preview or '' }}" alt="Preview">
        </div>
        <div class="row g-1 mb-1">
            <div class="col"><input type="text" name="nitrogen" placeholder="N" class="form-control form-control-sm" required></div>
            <div class="col"><input type="text" name="phosphorus" placeholder="P" class="form-control form-control-sm" required></div>
        </div>
        <div class="row g-1 mb-1">
            <div class="col"><input type="text" name="potassium" placeholder="K" class="form-control form-control-sm" required></div>
            <div class="col"><input type="text" name="ph" placeholder="pH" class="form-control form-control-sm" required></div>
        </div>
        <button type="submit" class="btn btn-success btn-sm w-100">Predict</button>
    </form>
</div>

{% if prediction %}
<div class="card mt-1">
    <h6>Prediction:</h6>
    {% if prediction.error %}
        <p class="text-danger">{{ prediction.error }}</p>
    {% else %}
        <p><strong>Soil Type:</strong> {{ prediction.soil_type }}</p>
        <p><strong>Recommended Crop:</strong> {{ prediction.recommended_crop }}</p>
        {% if image_preview %}
            <img src="{{ image_preview }}" class="output-image" alt="Uploaded Image">
        {% endif %}
    {% endif %}
</div>
{% endif %}
</div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const capturedImageInput = document.getElementById('capturedImage');
const previewImg = document.getElementById('preview');
const fileInput = document.getElementById('fileInput');

// Camera setup
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { console.error(err); });

// Capture button
captureBtn.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg');
    capturedImageInput.value = dataURL;
    previewImg.src = dataURL;
});

// File input
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if(!file) return;
    const reader = new FileReader();
    reader.onload = function(ev){
        capturedImageInput.value = ev.target.result;
        previewImg.src = ev.target.result;
    }
    reader.readAsDataURL(file);
});
</script>
</body>
</html>
"""


# ---------------- Flask Route ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    image_preview = None
    if request.method == "POST":
        try:
            img_data = request.form.get("image")
            if img_data:
                img_str = img_data.split(",")[1]
                img_bytes = io.BytesIO(base64.b64decode(img_str))
                img = Image.open(img_bytes).convert("RGB")
                img = correct_image_orientation(img)
                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                soil_pred = soil_model.predict(img_array)
                soil_label_str = soil_encoder.inverse_transform([np.argmax(soil_pred)])[
                    0
                ]
                soil_encoded = soil_encoder.transform([soil_label_str])[0]

                # Save image for preview/output
                save_path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
                img.save(save_path)
                image_preview = save_path
            else:
                return render_template_string(
                    html_template, prediction={"error": "No image selected"}
                )

            # NPK + pH
            N = float(request.form["nitrogen"].replace(",", "."))
            P = float(request.form["phosphorus"].replace(",", "."))
            K = float(request.form["potassium"].replace(",", "."))
            pH = float(request.form["ph"].replace(",", "."))

            nutrients = np.array([[N, P, K, pH]])
            nutrients_scaled = scaler.transform(nutrients)
            features = np.hstack((nutrients_scaled, [[soil_encoded]]))

            crop_pred = crop_model.predict(features)
            crop_name = crop_encoder.inverse_transform(crop_pred)[0]

            prediction_result = {
                "soil_type": soil_label_str,
                "recommended_crop": crop_name,
            }

        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template_string(
        html_template, prediction=prediction_result, image_preview=image_preview
    )


# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
