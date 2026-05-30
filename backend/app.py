import tensorflow as tf
import numpy as np
import cv2
import base64
import io

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ==============================
# INIT APP
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "model/brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

# ==============================
# FIND LAST CONV LAYER (AUTO FIX)
# ==============================
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

print("✅ Using Grad-CAM layer:", last_conv_layer_name)

# ==============================
# PREPROCESS IMAGE
# ==============================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# 🔥 GRAD-CAM FUNCTION
# ==============================
def get_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ✅ SAFE CONVERSION
    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else heatmap

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    return heatmap

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return "✅ Brain Tumor Detection API Running"

# ==============================
# 🔮 PREDICTION ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Preprocess
        img_array = preprocess_image(image)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        print("🔍 Raw Prediction:", prediction)

        if np.isnan(prediction):
            return jsonify({"error": "Model returned NaN prediction"})

        # Result + confidence
        if prediction > 0.5:
            result = "Tumor"
            confidence = float(prediction)
        else:
            result = "No Tumor"
            confidence = float(1 - prediction)

        # ==============================
        # 🔥 GRAD-CAM
        # ==============================
        heatmap = get_gradcam(img_array)

        original = np.array(image.resize((IMG_SIZE, IMG_SIZE)))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        # Encode image
        _, buffer = cv2.imencode(".png", superimposed_img)
        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "heatmap": heatmap_base64
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)})

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)