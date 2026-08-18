import os
import io
import base64

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# Load variables from C:\HCL Project\.env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# INIT FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# GROQ AI CONFIGURATION
# ============================================================

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY was not found in .env")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq API configured successfully.")

# This model is available to your Groq API key.
GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LOAD BRAIN TUMOR MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "backend",
    "model",
    "brain_tumor_model.h5"
)

print("Loading brain tumor model from:")
print(MODEL_PATH)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Brain tumor model loaded successfully.")
except Exception as e:
    print("ERROR: Could not load brain tumor model.")
    print(str(e))
    raise


IMG_SIZE = 224


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer_name = None

for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    print("WARNING: No Conv2D layer was found for Grad-CAM.")
else:
    print("Using Grad-CAM layer:", last_conv_layer_name)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Resize MRI image and normalize pixel values.
    """

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ============================================================
# GRAD-CAM
# ============================================================

def get_gradcam(img_array):
    """
    Generate Grad-CAM heatmap for the MRI prediction.
    """

    if last_conv_layer_name is None:
        raise RuntimeError(
            "Grad-CAM cannot run because no Conv2D layer was found."
        )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # The model appears to be binary classification.
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients.")

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    if hasattr(heatmap, "numpy"):
        heatmap = heatmap.numpy()

    heatmap = np.maximum(heatmap, 0)

    max_value = np.max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE, IMG_SIZE)
    )

    return heatmap


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Brain Tumor Detection API Running",
        "groq_configured": groq_client is not None,
        "groq_model": GROQ_MODEL if groq_client else None
    })


# ============================================================
# MRI PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "No file selected"
            }), 400


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "error": "Uploaded file is empty"
            }), 400

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        img_array = preprocess_image(image)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

        prediction = float(prediction)

        print("Raw Prediction:", prediction)


        # ----------------------------------------------------
        # Validate prediction
        # ----------------------------------------------------

        if np.isnan(prediction):

            return jsonify({
                "error": "Model returned NaN prediction"
            }), 500


        # ----------------------------------------------------
        # Determine result
        # ----------------------------------------------------

        if prediction > 0.5:

            result = "Tumor"

            confidence = prediction

        else:

            result = "No Tumor"

            confidence = 1.0 - prediction


        confidence = float(confidence)


        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        heatmap = get_gradcam(img_array)


        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        original = np.array(
            image.resize((IMG_SIZE, IMG_SIZE))
        )


        # ----------------------------------------------------
        # Convert heatmap
        # ----------------------------------------------------

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        # ----------------------------------------------------
        # Combine original + heatmap
        # ----------------------------------------------------

        superimposed_img = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )


        # ----------------------------------------------------
        # Encode heatmap image as Base64
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".png",
            superimposed_img
        )

        if not success:

            return jsonify({
                "error": "Could not encode Grad-CAM image"
            }), 500

        heatmap_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")


        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return jsonify({

            "prediction": result,

            "confidence": confidence,

            "heatmap": heatmap_base64

        })


    except Exception as e:

        print("PREDICTION ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# GROQ CHATBOT ROUTE
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        # ----------------------------------------------------
        # Check Groq configuration
        # ----------------------------------------------------

        if groq_client is None:

            return jsonify({
                "error": "Groq API key is not configured. Check your .env file."
            }), 500


        # ----------------------------------------------------
        # Read request
        # ----------------------------------------------------

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "error": "Invalid or missing JSON request."
            }), 400


        # ----------------------------------------------------
        # Get user message
        # ----------------------------------------------------

        user_message = data.get("message", "")

        if not isinstance(user_message, str):

            return jsonify({
                "error": "Message must be a string."
            }), 400

        user_message = user_message.strip()


        if not user_message:

            return jsonify({
                "error": "Message cannot be empty."
            }), 400


        # ----------------------------------------------------
        # Get optional MRI prediction information
        # ----------------------------------------------------

        prediction = data.get("prediction")

        confidence = data.get("confidence")


        # ----------------------------------------------------
        # Build MRI context
        # ----------------------------------------------------

        context = ""

        if prediction:

            context += (
                "\nThe project's MRI classification model "
                "currently reports:\n"
                f"Prediction: {prediction}\n"
            )


        if confidence is not None:

            try:

                confidence_value = float(confidence)

                # Handle either 0.87 or 87 format.
                if confidence_value <= 1:

                    confidence_percentage = confidence_value * 100

                else:

                    confidence_percentage = confidence_value

                context += (
                    f"Model confidence: "
                    f"{confidence_percentage:.2f}%\n"
                )

            except (TypeError, ValueError):

                context += (
                    "Model confidence: unavailable\n"
                )


        # ----------------------------------------------------
        # Medical safety system prompt
        # ----------------------------------------------------

        system_prompt = """
You are an educational AI assistant integrated into a
brain tumor detection project.

Your role is to provide clear, understandable educational
information about brain tumors, MRI terminology, symptoms,
diagnostic concepts, and the project's AI prediction.

IMPORTANT MEDICAL SAFETY RULES:

1. You are an educational AI assistant, not a doctor.

2. Never claim that an AI prediction is a confirmed medical
   diagnosis.

3. Explain that the project's MRI model is a screening/
   classification tool and that its prediction should be
   reviewed by a qualified healthcare professional.

4. Do not invent MRI findings, symptoms, medical history,
   laboratory results, or patient information.

5. Do not claim that you personally examined an MRI unless
   the provided information actually contains those findings.

6. Do not prescribe medication or provide personalized
   treatment instructions.

7. For questions about diagnosis or treatment, provide
   general educational information and recommend consultation
   with an appropriate healthcare professional.

8. If a user describes potentially life-threatening or
   emergency symptoms, advise them to seek urgent medical
   attention.

9. Use simple language unless the user requests technical
   medical terminology.

10. If the user asks about the model's confidence, explain
    that confidence is a model output and is not equivalent
    to the probability that a patient truly has a tumor.

11. Do not make the user unnecessarily afraid.

12. Be concise but helpful.
"""


        # ----------------------------------------------------
        # User prompt
        # ----------------------------------------------------

        user_prompt = f"""
{context}

User question:
{user_message}
"""


        # ----------------------------------------------------
        # Call Groq
        # ----------------------------------------------------

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.3,

            max_tokens=500

        )


        # ----------------------------------------------------
        # Extract AI response
        # ----------------------------------------------------

        answer = response.choices[0].message.content


        if not answer:

            return jsonify({
                "error": "Groq returned an empty response."
            }), 500


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return jsonify({

            "response": answer,

            "model": GROQ_MODEL

        })


    except Exception as e:

        print("CHAT ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("============================================")
    print(" Brain Tumor Detection + AI Chatbot")
    print("============================================")
    print(f"Model: {MODEL_PATH}")
    print(f"Grad-CAM Layer: {last_conv_layer_name}")
    print(f"Groq Model: {GROQ_MODEL}")
    print(
        f"Groq Configured: "
        f"{groq_client is not None}"
    )
    print("============================================")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )