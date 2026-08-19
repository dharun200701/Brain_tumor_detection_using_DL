# 🧠 Brain Tumor Detection & AI Assistant

An AI-powered web application for **brain tumor image analysis** with an integrated **AI chatbot assistant**. The application allows users to upload brain MRI images for tumor prediction and interact with an AI assistant for general information and guidance.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis or treatment.

## ✨ Features

* 🧠 **Brain Tumor Detection**

  * Upload a brain MRI image.
  * Analyze the uploaded image using the trained detection model.
  * Display the prediction/result through the web interface.

* 🤖 **AI Chatbot**

  * Integrated with the **Groq API**.
  * Provides AI-generated responses to user questions.
  * Designed to assist with general brain-tumor-related information.

* 🖼️ **Image Upload**

  * Supports uploading MRI images directly from the frontend.
  * Backend processes the uploaded image and returns the result.

* 🌐 **Web Interface**

  * Simple and responsive frontend.
  * Built using HTML, CSS, and JavaScript.

* 🔌 **Flask Backend**

  * REST API for frontend/backend communication.
  * Handles image uploads and AI chatbot requests.

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask
* Flask-CORS

### AI

* Brain tumor detection model
* Groq API
* Large Language Model (LLM)

### Other

* Git & GitHub
* REST API
* JSON

## 📁 Project Structure

```text
brain_tumor/
│
├── app.py                  # Flask backend application
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignored files
├── README.md               # Project documentation
│
└── frontend/
    ├── index.html          # Main web page
    ├── script.js           # Frontend JavaScript
    └── style.css           # Frontend styling
```

## ⚙️ Prerequisites

Make sure the following are installed:

* Python 3.9+
* Git
* A Groq API key

Check Python installation:

```bash
python --version
```

Check Git installation:

```bash
git --version
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd brain_tumor
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the required packages manually:

```bash
pip install flask flask-cors groq
```

## 🔐 Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

**Never upload your API key to GitHub.**

Make sure `.env` is included in `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
```

## ▶️ Running the Application

Start the Flask server:

```bash
python app.py
```

The backend will normally run at:

```text
http://127.0.0.1:5000
```

Open the frontend in your browser or access it through the Flask application, depending on how the project is configured.

## 🔄 Application Workflow

```text
User
 │
 ├── Upload MRI Image
 │        │
 │        ▼
 │   Flask Backend
 │        │
 │        ▼
 │   Tumor Detection Model
 │        │
 │        ▼
 │   Prediction Result
 │
 └── Ask AI Question
          │
          ▼
     Flask Backend
          │
          ▼
       Groq API
          │
          ▼
      AI Response
```

## 🧠 Brain Tumor Detection

The image-analysis component accepts an MRI image from the user and sends it to the backend for processing.

The backend then:

1. Receives the uploaded image.
2. Validates/processes the image.
3. Passes the image to the detection model.
4. Obtains the prediction.
5. Returns the result to the frontend.
6. Displays the prediction to the user.

## 🤖 AI Chatbot

The chatbot communicates with the Groq API through the Flask backend.

The general flow is:

```text
Frontend
   │
   │ User message
   ▼
Flask API
   │
   │ API request
   ▼
Groq
   │
   │ AI response
   ▼
Flask API
   │
   ▼
Frontend Chatbot
```

The Groq API key should remain on the **backend** and should never be exposed in frontend JavaScript.

## 🔗 API Endpoints

The exact endpoints depend on the current implementation in `app.py`.

Typical endpoints include:

| Method | Endpoint   | Purpose                                |
| ------ | ---------- | -------------------------------------- |
| `GET`  | `/`        | Load the application                   |
| `POST` | `/predict` | Upload MRI image and obtain prediction |
| `POST` | `/chat`    | Send a message to the AI chatbot       |

## 🧪 Testing

Start the backend:

```bash
python app.py
```

Then test the application by:

1. Opening the web interface.
2. Uploading a valid MRI image.
3. Checking the prediction result.
4. Sending a message through the chatbot.
5. Checking whether the AI response is displayed correctly.

## 🐛 Troubleshooting

### Flask server does not start

Check that the virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

### Groq chatbot is not responding

Check that your API key is correctly configured:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Also make sure the API key is not accidentally committed to GitHub.

### Image upload fails

Check:

* The selected file is a valid image.
* The frontend is sending the request to the correct backend endpoint.
* Flask is running.
* CORS configuration is correct if frontend and backend are running separately.

## 🔒 Security

* Never commit `.env` files.
* Never expose the Groq API key in frontend JavaScript.
* Do not upload private medical images to public repositories.
* Validate uploaded files on the backend.
* Use HTTPS when deploying the application publicly.

## ⚠️ Medical Disclaimer

This application is a **software/AI project for educational and research purposes**.

The predictions and chatbot responses should **not be considered medical advice, diagnosis, or treatment recommendations**. Always consult a qualified medical professional for medical decisions.

## 🔮 Future Improvements

* Improve tumor classification accuracy.
* Add confidence scores and visual explanations.
* Add Grad-CAM/heatmap visualization.
* Improve MRI preprocessing.
* Add support for additional tumor categories.
* Add user authentication.
* Store prediction history securely.
* Improve chatbot medical safety and response quality.
* Deploy the application to a cloud platform.
* Add automated model evaluation and testing.

## 👨‍💻 Development

To check the current Git status:

```bash
git status
```

Add changes:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Update brain tumor detection and chatbot"
```

Push changes:

```bash
git push origin main
```

## 📄 License

This project is intended for educational and research purposes. Add an appropriate open-source license if you plan to distribute the project publicly.

## ⭐ Acknowledgements

* Flask for the backend framework.
* Groq for the AI API.
* Open-source machine learning and medical-imaging resources used during development.

---

**Brain Tumor Detection & AI Assistant** — Combining medical image analysis with an AI-powered conversational assistant.
