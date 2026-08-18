// ==========================================
// MRI IMAGE ANALYSIS
// ==========================================

async function uploadImage() {

    const file = document.getElementById("imageInput").files[0];

    if (!file) {
        alert("Please upload an MRI image.");
        return;
    }

    // Show uploaded image
    document.getElementById("preview").src =
        URL.createObjectURL(file);

    let formData = new FormData();
    formData.append("file", file);

    try {

        const res = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await res.json();

        // Handle backend errors
        if (data.error) {

            alert("Backend Error: " + data.error);

            return;
        }

        // Display prediction
        document.getElementById("result").innerText =
            data.prediction;

        // Calculate confidence
        let conf = data.confidence * 100;

        document.getElementById("confidence-fill").style.width =
            conf + "%";

        document.getElementById("confidence-text").innerText =
            "Confidence: " + conf.toFixed(2) + "%";

        // Display Grad-CAM heatmap
        document.getElementById("heatmap").src =
            "data:image/png;base64," + data.heatmap;

    } catch (error) {

        console.error("Prediction Error:", error);

        alert(
            "Unable to connect to the backend. " +
            "Make sure Flask is running."
        );
    }
}


// ==========================================
// AI CHATBOT
// ==========================================

async function sendMessage() {

    const input = document.getElementById("chatInput");
    const message = input.value.trim();

    // Don't send empty messages
    if (!message) {
        return;
    }

    // Display user's message
    addChatMessage(
        message,
        "user"
    );

    // Clear input
    input.value = "";

    // Show temporary loading message
    const loadingMessage = addChatMessage(
        "Thinking...",
        "bot"
    );

    try {

        const res = await fetch(
            "http://127.0.0.1:5000/chat",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    message: message
                })
            }
        );

        const data = await res.json();

        // Remove "Thinking..."
        loadingMessage.remove();

        // Handle backend error
        if (data.error) {

            addChatMessage(
                "⚠️ " + data.error,
                "bot"
            );

            return;
        }

        // Display AI response
        addChatMessage(
            data.response,
            "bot"
        );

    } catch (error) {

        console.error("Chat Error:", error);

        // Remove loading message
        loadingMessage.remove();

        addChatMessage(
            "⚠️ Unable to connect to the AI assistant. " +
            "Please make sure the Flask backend is running.",
            "bot"
        );
    }
}


// ==========================================
// ADD MESSAGE TO CHAT
// ==========================================

function addChatMessage(message, sender) {

    const chatMessages =
        document.getElementById("chatMessages");

    const messageDiv =
        document.createElement("div");

    messageDiv.classList.add(
        "chat-message"
    );

    // User message
    if (sender === "user") {

        messageDiv.classList.add(
            "user-message"
        );

        messageDiv.innerHTML = `
            <strong>👤 You</strong>
            <p>${escapeHtml(message)}</p>
        `;

    }

    // AI message
    else {

        messageDiv.classList.add(
            "bot-message"
        );

        messageDiv.innerHTML = `
            <strong>🤖 AI Assistant</strong>
            <p>${escapeHtml(message)}</p>
        `;
    }

    chatMessages.appendChild(messageDiv);

    // Automatically scroll to newest message
    chatMessages.scrollTop =
        chatMessages.scrollHeight;

    return messageDiv;
}


// ==========================================
// SECURITY HELPER
// Prevent HTML injection in chat messages
// ==========================================

function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


// ==========================================
// SEND MESSAGE USING ENTER KEY
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const chatInput =
            document.getElementById("chatInput");

        if (chatInput) {

            chatInput.addEventListener(
                "keydown",
                function (event) {

                    if (event.key === "Enter") {

                        event.preventDefault();

                        sendMessage();
                    }

                }
            );
        }

    }
);