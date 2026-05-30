async function uploadImage() {
    const file = document.getElementById("imageInput").files[0];

    if (!file) {
        alert("Upload image");
        return;
    }

    document.getElementById("preview").src = URL.createObjectURL(file);

    let formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    // 🔥 HANDLE ERRORS
    if (data.error) {
        alert("Backend Error: " + data.error);
        return;
    }

    document.getElementById("result").innerText = data.prediction;

    let conf = data.confidence * 100;

    document.getElementById("confidence-fill").style.width = conf + "%";
    document.getElementById("confidence-text").innerText =
        "Confidence: " + conf.toFixed(2) + "%";

    document.getElementById("heatmap").src =
        "data:image/png;base64," + data.heatmap;
}