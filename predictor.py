from pathlib import Path
from PIL import Image
import numpy as np
import joblib
from io import BytesIO
from fastapi import HTTPException

# Load model
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

# Load navigation HTML
NAV_PATH = Path(__file__).parent / "static" / "nav.html"
with open(NAV_PATH, "r") as f:
    NAV_HTML = f.read()


def get_predict_page() -> str:
    """Returns the predict page HTML"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predict - ASD 4 ML Inference Demo</title>
        <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
        ___NAV_PLACEHOLDER___
        <div class="container">
            <h1>Digit Prediction</h1>
            <p class="subtitle">Upload an image to predict the digit (0-9)</p>
            
            <form id="uploadForm">
                <input type="file" id="imageFile" accept="image/png, image/jpeg" required>
                <button type="submit">Predict</button>
            </form>
            
            <div id="loading" class="loading">Processing image...</div>
            <div id="error" class="error"></div>
            <img id="previewImage" />
            <div id="result"></div>
        </div>

        <script>
            const form = document.getElementById('uploadForm');
            const fileInput = document.getElementById('imageFile');
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            const loadingDiv = document.getElementById('loading');
            const previewImage = document.getElementById('previewImage');
            
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        previewImage.src = event.target.result;
                        previewImage.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                resultDiv.style.display = 'none';
                errorDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    loadingDiv.style.display = 'none';
                    
                    if (!response.ok) {
                        const error = await response.json();
                        errorDiv.textContent = '❌ Error: ' + error.detail;
                        errorDiv.style.display = 'block';
                        return;
                    }
                    
                    const data = await response.json();
                    
                    resultDiv.innerHTML = `
                        <div class="result-card">
                            <div class="result-row">
                                <div>
                                    <h3>Predicted Digit</h3>
                                    <div class="prediction-value">${data.prediction}</div>
                                </div>
                                <div>
                                    <h3>Confidence</h3>
                                    <div class="confidence-value">${(data.confidence * 100).toFixed(2)}%</div>
                                </div>
                            </div>
                        </div>
                    `;
                    resultDiv.style.display = 'block';
                } catch (err) {
                    loadingDiv.style.display = 'none';
                    errorDiv.textContent = '❌ Error: ' + err.message;
                    errorDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return html.replace("___NAV_PLACEHOLDER___", NAV_HTML)


async def predict_digit(image_data: bytes) -> dict:
    """
    Predict digit from image data
    
    Args:
        image_data: Binary image data
        
    Returns:
        Dictionary with prediction, confidence, model info, and input shape
    """
    try:
        # Open and process image
        image = Image.open(BytesIO(image_data)).convert("L")
        image = image.resize((8, 8))
        pixels = 16 - (np.asarray(image, dtype=np.float32) / 255.0 * 16)
        features = pixels.reshape(1, -1)

        # Make prediction
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model": "logistic-regression-digits",
            "input_shape": [8, 8]
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
