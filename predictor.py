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


def _preprocess_digit_image(image: Image.Image) -> np.ndarray:
    """Convert uploaded image to the same 8x8 feature space used during training."""
    # Match sklearn digits training data: 8x8 grayscale values in [0, 16]
    image = image.convert("L")
    image = image.resize((8, 8), Image.Resampling.LANCZOS)

    # Threshold to a clean black-digit-on-white-background representation.
    # This keeps the input consistent with the scikit-learn digits dataset.
    image = image.point(lambda p: 255 if p > 200 else 0)
    pixels = np.asarray(image, dtype=np.float32) / 255.0 * 16.0
    features = pixels.reshape(1, -1)

    # Reject blank or near-constant images. They collapse to a single class and
    # produce bogus confidence values like 4 / 100%.
    if np.allclose(features, features[0, 0]) or np.std(features) < 0.1:
        raise ValueError("Image does not contain a recognizable digit")

    return features


def get_predict_page() -> str:
    """Returns the predict page HTML"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predict - ASD 4 ML Inference Demo</title>
        <link rel="stylesheet" href="/static/styles.css">
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
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
        image = Image.open(BytesIO(image_data))
        features = _preprocess_digit_image(image)

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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
