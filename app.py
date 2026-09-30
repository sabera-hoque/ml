from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from health import get_health_page, get_health_status
from predictor import get_predict_page, predict_digit

app = FastAPI(title="ASD 4 ML Inference Demo", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== HEALTH CHECK ROUTES ====================

@app.get("/", response_class=HTMLResponse)
def health_page():
    """Health check page with system status and model information"""
    return get_health_page()


@app.get("/health")
def health_check():
    """Health check API endpoint"""
    return get_health_status()

# ==================== PREDICT ROUTES ====================

@app.get("/predict-page", response_class=HTMLResponse)
def predict_page():
    """Prediction page for uploading and predicting digits"""
    return get_predict_page()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict endpoint - accepts image file and returns digit prediction
    
    Parameters:
        file: Image file (PNG or JPEG)
        
    Returns:
        JSON with prediction, confidence, model info
    """
    # Validate file type
    if file.content_type not in {"image/png", "image/jpeg"}:
        from fastapi import HTTPException
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image")
    
    # Read file and predict
    image_data = await file.read()
    return await predict_digit(image_data)
