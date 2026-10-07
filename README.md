# ML Inference API — ASD 4 Application Project

A small Python machine-learning inference service demonstrating the integration of a trained ML model into a reusable FastAPI application. The service validates image input, performs preprocessing, runs model inference, and returns a prediction with a confidence score. It also includes a health endpoint, Docker containerisation, API documentation, and deployment considerations.

The project was developed as a practical demonstration of ML engineering skills relevant to AI/ML model integration and inference services.


## Why this project

This project is intentionally small and demonstrates core engineering concepts relevant to an AI/ML engineering role:

- loading a trained ML model
- validating API input
- preprocessing image data
- running model inference
- returning a prediction and confidence score
- exposing a health endpoint
- packaging the service with Docker

The project uses the public scikit-learn digits dataset. It is a technical demonstration, not a production or classified system.

## Architecture

```text
PNG/JPEG image
      |
      v
FastAPI /predict
      |
      v
Image validation + preprocessing
      |
      v
Trained Logistic Regression model
      |
      v
Prediction + confidence
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

The `/predict` endpoint accepts a PNG or JPEG image.

## Docker

Build:

```bash
docker build -t ml-inference-api .
```

Run:

```bash
docker run --rm -p 8000:8000 ml-inference-api
```

## Engineering considerations

For a production implementation, I would add automated tests, structured logging, authentication, request-size limits, model/version metadata, monitoring, dependency pinning, CI/CD and controlled model deployment.

## Project scope

ASD application information pack.
