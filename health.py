from pathlib import Path
from fastapi.responses import HTMLResponse

# Load navigation HTML
NAV_PATH = Path(__file__).parent / "static" / "nav.html"
with open(NAV_PATH, "r") as f:
    NAV_HTML = f.read()


def get_health_page() -> str:
    """Returns the health check page HTML"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Check - ASD 4 ML Inference Demo</title>
        <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
        {NAV_HTML}
        <div class="container">
            <h1>Health Check</h1>
            <h2>System Status</h2>
            <div class="status-grid">
                <div class="status-card">
                    <h3>Status</h3>
                    <div class="value">✅ Healthy</div>
                </div>
                <div class="status-card">
                    <h3>Model</h3>
                    <div class="value">Ready</div>
                </div>
                <div class="status-card">
                    <h3>Input Shape</h3>
                    <div class="value">8x8</div>
                </div>
            </div>
            <h2>Model Information</h2>
            <table class="info-table">
                <tr>
                    <td>Model Type</td>
                    <td>Logistic Regression Digits Classifier</td>
                </tr>
                <tr>
                    <td>Input Shape</td>
                    <td>[8, 8] pixels</td>
                </tr>
                <tr>
                    <td>Output</td>
                    <td>Digit prediction (0-9)</td>
                </tr>
                <tr>
                    <td>Status</td>
                    <td>✅ Ready for predictions</td>
                </tr>
                <tr>
                    <td>Version</td>
                    <td>1.0.0</td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """
    return html


def get_health_status() -> dict:
    """Returns the health status as JSON"""
    return {"status": "healthy", "model": "digits-classifier"}
