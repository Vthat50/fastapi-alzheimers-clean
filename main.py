
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 🔁 Set this to your active ngrok tunnel exposing local FastAPI
LOCAL_BACKEND = "https://your-ngrok-url.ngrok-free.app"  # Replace with your live URL

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nurodot.webflow.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === CORS Preflight Endpoints ===
@app.options("/analyze-mri/")
async def preflight_analyze():
    return JSONResponse(status_code=200, content={"ok": True})

@app.options("/get-results")
async def preflight_results():
    return JSONResponse(status_code=200, content={"ok": True})

# === Proxy to Local Analyze MRI ===
@app.post("/analyze-mri/")
async def analyze_mri_proxy(file: UploadFile = File(...),
                            mmse: int = Form(...),
                            cdr: float = Form(...),
                            adas_cog: float = Form(...),
                            email: str = Form(...)):

    try:
        # Prepare file and form data
        files = {
            'file': (file.filename, await file.read(), file.content_type)
        }
        data = {
            'mmse': str(mmse),
            'cdr': str(cdr),
            'adas_cog': str(adas_cog),
            'email': email
        }

        # Proxy the request to your local backend
        response = requests.post(f"{LOCAL_BACKEND}/analyze-mri/", files=files, data=data)
        return response.json()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Proxy to Local Get Results ===
@app.post("/get-results")
async def get_results_proxy(mmse: int = Form(...),
                            cdr: float = Form(...),
                            adas_cog: float = Form(...),
                            email: str = Form(...)):

    try:
        data = {
            'mmse': str(mmse),
            'cdr': str(cdr),
            'adas_cog': str(adas_cog),
            'email': email
        }

        # Proxy the request to your local backend
        response = requests.post(f"{LOCAL_BACKEND}/get-results", data=data)
        return response.json()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Health Check ===
@app.get("/")
def health_check():
    return {"message": "✅ Render proxy API is up and forwarding to local FastSurfer backend"}
