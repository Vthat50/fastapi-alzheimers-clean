
# main.py — hosted on Render
import os, base64, io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fpdf import FPDF
from openai import OpenAI
import requests

# === Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_FORWARD_URL = os.getenv("NGROK_FORWARD_URL")  # e.g. https://abc123.ngrok-free.app
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use Webflow domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...),
                  mmse: int = Form(...),
                  cdr: float = Form(...),
                  adas_cog: float = Form(...),
                  email: str = Form(...)):

    # 1. Save MRI file
    mri_path = f"/tmp/{file.filename}"
    with open(mri_path, "wb") as f:
        f.write(await file.read())

    # 2. Forward to LOCAL ngrok tunnel
    with open(mri_path, "rb") as f:
        files = {"file": (file.filename, f, file.content_type)}
        data = {"mmse": mmse, "cdr": cdr, "adas_cog": adas_cog, "email": email}
        try:
            r = requests.post(f"{NGROK_FORWARD_URL}/process/", files=files, data=data, timeout=600)
            return r.json()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "✅ Render backend ready. Connected to local processing."}
