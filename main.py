import logging
import os
import shutil
import urllib.request
import zipfile
import torch
import numpy as np
import nibabel as nib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import sendgrid
from sendgrid.helpers.mail import Mail

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nurodot-com.webflow.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)

# === API Key Setup ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY missing")
if not SENDGRID_API_KEY:
    raise ValueError("❌ SENDGRID_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

# === Email: Send Test Link ===
def send_test_email(email: str, test_type: str):
    test_link = f"https://nurodot-com.webflow.io/{test_type.lower()}-form"
    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    message = Mail(
        from_email='noreply@alzapi.com',
        to_emails=email,
        subject=f"🧠 Your {test_type} Cognitive Test",
        html_content=f"""
        Hello,<br><br>
        Please complete your {test_type} test here:<br>
        <a href="{test_link}">{test_link}</a><br><br>
        After completion, return to the website to submit your score.<br><br>
        Your doctor will receive an analysis.<br><br>
        - Alzheimer's AI Assistant
        """
    )
    sg.send(message)

class EmailRequest(BaseModel):
    email: str
    test_type: str

@app.post("/send-cognitive-test/")
def send_cognitive_test(request: EmailRequest):
    if request.test_type not in ["MMSE", "MoCA"]:
        return {"error": "Invalid test type."}
    send_test_email(request.email, request.test_type)
    return {"message": f"✅ Sent {request.test_type} test to {request.email}"}

# === Download Model from S3 ===
def download_model_zip():
    model_folder = "roberta_final_checkpoint"
    zip_path = "roberta_final_checkpoint.zip"
    url = "https://fastapi-app-bucket-varsh.s3.amazonaws.com/roberta_final_checkpoint.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tmp_unzip")
    os.remove(zip_path)
    for root, _, files in os.walk("tmp_unzip"):
        if "config.json" in files:
            shutil.move(root, model_folder)
            break
    shutil.rmtree("tmp_unzip", ignore_errors=True)

download_model_zip()

# === Load Model ===
model_path = "roberta_final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
label_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# === MRI Processing ===
def parse_aseg_stats(path):
    left = right = None
    with open(path, "r") as f:
        for line in f:
            if "Left-Hippocampus" in line:
                left = float(line.split()[3])
            if "Right-Hippocampus" in line:
                right = float(line.split()[3])
    return {"Left": float(left), "Right": float(right)} if left and right else None

def handle_other_mri_formats(path):
    img = nib.load(path)
    data = img.get_fdata()
    left = np.sum(data[30:50, 40:60, 20:40] > np.percentile(data, 95)) * np.prod(img.header.get_zooms())
    right = np.sum(data[60:80, 40:60, 20:40] > np.percentile(data, 95)) * np.prod(img.header.get_zooms())
    return {"Left": float(round(left, 2)), "Right": float(round(right, 2))}

def compute_mmse(volumes):
    avg = (volumes["Left"] + volumes["Right"]) / 2
    return 29 if avg >= 3400 else 26 if avg >= 2900 else 22 if avg >= 2500 else 20

def risk_from_volumes(vol):
    avg = (vol["Left"] + vol["Right"]) / 2
    return "Low Risk" if avg >= 3400 else "Medium Risk" if avg >= 2900 else "High Risk"

def generate_medical_report(volumes):
    prompt = f"""
    A patient's hippocampal volumes:
    - Left: {volumes['Left']} mm³
    - Right: {volumes['Right']} mm³

    Please generate:
    - Risk assessment
    - Signs of degeneration
    - Treatment & follow-up
    - Suggested diagnostic tests
    - Markdown formatting
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
    )
    return response.choices[0].message.content.strip()

@app.post("/process-mri/")
async def process_mri(file: UploadFile = File(...)):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    volumes = parse_aseg_stats(path) if path.endswith(".stats") else handle_other_mri_formats(path)
    if not volumes:
        return {"detail": "Volume extraction failed"}
    mmse = compute_mmse(volumes)
    risk_true = risk_from_volumes(volumes)
    input_text = f"MMSE score is {mmse}."
    tokens = tokenizer(input_text, return_tensors="pt")
    logits = model(**tokens).logits
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    prediction = label_mapping[int(np.argmax(probs))]
    return {
        "MRI Biomarkers": volumes,
        "MMSE Value": float(mmse),
        "Predicted Risk": prediction,
        "Ground Truth Risk": risk_true,
        "Accuracy Score": "100%" if prediction == risk_true else "0%",
        "Model Probabilities": {
            "Low Risk": float(round(probs[0], 4)),
            "Medium Risk": float(round(probs[1], 4)),
            "High Risk": float(round(probs[2], 4))
        },
        "Generated Medical Report": generate_medical_report(volumes)
    }

# === Analyze MMSE / MoCA ===
class CognitiveTestInput(BaseModel):
    patient_name: str
    test_type: str
    score: int

@app.post("/analyze-cognitive-score/")
def analyze_cognitive_score(data: CognitiveTestInput):
    prompt = f"""
    Patient: {data.patient_name}
    Test: {data.test_type}
    Score: {data.score}

    Please provide:
    - What this score indicates
    - Affected cognitive domains
    - Follow-up recommendations
    - Alzheimer's relevance
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return {"report": response.choices[0].message.content.strip()}

@app.get("/")
def root():
    return {"message": "✅ FastAPI: MRI + MMSE/MoCA Cognitive Risk API Live"}

