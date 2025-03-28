import logging
import os
import shutil
import zipfile
import urllib.request
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)

# === OpenAI Setup (Hardcoded for now) ===
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# === Download model zip from S3 if needed ===
def download_model_zip():
    model_folder = "roberta_final_checkpoint"
    zip_path = "roberta_final_checkpoint.zip"

    logging.info("⬇️ Downloading model zip from public S3 URL...")
    url = "https://fastapi-app-bucket-varsh.s3.amazonaws.com/roberta_final_checkpoint.zip"
    urllib.request.urlretrieve(url, zip_path)

    logging.info("📦 Unzipping model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tmp_unzip")

    os.remove(zip_path)

    for root, dirs, files in os.walk("tmp_unzip"):
        if "config.json" in files:
            shutil.move(root, model_folder)
            break

    shutil.rmtree("tmp_unzip", ignore_errors=True)
    assert os.path.exists(os.path.join(model_folder, "config.json")), "❌ config.json not found after unzip!"

# === Download & Load Model ===
download_model_zip()
model_path = "roberta_final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# === Helper: Parse .stats ===
def parse_aseg_stats(path):
    left_vol = right_vol = None
    with open(path, "r") as f:
        for line in f:
            if "Left-Hippocampus" in line:
                left_vol = float(line.split()[3])
            if "Right-Hippocampus" in line:
                right_vol = float(line.split()[3])
    return {"Left": left_vol, "Right": right_vol} if left_vol and right_vol else None

# === Helper: Parse .nii / .nii.gz ===
def handle_other_mri_formats(path):
    try:
        img = nib.load(path)
        data = img.get_fdata()
        left_roi = data[30:50, 40:60, 20:40]
        right_roi = data[60:80, 40:60, 20:40]
        zoom = np.prod(img.header.get_zooms())
        left_vol = np.sum(left_roi > np.percentile(left_roi, 95)) * zoom
        right_vol = np.sum(right_roi > np.percentile(right_roi, 95)) * zoom
        return {"Left": round(left_vol, 2), "Right": round(right_vol, 2)}
    except Exception as e:
        logging.error(f"❌ NIfTI parse failed: {e}")
        return None

# === Risk Logic ===
def compute_ground_truth_risk(vol):
    avg = (vol["Left"] + vol["Right"]) / 2
    return "Low Risk" if avg >= 3400 else "Medium Risk" if avg >= 2900 else "High Risk"

def compute_mmse_value(vol):
    avg = (vol["Left"] + vol["Right"]) / 2
    return 29 if avg >= 3400 else 26 if avg >= 2900 else 22 if avg >= 2500 else 20

def calculate_accuracy(pred, actual):
    return "100%" if pred == actual else "0%"

def generate_medical_report(vol):
    prompt = f"""
    You are an AI neurologist analyzing MRI biomarkers for Alzheimer's.

    **Left Volume:** {vol['Left']} mm³  
    **Right Volume:** {vol['Right']} mm³  

    Provide:
    - Risk Level
    - Neurodegeneration signs
    - Progression timeline
    - Recommended treatments
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()

# === API Endpoint ===
@app.post("/process-mri/")
async def process_mri(file: UploadFile = File(...)):
    try:
        path = f"/tmp/{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if path.endswith(".stats"):
            volumes = parse_aseg_stats(path)
        elif path.endswith(".nii") or path.endswith(".nii.gz"):
            volumes = handle_other_mri_formats(path)
        else:
            return {"detail": "Unsupported format. Use .stats, .nii, or .nii.gz"}

        if not volumes:
            return {"detail": "Volume extraction failed."}

        mmse = compute_mmse_value(volumes)
        ground_truth = compute_ground_truth_risk(volumes)
        input_text = f"MMSE score is {mmse}."
        tokens = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**tokens).logits
            probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        predicted_risk = label_mapping[int(np.argmax(probs))]

        return {
            "MRI Biomarkers": volumes,
            "MMSE Value": mmse,
            "Model Probabilities": {
                "Low Risk": round(probs[0], 4),
                "Medium Risk": round(probs[1], 4),
                "High Risk": round(probs[2], 4)
            },
            "Predicted Risk": predicted_risk,
            "Ground Truth Risk": ground_truth,
            "Accuracy Score": calculate_accuracy(predicted_risk, ground_truth),
            "Generated Medical Report": generate_medical_report(volumes)
        }

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}", exc_info=True)
        return {"detail": f"Error => {str(e)}"}

# === Healthcheck Route ===
@app.get("/")
def root():
    return {"message": "✅ RoBERTa MMSE Risk Prediction API is running! Supports .stats, .nii, and .nii.gz."}


