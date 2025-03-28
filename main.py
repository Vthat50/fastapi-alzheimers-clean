import logging
import os
import shutil
import urllib.request
import zipfile
import pandas as pd
import torch
import numpy as np
import nibabel as nib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def download_model_zip():
    model_folder = "roberta_final_checkpoint"
    zip_path = "roberta_final_checkpoint.zip"
    if not os.path.exists(os.path.join(model_folder, "config.json")):
        logging.info("⬇️ Downloading model zip from public S3 URL...")
        url = "https://fastapi-app-bucket-varsh.s3.amazonaws.com/roberta_final_checkpoint.zip"
        urllib.request.urlretrieve(url, zip_path)

        logging.info("📦 Unzipping model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_folder)
        os.remove(zip_path)

download_model_zip()

model_path = "roberta_final_checkpoint"
assert os.path.exists(os.path.join(model_path, "config.json")), "❌ config.json not found!"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

def parse_aseg_stats(stats_path):
    if not os.path.exists(stats_path):
        return None
    left_vol, right_vol = None, None
    with open(stats_path) as f:
        for line in f:
            if "Left-Hippocampus" in line:
                left_vol = float(line.split()[3])
            elif "Right-Hippocampus" in line:
                right_vol = float(line.split()[3])
    return {"Left": left_vol, "Right": right_vol} if left_vol and right_vol else None

def handle_other_mri_formats(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        left_roi = data[30:50, 40:60, 20:40]
        right_roi = data[60:80, 40:60, 20:40]
        left_vol = float(np.sum(left_roi > np.percentile(left_roi, 95))) * np.prod(img.header.get_zooms())
        right_vol = float(np.sum(right_roi > np.percentile(right_roi, 95))) * np.prod(img.header.get_zooms())
        return {"Left": round(left_vol, 2), "Right": round(right_vol, 2)}
    except Exception as e:
        logging.error(f"Failed to parse NIfTI: {e}")
        return None

def compute_ground_truth_risk(volumes):
    avg = (volumes["Left"] + volumes["Right"]) / 2
    if avg >= 3400:
        return "Low Risk"
    elif avg >= 2900:
        return "Medium Risk"
    return "High Risk"

def compute_mmse_value(volumes):
    avg = (volumes["Left"] + volumes["Right"]) / 2
    if avg >= 3400:
        return 29
    elif avg >= 2900:
        return 26
    elif avg >= 2500:
        return 22
    return 20

def calculate_accuracy(pred, truth):
    return "100%" if pred == truth else "0%"

def generate_medical_report(volumes):
    prompt = f"""
You are an AI neurologist analyzing MRI biomarkers for Alzheimer's disease.

**Patient MRI Biomarkers:**
- Left Hippocampal Volume: {volumes['Left']} mm³
- Right Hippocampal Volume: {volumes['Right']} mm³

Generate a structured report with:
- Risk Level
- Key Neurodegeneration Indicators
- Estimated Disease Progression Timeline
- Treatment Recommendations
"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

@app.post("/process-mri/")
async def process_mri(file: UploadFile = File(...)):
    try:
        upload_dir = "/tmp/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith(".stats"):
            volumes = parse_aseg_stats(file_path)
        elif file.filename.endswith(".nii") or file.filename.endswith(".nii.gz"):
            volumes = handle_other_mri_formats(file_path)
        else:
            return {"detail": "Unsupported file format."}

        if not volumes:
            return {"detail": "Could not extract hippocampal volumes."}

        mmse_val = compute_mmse_value(volumes)
        input_text = f"MMSE score is {mmse_val}."
        tokens = tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            logits = model(**tokens).logits
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            predicted = label_mapping[int(np.argmax(probs))]

        ground_truth = compute_ground_truth_risk(volumes)
        accuracy = calculate_accuracy(predicted, ground_truth)
        report = generate_medical_report(volumes)

        return {
            "MRI Biomarkers": volumes,
            "MMSE Value": mmse_val,
            "Model Probabilities": {
                "Low Risk": round(float(probs[0]), 4),
                "Medium Risk": round(float(probs[1]), 4),
                "High Risk": round(float(probs[2]), 4),
            },
            "Predicted Risk": predicted,
            "Ground Truth Risk": ground_truth,
            "Accuracy Score": accuracy,
            "Generated Medical Report": report
        }
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return {"detail": str(e)}

@app.get("/")
def root():
    return {"message": "✅ FastAPI Alzheimer's Risk API is running."}



