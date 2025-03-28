import logging
import os
import shutil
import json
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

############################################################
# FASTAPI SETUP
############################################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)

############################################################
# OPENAI SETUP
############################################################
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

############################################################
# DOWNLOAD MODEL FROM PUBLIC S3 URL
############################################################
def download_model_zip():
    model_folder = "roberta_final_checkpoint"
    zip_path = "roberta_final_checkpoint.zip"

    logging.info("⬇️ Downloading model zip from public S3 URL...")
    url = "https://fastapi-app-bucket-varsh.s3.amazonaws.com/roberta_final_checkpoint.zip"
    urllib.request.urlretrieve(url, zip_path)

    logging.info("📦 Unzipping model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_folder)

    os.remove(zip_path)

    config_path = os.path.join(model_folder, "config.json")
    assert os.path.exists(config_path), "❌ config.json not found after unzip!"

# Run on startup
download_model_zip()

############################################################
# LOAD ROBERTA MODEL
############################################################
model_path = "roberta_final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_mapping = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

############################################################
# PARSE STATS FILE FROM FREESURFER
############################################################
def parse_aseg_stats(stats_path):
    if not os.path.exists(stats_path):
        logging.error(f"❌ .stats file not found: {stats_path}")
        return None

    left_vol = None
    right_vol = None
    with open(stats_path, "r") as f:
        for line in f:
            if "Left-Hippocampus" in line:
                left_vol = float(line.split()[3])
            if "Right-Hippocampus" in line:
                right_vol = float(line.split()[3])

    if left_vol is None or right_vol is None:
        logging.error("❌ HPC volumes not found in .stats file.")
        return None

    return {"Left": left_vol, "Right": right_vol}

############################################################
# HANDLE NIFTI FILES (.nii or .nii.gz)
############################################################
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
        logging.error(f"❌ Failed to parse NIfTI file: {str(e)}")
        return None

############################################################
# DETERMINE GROUND TRUTH RISK
############################################################
def compute_ground_truth_risk(volumes):
    avg_hippo = (volumes["Left"] + volumes["Right"]) / 2
    if avg_hippo >= 3400:
        return "Low Risk"
    elif 2900 <= avg_hippo < 3400:
        return "Medium Risk"
    else:
        return "High Risk"

############################################################
# CONVERT HPC => MMSE
############################################################
def compute_mmse_value(volumes):
    avg_v = (volumes["Left"] + volumes["Right"]) / 2
    if avg_v >= 3400:
        return 29
    elif 2900 <= avg_v < 3400:
        return 26
    elif 2500 <= avg_v < 2900:
        return 22
    else:
        return 20

############################################################
# CALCULATE ACCURACY
############################################################
def calculate_accuracy(predicted_risk, ground_truth_risk):
    return "100%" if predicted_risk == ground_truth_risk else "0%"

############################################################
# GENERATE MEDICAL REPORT USING OPENAI
############################################################
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

############################################################
# MAIN INFERENCE ROUTE
############################################################
@app.post("/process-mri/")
async def process_mri(file: UploadFile = File(...)):
    try:
        file_path = os.path.join("/home/ec2-user/uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"✅ File uploaded => {file_path}")

        if file.filename.endswith(".stats"):
            volumes = parse_aseg_stats(file_path)
        elif file.filename.endswith(".nii") or file.filename.endswith(".nii.gz"):
            volumes = handle_other_mri_formats(file_path)
        else:
            return {"detail": "Unsupported MRI file format. Only .stats, .nii, or .nii.gz are supported."}

        if not volumes:
            return {"detail": "MRI biomarker extraction failed."}

        ground_truth = compute_ground_truth_risk(volumes)
        mmse_val = compute_mmse_value(volumes)
        logging.info(f"Ground truth => {ground_truth}, mmse={mmse_val}")

        input_text = f"MMSE score is {mmse_val}."
        tokens = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**tokens).logits
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            predicted_idx = int(np.argmax(probs))
            predicted_risk = label_mapping[predicted_idx]

        accuracy = calculate_accuracy(predicted_risk, ground_truth)
        medical_report = generate_medical_report(volumes)

        return {
            "MRI Biomarkers": volumes,
            "MMSE Value": mmse_val,
            "Model Probabilities": {
                "Low Risk": round(float(probs[0]), 4),
                "Medium Risk": round(float(probs[1]), 4),
                "High Risk": round(float(probs[2]), 4)
            },
            "Predicted Risk": predicted_risk,
            "Ground Truth Risk": ground_truth,
            "Accuracy Score": accuracy,
            "Generated Medical Report": medical_report
        }

    except Exception as e:
        logging.error(f"Error processing file => {str(e)}", exc_info=True)
        return {"detail": f"Error => {str(e)}"}

############################################################
# TEST ROUTE
############################################################
@app.get("/")
def root():
    return {"message": "✅ RoBERTa MMSE Risk Prediction API is running! Supports .stats, .nii, and .nii.gz."}




