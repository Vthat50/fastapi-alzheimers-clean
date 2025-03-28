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

# === Load OpenAI API key securely ===
OPENAI_API_KEY = os.environ.get("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")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY environment variable is not set!")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Download model from S3 ===
def download_model_zip():
    model_folder = "roberta_final_checkpoint"
    zip_path = "roberta_final_checkpoint.zip"
    url = "https://fastapi-app-bucket-varsh.s3.amazonaws.com/roberta_final_checkpoint.zip"

    if os.path.exists(model_folder):
        return

    logging.info("⬇️ Downloading model zip...")
    urllib.request.urlretrieve(url, zip_path)

    logging.info("📦 Unzipping model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tmp_unzip")
    os.remove(zip_path)

    for root, _, files in os.walk("tmp_unzip"):
        if "config.json" in files:
            shutil.move(root, model_folder)
            break
    shutil.rmtree("tmp_unzip", ignore_errors=True)

    config_path = os.path.join(model_folder, "config.json")
    assert os.path.exists(config_path), "❌ config.json not found after unzip!"

download_model_zip()

# === Load model ===
model_path = "roberta_final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# === Volume extraction ===
def parse_aseg_stats(path):
    left_vol = right_vol = None
    with open(path, "r") as f:
        for line in f:
            if "Left-Hippocampus" in line:
                left_vol = float(line.split()[3])
            if "Right-Hippocampus" in line:
                right_vol = float(line.split()[3])
    return {"Left": left_vol, "Right": right_vol} if left_vol and right_vol else None

def handle_other_mri_formats(path):
    try:
        img = nib.load(path)
        data = img.get_fdata()
        left_roi = data[30:50, 40:60, 20:40]
        right_roi = data[60:80, 40:60, 20:40]
        left_vol = np.sum(left_roi > np.percentile(left_roi, 95)) * np.prod(img.header.get_zooms())
        right_vol = np.sum(right_roi > np.percentile(right_roi, 95)) * np.prod(img.header.get_zooms())
        return {"Left": round(left_vol, 2), "Right": round(right_vol, 2)}
    except Exception as e:
        logging.error(f"❌ NIfTI parse failed: {e}")
        return None

# === Risk calculation ===
def compute_ground_truth_risk(vol):
    avg = (vol["Left"] + vol["Right"]) / 2
    return "Low Risk" if avg >= 3400 else "Medium Risk" if avg >= 2900 else "High Risk"

def compute_mmse_value(vol):
    avg = (vol["Left"] + vol["Right"]) / 2
    return 29 if avg >= 3400 else 26 if avg >= 2900 else 22 if avg >= 2500 else 20

def calculate_accuracy(pred, true):
    return "100%" if pred == true else "0%"

def generate_medical_report(vol):
    prompt = f"""
    You are an AI neurologist analyzing MRI biomarkers for Alzheimer's.

    **Left Volume:** {vol['Left']} mm³
    **Right Volume:** {vol['Right']} mm³

    Provide:
    - Risk level
    - Neurodegeneration signs
    - Progression timeline
    - Recommended treatments
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()

# === /process-mri/ ===
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

# === Root route ===
@app.get("/")
def root():
    return {"message": "✅ API is up. Upload .stats, .nii, or .nii.gz for MMSE risk prediction."}

