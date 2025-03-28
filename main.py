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

# === OPENAI Setup ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables.")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Model Download from Public S3 ===
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

    config_path = os.path.join(model_folder, "config.json")
    assert os.path.exists(config_path), "❌ config.json not found after unzip!"

download_model_zip()

# === Load Model ===
model_path = "roberta_final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# === MRI Volume Extraction ===
def parse_aseg_stats(stats_path):
    left_vol = right_vol = None
    with open(stats_path, "r") as f:
        for line in f:
            if "Left-Hippocampus" in line:
                left_vol = float(line.split()[3])
            if "Right-Hippocampus" in line:
                right_vol = float(line.split()[3])
    return {"Left": float(left_vol), "Right": float(right_vol)} if left_vol and right_vol else None

def handle_other_mri_formats(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        left_roi = data[30:50, 40:60, 20:40]
        right_roi = data[60:80, 40:60, 20:40]
        left_vol = np.sum(left_roi > np.percentile(left_roi, 95)) * np.prod(img.header.get_zooms())
        right_vol = np.sum(right_roi > np.percentile(right_roi, 95)) * np.prod(img.header.get_zooms())
        return {
            "Left": float(round(left_vol, 2)),
            "Right": float(round(right_vol, 2))
        }
    except Exception as e:
        logging.error(f"❌ NIfTI parse failed: {e}")
        return None

# === Risk Mapping ===
def compute_ground_truth_risk(volumes):
    avg = (volumes["Left"] + volumes["Right"]) / 2
    return "Low Risk" if avg >= 3400 else "Medium Risk" if avg >= 2900 else "High Risk"

def compute_mmse_value(volumes):
    avg = (volumes["Left"] + volumes["Right"]) / 2
    return 29 if avg >= 3400 else 26 if avg >= 2900 else 22 if avg >= 2500 else 20

def calculate_accuracy(pred, true):
    return "100%" if pred == true else "0%"

# === Medical Report with Diagnostics ===
def generate_medical_report(volumes):
    prompt = f"""
You are an expert AI neurologist assisting in Alzheimer's risk analysis.

A patient's MRI scan reveals hippocampal volumes as follows:
- Left Volume: {volumes['Left']} mm³
- Right Volume: {volumes['Right']} mm³

Based on these values, provide a comprehensive clinical report including the following:

---

### 🧠 1. Clinical Summary:
- Brief overview of the volumes and what they imply about hippocampal atrophy.
- Highlight any asymmetry or abnormalities.

### ⚠️ 2. Risk Level Assessment:
- Classify Alzheimer's risk: Low / Medium / High.
- Justify this risk score based on volume and typical clinical patterns.

### 🧬 3. Signs of Neurodegeneration:
- Describe what these hippocampal volumes suggest about the patient's current stage of neurodegeneration.
- Compare to typical atrophy levels in MCI vs. AD.

### 🗓️ 4. Progression Timeline:
- Estimate how advanced the disease may be.
- Mention if symptoms are likely to be mild, moderate, or severe.

### 💊 5. Recommended Medical Treatments:
- Provide a bulleted list of medications or medical interventions.
- Mention if combination therapies are advisable.

### 🧘‍♀️ 6. Lifestyle and Cognitive Care:
- Recommend lifestyle habits or routines that could help preserve cognitive function.

### 🧪 7. Recommended Diagnostic Tests:
- Suggest specific next-step diagnostic tests to confirm or rule out Alzheimer’s.
- This may include: 
  - PET scan (amyloid/tau)
  - CSF analysis
  - Neuropsychological cognitive assessments
  - Genetic testing (e.g., APOE ε4)
  - Repeat MRI in 6-12 months for atrophy monitoring

### 📋 8. Follow-up Plan:
- Describe how soon the patient should see a neurologist or memory care clinic.
- Suggest monitoring frequency or referral for comprehensive workup.

### ⚠️ 9. Clinical Disclaimer:
- Make it clear that this is an AI-generated summary and not a substitute for professional medical evaluation.

---

Format everything in professional **Markdown**. Use clear section headers and concise bullet points. Avoid jargon where possible.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
    )
    return response.choices[0].message.content.strip()

# === MRI Upload Endpoint ===
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
        logits = model(**tokens).logits
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        predicted_risk = label_mapping[int(np.argmax(probs))]

        return {
            "MRI Biomarkers": volumes,
            "MMSE Value": float(mmse),
            "Model Probabilities": {
                "Low Risk": float(round(probs[0], 4)),
                "Medium Risk": float(round(probs[1], 4)),
                "High Risk": float(round(probs[2], 4))
            },
            "Predicted Risk": predicted_risk,
            "Ground Truth Risk": ground_truth,
            "Accuracy Score": calculate_accuracy(predicted_risk, ground_truth),
            "Generated Medical Report": generate_medical_report(volumes)
        }

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}", exc_info=True)
        return {"detail": f"Error => {str(e)}"}

# === Cognitive Test Scoring Endpoint ===
class CognitiveTestInput(BaseModel):
    patient_name: str
    test_type: str  # MMSE or MoCA
    score: int

@app.post("/analyze-cognitive-score/")
def analyze_cognitive_score(data: CognitiveTestInput):
    prompt = f"""
    A patient named {data.patient_name} has completed a {data.test_type} test and scored {data.score}.

    Please analyze:
    1. What this score indicates in terms of cognitive health.
    2. What cognitive domains might be affected.
    3. Recommendations for next steps, further evaluation or follow-ups.
    4. What this may suggest in terms of Alzheimer's or related risk.

    Format the output like a medical summary with subheadings.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return {"report": response.choices[0].message.content.strip()}

# === Root Endpoint ===
@app.get("/")
def root():
    return {"message": "✅ Alzheimer's Risk Prediction API is running. MRI + Cognitive Test Ready."}
