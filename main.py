import os, shutil, base64, re, gzip, subprocess
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# App setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "/tmp/uploads"
FASTSURFER_INPUT = os.path.expanduser("~/fastsurfer-input")
FASTSURFER_OUTPUT = os.path.expanduser("~/fastsurfer-output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FASTSURFER_INPUT, exist_ok=True)
os.makedirs(FASTSURFER_OUTPUT, exist_ok=True)

# Run FastSurfer
def run_fastsurfer(nifti_path: str, subject_id: str):
    compressed_path = os.path.join(FASTSURFER_INPUT, f"{subject_id}_T1w.nii.gz")
    if not nifti_path.endswith(".gz"):
        with open(nifti_path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(nifti_path, compressed_path)

    print(f"🧠 Running FastSurfer on: {compressed_path}")
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--user", "root",
        "-v", f"{FASTSURFER_INPUT}:/data:ro",
        "-v", f"{FASTSURFER_OUTPUT}:/output",
        "-e", "SUBJECTS_DIR=/output",
        "deepmi/fastsurfer:cu124-v2.3.3",
        "--t1", f"/data/{subject_id}_T1w.nii.gz",
        "--sid", subject_id,
        "--sd", "/output",
        "--parallel", "--seg_only", "--allow_root"
    ]
    subprocess.run(command, check=True)

# Parse FastSurfer stats
def parse_stats(subject_id):
    stats_dir = os.path.join(FASTSURFER_OUTPUT, subject_id, "stats")
    aseg = os.path.join(stats_dir, "aseg+DKT.stats")
    aparc = os.path.join(stats_dir, "lh.aparc.stats")
    metrics, thickness = {}, []

    if os.path.exists(aseg):
        with open(aseg) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 5: continue
                try:
                    metrics[parts[4]] = float(parts[3])
                except:
                    continue

    if os.path.exists(aparc):
        with open(aparc) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        thickness.append(float(parts[5]))
                    except:
                        continue

    lh = metrics.get("Left-Hippocampus", 0)
    rh = metrics.get("Right-Hippocampus", 0)
    lv = metrics.get("Left-Lateral-Ventricle", 0)
    rv = metrics.get("Right-Lateral-Ventricle", 0)

    return {
        "Left Hippocampus": lh,
        "Right Hippocampus": rh,
        "Asymmetry Index": round(abs(lh - rh) / max(lh, rh + 1e-6), 2),
        "Evans Index": round((lv + rv) / (lh + rh + 1e-6), 2),
        "Average Cortical Thickness": round(sum(thickness)/len(thickness), 2) if thickness else None
    }

# Stage classification
def predict_stage(mmse, cdr, adas):
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI"
    elif cdr == 0 and mmse >= 26:
        return "Normal"
    return "Uncertain"

# GPT summary
def generate_summary(biomarkers, mmse, cdr, adas):
    prompt = f"""Patient MRI & cognitive results:
- Left Hippocampus: {biomarkers['Left Hippocampus']} mm³
- Right Hippocampus: {biomarkers['Right Hippocampus']} mm³
- Asymmetry Index: {biomarkers['Asymmetry Index']}
- Evans Index: {biomarkers['Evans Index']}
- Cortical Thickness: {biomarkers['Average Cortical Thickness']} mm
- MMSE: {mmse}, CDR: {cdr}, ADAS-Cog: {adas}

Generate:
- Clinical interpretation
- Cognitive decline explanation
- Alzheimer's stage
- Next steps and recommendations
- Markdown format"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600
    )
    return response.choices[0].message.content.strip()

# PDF report
def clean_text(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r'[^\x00-\xff]', '', text)

def create_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    cleaned_text = clean_text(summary_text)
    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin1')

# Full analysis endpoint
@app.post("/analyze/")
async def full_analysis(file: UploadFile = File(...),
                        mmse: int = Form(...),
                        cdr: float = Form(...),
                        adas_cog: float = Form(...),
                        email: str = Form(...)):
    subject_id = "sub-01"
    nifti_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(nifti_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_fastsurfer(nifti_path, subject_id)
    biomarkers = parse_stats(subject_id)
    stage = predict_stage(mmse, cdr, adas_cog)
    summary = generate_summary(biomarkers, mmse, cdr, adas_cog)
    pdf = create_pdf(summary)

    seg_path = os.path.join(FASTSURFER_OUTPUT, subject_id, "mri", "aparc+aseg.png")
    seg_base64 = ""
    if os.path.exists(seg_path):
        with open(seg_path, "rb") as f:
            seg_base64 = base64.b64encode(f.read()).decode()

    return {
        "🧠 Clinical Biomarkers": biomarkers,
        "📈 Cognitive Scores": {"MMSE": mmse, "CDR": cdr, "ADAS-Cog": adas_cog},
        "🧬 Disease Stage": stage,
        "📋 GPT Summary": summary,
        "🧾 PDF Report": "✅ Generated",
        "🧠 Brain Segmentation Preview (base64)": seg_base64
    }

# Root endpoint
@app.get("/")
def root():
    return {"message": "✅ FastAPI backend ready for Alzheimer's MRI + Cognitive Analysis"}
