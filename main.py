import os
import shutil
import subprocess
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/uploads"
FASTSURFER_INPUT = os.path.expanduser("~/fastsurfer-input")
FASTSURFER_OUTPUT = os.path.expanduser("~/fastsurfer-output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FASTSURFER_INPUT, exist_ok=True)
os.makedirs(FASTSURFER_OUTPUT, exist_ok=True)

def run_fastsurfer(nifti_path: str, subject_id: str = "sub-01"):
    input_path = os.path.join(FASTSURFER_INPUT, f"{subject_id}_T1w.nii.gz")
    shutil.copy(nifti_path, input_path)

    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{FASTSURFER_INPUT}:/data:ro",
        "-v", f"{FASTSURFER_OUTPUT}:/output",
        "-e", "SUBJECTS_DIR=/output",
        "deepmi/fastsurfer:cu124-v2.3.3",
        "--t1", f"/data/{subject_id}_T1w.nii.gz",
        "--sid", subject_id,
        "--sd", "/output",
        "--parallel", "--seg_only", "--allow_root"
    ]
    subprocess.run(docker_cmd, check=True)

def parse_stats(subject_id):
    stats_dir = os.path.join(FASTSURFER_OUTPUT, subject_id, "stats")
    aseg_path = os.path.join(stats_dir, "aseg+DKT.stats")
    lh_aparc_path = os.path.join(stats_dir, "lh.aparc.stats")

    metrics = {
        "Left-Hippocampus": None,
        "Right-Hippocampus": None,
        "Left-Lateral-Ventricle": None,
        "Right-Lateral-Ventricle": None,
        "Left-Cerebral-White-Matter": None,
        "Right-Cerebral-White-Matter": None,
        "ctx-lh": None,
        "ctx-rh": None
    }

    with open(aseg_path) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            if len(parts) < 5: continue
            structure = parts[4]
            volume = float(parts[3])
            if structure in metrics:
                metrics[structure] = volume

    thickness = []
    if os.path.exists(lh_aparc_path):
        with open(lh_aparc_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        thickness.append(float(parts[5]))
                    except:
                        continue

    lh = metrics["Left-Hippocampus"]
    rh = metrics["Right-Hippocampus"]
    lv = metrics["Left-Lateral-Ventricle"]
    rv = metrics["Right-Lateral-Ventricle"]

    hip_asym = abs(lh - rh) / max(lh, rh) if lh and rh else None
    evans_index = (lv + rv) / (lh + rh + 1e-6) if lh and rh else None
    avg_thickness = round(sum(thickness) / len(thickness), 2) if thickness else None

    return {
        "Left Hippocampus": lh,
        "Right Hippocampus": rh,
        "Asymmetry Index": round(hip_asym, 2) if hip_asym else None,
        "Evans Index": round(evans_index, 2) if evans_index else None,
        "Left Cortex Volume": metrics["ctx-lh"],
        "Right Cortex Volume": metrics["ctx-rh"],
        "Left WM Volume": metrics["Left-Cerebral-White-Matter"],
        "Right WM Volume": metrics["Right-Cerebral-White-Matter"],
        "Average Cortical Thickness": avg_thickness
    }

def predict_stage(mmse: int, cdr: float, adas: float) -> str:
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI"
    elif cdr == 0 and mmse >= 26:
        return "Normal"
    else:
        return "Uncertain"

def send_summary_email(report):
    api_key = os.getenv("SENDGRID_API_KEY")
    doctor_email = os.getenv("DOCTOR_EMAIL", "doctor@example.com")

    if api_key:
        message = Mail(
            from_email="noreply@alzheimers.ai",
            to_emails=doctor_email,
            subject="🧠 Cognitive Test Results",
            html_content=report.replace("\n", "<br>")
        )
        try:
            sg = SendGridAPIClient(api_key)
            sg.send(message)
            print("✅ Email sent.")
        except Exception as e:
            print("❌ Email failed:", str(e))

class ScoreInput(BaseModel):
    mmse: int
    cdr: float
    adas_cog: float

@app.post("/analyze-mri/")
async def analyze_mri(file: UploadFile = File(...)):
    filename = os.path.join(UPLOAD_DIR, file.filename)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_fastsurfer(filename, subject_id="sub-01")
    biomarkers = parse_stats("sub-01")

    return {
        "🧠 Clinical Biomarkers": biomarkers,
        "message": "MRI scan complete. Please continue to cognitive testing."
    }

@app.post("/analyze-cognitive-score/")
async def analyze_cognitive_score(input: ScoreInput):
    mmse = input.mmse
    cdr = input.cdr
    adas = input.adas_cog
    stage = predict_stage(mmse, cdr, adas)

    report = (
        f"📝 Cognitive Test Summary:\n"
        f"- MMSE: {mmse}/30\n"
        f"- CDR: {cdr}\n"
        f"- ADAS-Cog: {adas}\n\n"
        f"🧬 Predicted Stage: {stage}"
    )

    send_summary_email(report)

    return {
        "mmse": mmse,
        "cdr": cdr,
        "adas": adas,
        "stage": stage,
        "report": report
    }

@app.get("/")
def root():
    return {"message": "✅ FastAPI MRI + Cognitive Test + Summary Report API"}

