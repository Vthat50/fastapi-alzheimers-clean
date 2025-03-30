import os, shutil, subprocess, base64, io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fpdf import FPDF
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

# === App Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

UPLOAD_DIR = "/tmp/uploads"
FASTSURFER_INPUT = os.path.expanduser("~/fastsurfer-input")
FASTSURFER_OUTPUT = os.path.expanduser("~/fastsurfer-output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FASTSURFER_INPUT, exist_ok=True)
os.makedirs(FASTSURFER_OUTPUT, exist_ok=True)

# === Run FastSurfer ===
def run_fastsurfer(nifti_path: str, subject_id: str):
    input_path = os.path.join(FASTSURFER_INPUT, f"{subject_id}_T1w.nii.gz")
    shutil.copy(nifti_path, input_path)
    subprocess.run([
        "docker", "run", "--rm", "--gpus", "all", "--user", "root",
        "-v", f"{FASTSURFER_INPUT}:/data:ro",
        "-v", f"{FASTSURFER_OUTPUT}:/output",
        "-e", "SUBJECTS_DIR=/output",
        "deepmi/fastsurfer:cu124-v2.3.3",
        "--t1", f"/data/{subject_id}_T1w.nii.gz",
        "--sid", subject_id,
        "--sd", "/output",
        "--parallel", "--seg_only", "--allow_root"
    ], check=True)

# === Parse biomarkers ===
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
                label = parts[4]
                try: volume = float(parts[3])
                except: continue
                metrics[label] = volume

    if os.path.exists(aparc):
        with open(aparc) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) >= 6:
                    try: thickness.append(float(parts[5]))
                    except: continue

    lh = metrics.get("Left-Hippocampus", 0)
    rh = metrics.get("Right-Hippocampus", 0)
    lv = metrics.get("Left-Lateral-Ventricle", 0)
    rv = metrics.get("Right-Lateral-Ventricle", 0)
    wm_left = metrics.get("Left-Cerebral-White-Matter")
    wm_right = metrics.get("Right-Cerebral-White-Matter")
    ctx_left = metrics.get("ctx-lh")
    ctx_right = metrics.get("ctx-rh")

    return {
        "Left Hippocampus": lh,
        "Right Hippocampus": rh,
        "Asymmetry Index": round(abs(lh - rh) / max(lh, rh + 1e-6), 2),
        "Evans Index": round((lv + rv) / (lh + rh + 1e-6), 2),
        "Left WM Volume": wm_left,
        "Right WM Volume": wm_right,
        "Left Cortex Volume": ctx_left,
        "Right Cortex Volume": ctx_right,
        "Average Cortical Thickness": round(sum(thickness)/len(thickness), 2) if thickness else None
    }

# === Predict stage ===
def predict_stage(mmse, cdr, adas):
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI"
    elif cdr == 0 and mmse >= 26:
        return "Normal"
    return "Uncertain"

# === GPT Summary ===
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

# === Generate PDF Report ===
def create_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

# === Send Email ===
def send_email_report(email, pdf_bytes, segmentation_b64):
    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    attachment = Attachment(
        FileContent(base64.b64encode(pdf_bytes).decode()),
        FileName("report.pdf"),
        FileType("application/pdf"),
        Disposition("attachment")
    )
    html_content = f"""
    <p>Hi,</p>
    <p>Your MRI report is ready. See attached PDF.</p>
    <p><img src="data:image/png;base64,{segmentation_b64}" width="400"/></p>
    <p>Stay healthy,<br>Alzheimer's Risk Platform</p>
    """
    message = Mail(
        from_email="noreply@alzapi.com",
        to_emails=email,
        subject="🧠 Your Alzheimer's MRI Report",
        html_content=html_content
    )
    message.attachment = attachment
    sg.send(message)

# === Endpoint ===
@app.post("/analyze-mri/")
async def analyze_mri(file: UploadFile = File(...),
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
    with open(seg_path, "rb") as f:
        seg_base64 = base64.b64encode(f.read()).decode()

    send_email_report(email, pdf, seg_base64)

    return {
        "🧠 Clinical Biomarkers": biomarkers,
        "📈 Cognitive Scores": {"MMSE": mmse, "CDR": cdr, "ADAS-Cog": adas_cog},
        "🧬 Disease Stage": stage,
        "📋 GPT Summary": summary,
        "🧾 PDF Report": "✅ Sent to email",
        "🧠 Brain Segmentation Preview (base64)": seg_base64
    }

@app.get("/")
def root():
    return {"message": "✅ FastAPI MRI + GPT + PDF + Email + Segmentation Preview"}


