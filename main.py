
# main_render.py
import os, shutil, base64, io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fpdf import FPDF
from openai import OpenAI
from dotenv import load_dotenv
import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to Webflow domain for security later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.options("/analyze-mri/")
async def preflight_analyze(): return JSONResponse(status_code=200, content={"ok": True})

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

UPLOAD_DIR = "/tmp/uploads"
FASTSURFER_OUTPUT = "/tmp/fastsurfer-output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FASTSURFER_OUTPUT, exist_ok=True)

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
                if len(parts) >= 5:
                    try: metrics[parts[4]] = float(parts[3])
                    except: continue

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
    return {
        "Left Hippocampus": lh,
        "Right Hippocampus": rh,
        "Asymmetry Index": round(abs(lh - rh) / max(lh, rh + 1e-6), 2),
        "Evans Index": round((lv + rv) / (lh + rh + 1e-6), 2),
        "Average Cortical Thickness": round(sum(thickness)/len(thickness), 2) if thickness else None
    }

def predict_stage(mmse, cdr, adas):
    if cdr >= 1 or mmse < 21 or adas > 35: return "Alzheimer's"
    if 0.5 <= cdr < 1 or 21 <= mmse < 26: return "MCI"
    if cdr == 0 and mmse >= 26: return "Normal"
    return "Uncertain"

def generate_summary(biomarkers, mmse, cdr, adas):
    prompt = f"""Patient MRI & cognitive results:
- Left Hippocampus: {biomarkers['Left Hippocampus']} mm³
- Right Hippocampus: {biomarkers['Right Hippocampus']} mm³
- Asymmetry Index: {biomarkers['Asymmetry Index']}
- Evans Index: {biomarkers['Evans Index']}
- Cortical Thickness: {biomarkers['Average Cortical Thickness']} mm
- MMSE: {mmse}, CDR: {cdr}, ADAS-Cog: {adas}
Generate clinical interpretation, Alzheimer's stage, and recommendations."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

def create_pdf(summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

def send_email_report(email, pdf_bytes, seg_b64):
    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    attachment = Attachment(
        FileContent(base64.b64encode(pdf_bytes).decode()),
        FileName("report.pdf"),
        FileType("application/pdf"),
        Disposition("attachment")
    )
    html = f"<p>Report attached. Brain preview:</p><img src='data:image/png;base64,{seg_b64}' width='400'/>"
    message = Mail("noreply@alzapi.com", email, "🧠 Your MRI Report", html)
    message.attachment = attachment
    sg.send(message)

@app.post("/analyze-mri/")
async def upload_mri(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "🧠 MRI uploaded. Run FastSurfer locally and hit /get-results."}

@app.post("/get-results/")
async def get_results(mmse: int = Form(...), cdr: float = Form(...), adas_cog: float = Form(...), email: str = Form(...)):
    subject_id = "sub-01"
    biomarkers = parse_stats(subject_id)
    summary = generate_summary(biomarkers, mmse, cdr, adas_cog)
    stage = predict_stage(mmse, cdr, adas_cog)
    pdf = create_pdf(summary)
    seg_path = os.path.join(FASTSURFER_OUTPUT, subject_id, "mri", "aparc+aseg.png")
    with open(seg_path, "rb") as f:
        seg_b64 = base64.b64encode(f.read()).decode()
    send_email_report(email, pdf, seg_b64)
    return {"summary": summary, "stage": stage}

@app.get("/")
def root():
    return {"message": "✅ Render API ready. Upload MRI, run FastSurfer locally, then /get-results."}
