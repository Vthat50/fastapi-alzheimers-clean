import os, base64, io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF
from pydantic import BaseModel
from openai import OpenAI
import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

UPLOAD_DIR = "/tmp/stats"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def parse_stats(aseg_path, aparc_path):
    metrics, thickness = {}, []
    if os.path.exists(aseg_path):
        with open(aseg_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 5: continue
                label = parts[4]
                try: volume = float(parts[3])
                except: continue
                metrics[label] = volume

    if os.path.exists(aparc_path):
        with open(aparc_path) as f:
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

def predict_stage(mmse, cdr, adas):
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI"
    elif cdr == 0 and mmse >= 26:
        return "Normal"
    return "Uncertain"

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

def send_email_report(email, pdf_bytes):
    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    attachment = Attachment(
        FileContent(base64.b64encode(pdf_bytes).decode()),
        FileName("report.pdf"),
        FileType("application/pdf"),
        Disposition("attachment")
    )
    message = Mail(
        from_email="noreply@alzapi.com",
        to_emails=email,
        subject="🧠 Your Alzheimer's MRI Report",
        html_content="Your report is attached as a PDF."
    )
    message.attachment = attachment
    sg.send(message)

@app.post("/upload-stats/")
async def upload_stats(aseg: UploadFile = File(...),
                       aparc: UploadFile = File(...),
                       mmse: int = Form(...),
                       cdr: float = Form(...),
                       adas: float = Form(...),
                       email: str = Form(...)):
    aseg_path = os.path.join(UPLOAD_DIR, "aseg+DKT.stats")
    aparc_path = os.path.join(UPLOAD_DIR, "lh.aparc.stats")
    with open(aseg_path, "wb") as f:
        f.write(await aseg.read())
    with open(aparc_path, "wb") as f:
        f.write(await aparc.read())

    biomarkers = parse_stats(aseg_path, aparc_path)
    stage = predict_stage(mmse, cdr, adas)
    summary = generate_summary(biomarkers, mmse, cdr, adas)
    pdf = create_pdf(summary)
    send_email_report(email, pdf)

    return {
        "🧠 Clinical Biomarkers": biomarkers,
        "📈 Cognitive Scores": {"MMSE": mmse, "CDR": cdr, "ADAS-Cog": adas},
        "🧬 Disease Stage": stage,
        "📋 GPT Summary": summary,
        "🧾 PDF Report": "✅ Sent to email"
    }

@app.get("/")
def root():
    return {"message": "✅ Cloud backend ready to parse FastSurfer .stats and email reports"}
