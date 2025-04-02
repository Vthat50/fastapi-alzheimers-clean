import os, shutil, base64, io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fpdf import FPDF
from openai import OpenAI

# Environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# App setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "✅ Render backend ready. Upload MRI and analyze results."}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...),
                  mmse: int = Form(...),
                  cdr: float = Form(...),
                  adas_cog: float = Form(...),
                  email: str = Form(...)):

    subject_id = "sub-01"
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Instructions to user (e.g. your own terminal) to run FastSurfer locally
    return {
        "message": "🧠 MRI uploaded. Run FastSurfer locally, then send results via POST to /get-results"
    }

@app.post("/get-results/")
async def get_results(mmse: int = Form(...),
                      cdr: float = Form(...),
                      adas_cog: float = Form(...)):

    # Simulate stats parsed from FastSurfer output (replace with actual parser later)
    biomarkers = {
        "Left Hippocampus": 2200.0,
        "Right Hippocampus": 2100.0,
        "Asymmetry Index": 0.05,
        "Evans Index": 0.3,
        "Average Cortical Thickness": 2.4
    }

    summary = generate_summary(biomarkers, mmse, cdr, adas_cog)
    pdf = create_pdf(summary)
    encoded_pdf = base64.b64encode(pdf).decode()

    return {
        "🧠 Clinical Biomarkers": biomarkers,
        "📈 Cognitive Scores": {"MMSE": mmse, "CDR": cdr, "ADAS-Cog": adas_cog},
        "🧬 Disease Stage": predict_stage(mmse, cdr, adas_cog),
        "📋 GPT Summary": summary,
        "🧾 PDF Report (base64)": encoded_pdf
    }

def generate_summary(biomarkers, mmse, cdr, adas):
    prompt = f"""Patient MRI & Cognitive Results:
- Left Hippocampus: {biomarkers['Left Hippocampus']} mm³
- Right Hippocampus: {biomarkers['Right Hippocampus']} mm³
- Asymmetry Index: {biomarkers['Asymmetry Index']}
- Evans Index: {biomarkers['Evans Index']}
- Cortical Thickness: {biomarkers['Average Cortical Thickness']} mm
- MMSE: {mmse}, CDR: {cdr}, ADAS-Cog: {adas}

Please write:
- Clinical interpretation
- Stage (Normal/MCI/Alzheimer’s)
- Explanation of results
- Recommendations
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
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

def predict_stage(mmse, cdr, adas):
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI"
    elif cdr == 0 and mmse >= 26:
        return "Normal"
    return "Uncertain"
