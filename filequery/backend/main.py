import os
from io import BytesIO
from pathlib import Path
import google.generativeai as genai
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

def load_env_file():
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["POST"], allow_headers=["*"])

def extract_text(filename, content):
    extension = Path(filename).suffix.lower()
    if extension == ".txt": return content.decode("utf-8", errors="replace")
    if extension == ".pdf": return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    if extension == ".docx": return "\n".join(p.text for p in Document(BytesIO(content)).paragraphs)
    raise HTTPException(400, "Unsupported file type. Please upload a .txt, .pdf, or .docx file.")

@app.post("/query")
async def query(file: UploadFile = File(...), question: str = Form(...)):
    if not file.filename: raise HTTPException(400, "Please choose a document to upload.")
    if not question.strip(): raise HTTPException(400, "Please enter a question.")
    try: text = extract_text(file.filename, await file.read()).strip()
    except HTTPException: raise
    except Exception as error: raise HTTPException(400, f"Could not read this document: {error}") from error
    if not text: raise HTTPException(400, "No readable text was found in this document.")
    key = os.environ.get("GEMINI_API_KEY")
    if not key: raise HTTPException(500, "GEMINI_API_KEY is not configured. Add it to backend/.env.")
    prompt = f"Answer ONLY from this document. If the answer is not present, say that clearly.\n\nDOCUMENT:\n{text[:30000]}\n\nQUESTION:\n{question.strip()}"
    try:
        genai.configure(api_key=key)
        answer = (genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text or "").strip()
        if not answer: raise ValueError("Gemini returned an empty answer.")
        return {"answer": answer}
    except Exception as error: raise HTTPException(502, f"Gemini request failed: {error}") from error
