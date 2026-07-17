from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import sys
from pathlib import Path

# Get current file directory (src/backend)
BASE_DIR = Path(__file__).resolve().parent  # src/backend/
SRC_DIR = BASE_DIR.parent                   # src/
PROJECT_ROOT = SRC_DIR.parent               # pdf-rag-system/
# frontend directory path src/frontend
FRONTEND_DIR = SRC_DIR / "frontend"

# Add src directory to Python path to import rag_ollama
sys.path.append(str(SRC_DIR))

# Import RAG components
from rag_ollama.pdf_loader import extract_lines, merge_lines
from rag_ollama.chunker import create_sections, flatten_sections
from rag_ollama.db import VectorDB
from rag_ollama.rag import RAG

app = FastAPI()

# Debug directory
# print(f"BASE_DIR: {BASE_DIR}")
# print(f"SRC_DIR: {SRC_DIR}")
# print(f"FRONTEND_DIR: {FRONTEND_DIR}")
# print(f"FRONTEND_DIR exists: {FRONTEND_DIR.exists()}")

# when call for static, find file in src/frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# pdf-rag-system/uploads
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Clear db
    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db")
    )
    db.clear()

    file_path = UPLOAD_FOLDER / file.filename

    # Save uploaded pdf
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract PDF
    elements = extract_lines(str(file_path))
    elements = merge_lines(elements)

    sections, title = create_sections(elements)

    documents = flatten_sections(sections, title)

    # Save to ChromaDB
    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db")
    )
    db.add_documents(documents)

    return {
        "message": "Upload successful",
        "title": title,
        "chunks": len(documents)
    }

@app.post("/chat")
async def chat(data: dict):
    query = data["query"]

    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db")
    )

    rag = RAG(db)

    answer = rag.ask(query)

    return {
        "answer": answer
    }

@app.get("/health")
def health():
    return {"status": "API running"}

