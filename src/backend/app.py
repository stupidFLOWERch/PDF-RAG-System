"""
FastAPI Application for PDF RAG System
Handles PDF upload, text extraction, and chat functionality.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import sys
import fitz  # PyMuPDF for PDF text extraction
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
FRONTEND_DIR = SRC_DIR / "frontend"

# Add src directory to Python path
sys.path.append(str(SRC_DIR))

# Import RAG components
from rag_ollama.pdf_loader import extract_lines, merge_lines
from rag_ollama.chunker import create_sections, flatten_sections
from rag_ollama.db import VectorDB
from rag_ollama.rag import RAG

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from frontend directory
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Upload folder
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================
def is_scanned_pdf(pdf_path: str, threshold: int = 50) -> bool:
    """
    Detect if a PDF is scanned (no text layer) or text-based.
    
    Args:
        pdf_path: Path to the PDF file
        threshold: Minimum character count to consider as text-based
        
    Returns:
        True if scanned (needs OCR), False if text-based
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return len(text.strip()) < threshold
    except:
        return True


# ============================================================
# Routes
# ============================================================
@app.get("/")
def home():
    """Serve the frontend index page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    - Detects if PDF is scanned or text-based
    - Uses PyMuPDF for text-based PDFs (fast)
    - Uses PaddleOCR for scanned PDFs (accurate but slower)
    - Stores chunks in vector database for retrieval
    """
    # Clear existing database
    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db")
    )
    db.clear()

    file_path = UPLOAD_FOLDER / file.filename

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Detect PDF type
    is_scanned = is_scanned_pdf(str(file_path))
    print(f"🔍 PDF Type: {'Scanned PDF (OCR)' if is_scanned else 'Text PDF (PyMuPDF)'}")

    if is_scanned:
        # Scanned PDF → Use PaddleOCR
        print("📄 Use PaddleOCR to process scanned PDF...")
        from rag_ollama.paddle_loader import extract_with_paddle
        
        sections, title = extract_with_paddle(
            str(file_path),
            use_gpu=False,
            use_vl=True
        )
    else:
        # Text-based PDF → Use PyMuPDF (fast)
        print("📄 Use PyMuPDF to process text PDF...")
        elements = extract_lines(str(file_path))
        elements = merge_lines(elements)
        sections, title = create_sections(elements)

    # Flatten sections into chunks
    documents = flatten_sections(sections, title)

    # Store in vector database
    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db")
    )
    db.add_documents(documents)

    return {
        "message": "Upload successful",
        "title": title,
        "chunks": len(documents),
        "is_scanned": is_scanned
    }


@app.post("/chat")
async def chat(data: dict):
    """
    Chat endpoint for RAG (Retrieval-Augmented Generation).
    Retrieves relevant documents and generates answers using Ollama.
    """
    query = data["query"]

    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db")
    )

    rag = RAG(db)
    answer = rag.ask(query)

    return {"answer": answer}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "API running"}