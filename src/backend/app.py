"""
FastAPI Application for PDF RAG System
Handles PDF upload, text extraction, and chat functionality.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
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
from rag_ollama.chunker import flatten_sections, chunk_document
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
# Constants
# ============================================================
# Minimum number of words that must be extracted for an upload
# to be considered valid. Tune this to fit your documents.
MIN_WORDS = 20

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
    except Exception:
        return True


def validate_documents(documents: list) -> int:
    """
    Validate that a PDF produced usable content.

    Returns the total word count on success.
    Raises HTTPException(422) otherwise.
    """
    # 1. No chunks at all
    if not documents:
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text was extracted from this PDF. "
                "It may be a scanned image without OCR, empty, or corrupted."
            ),
        )

    # 2. Count words across all chunks
    total_words = sum(
        len(str(doc.get("text", "")).split())
        for doc in documents
    )

    # 3. Too few words — likely just headings or noise
    if total_words < MIN_WORDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {total_words} words were extracted. "
                "The PDF appears to contain little or no text content."
            ),
        )

    return total_words


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
    - Rejects PDFs that produce zero usable chunks
    """
    # --------------------------------------------------------
    # 0. Basic validation
    # --------------------------------------------------------
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    file_path = UPLOAD_FOLDER / file.filename

    # --------------------------------------------------------
    # 1. Save uploaded file
    # --------------------------------------------------------
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {e}",
        )

    # --------------------------------------------------------
    # 2. Extract + chunk (DO NOT touch DB yet)
    # --------------------------------------------------------
    try:
        is_scanned = is_scanned_pdf(str(file_path))
        print(
            f"🔍 PDF Type: "
            f"{'Scanned PDF (OCR)' if is_scanned else 'Text PDF (PyMuPDF)'}"
        )

        elements = []

        if is_scanned:
            from rag_ollama.paddle_loader import extract_with_paddle
            sections, title = extract_with_paddle(
                str(file_path), use_gpu=False, use_vl=True,
            )
            # paddle_loader.create_sections() already provides a
            # "Document Start" fallback section when no headings exist,
            # so empty documents here means OCR truly extracted nothing.
            documents = flatten_sections(sections, title)
            if not documents:
                raise HTTPException(
                    status_code=422,
                    detail="OCR produced no usable text.",
                )
        else:
            elements = merge_lines(extract_lines(str(file_path)))
            documents, title = chunk_document(elements)     # ✅ 用统一入口

        # ----------------------------------------------------
        # 3. VALIDATION: reject empty / near-empty PDFs
        # ----------------------------------------------------
        total_words = validate_documents(documents)

    except HTTPException:
        # Re-raise our own validation errors untouched
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {e}",
        )

    # --------------------------------------------------------
    # 4. Only NOW clear the DB and store the new documents.
    #    (So a bad upload doesn't wipe your existing data.)
    # --------------------------------------------------------
    try:
        db = VectorDB(
            collection_name="documents",
            persist_directory=str(PROJECT_ROOT / "chroma_db"),
        )
        db.clear()

        # db.clear() may replace the collection object — re-add docs
        db.add_documents(documents)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store documents in vector DB: {e}",
        )

    # --------------------------------------------------------
    # 5. Success
    # --------------------------------------------------------
    return {
        "status": "success",
        "message": "Upload successful",
        "title": title or "Untitled",
        "chunks": len(documents),
        "words": total_words,
        "is_scanned": is_scanned,
        "filename": file.filename,
    }


@app.post("/chat")
async def chat(data: dict):
    """
    Chat endpoint for RAG (Retrieval-Augmented Generation).
    Retrieves relevant documents and generates answers using Ollama.
    """
    query = data.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    db = VectorDB(
        collection_name="documents",
        persist_directory=str(PROJECT_ROOT / "chroma_db"),
    )

    if db.collection.count() == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents have been uploaded yet.",
        )

    rag = RAG(db)
    answer = rag.ask(query)

    return {"answer": answer}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "API running"}