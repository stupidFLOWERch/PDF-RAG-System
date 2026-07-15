from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from ..rag_ollama.pdf_loader import extract_lines, merge_lines
from ..rag_ollama.chunker import create_sections, flatten_sections
from ..rag_ollama.db import VectorDB
from ..rag_ollama.rag import RAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # clear old database
    db = VectorDB(
        collection_name="documents",
        persist_directory="./chroma_db"
    )

    db.clear()

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    # Save uploaded pdf
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse PDF
    elements = extract_lines(file_path)
    elements = merge_lines(elements)

    sections, title = create_sections(elements)

    documents = flatten_sections(
        sections,
        title
    )

    # Save into ChromaDB
    db = VectorDB(
        collection_name="documents",
        persist_directory="./chroma_db"
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
        persist_directory="./chroma_db"
    )

    rag = RAG(db)

    answer = rag.ask(query)

    return {
        "answer": answer
    }

@app.get("/")
def home():
    return {
        "status": "API running"
    }