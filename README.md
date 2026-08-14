# PDF-Based RAG Chatbot

A RAG chatbot that answers questions based on the content of uploaded PDF documents.

## Prerequisites

- Python 3.12+
- Ollama (with llama3:latest)
- Docker (optional)

## Features

- 🔍 **Smart PDF Detection** – Automatically detects if a PDF is text-based or scanned
- ⚡ **Fast Processing** – Uses PyMuPDF for text-based PDFs (fast)
- 🧠 **OCR Support** – Uses PaddleOCR-VL-1.6 / PP-StructureV3 for scanned PDFs
- 📊 **Hybrid Search** – Combines text search + semantic search for better retrieval
- 🎯 **Reranking** – Uses BGE Reranker for improved result quality
- 💬 **Local LLM** – Runs Llama 3 via Ollama for offline answer generation

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| PDF Extraction (Text-based) | PyMuPDF (fitz) |
| PDF Extraction (Scanned) | PaddleOCR-VL-1.6 / PP-StructureV3 |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Reranker | BGE Reranker (BAAI/bge-reranker-v2-m3) |
| LLM | Ollama + Llama 3 |
| ASGI Server | Uvicorn |

## RAG Pipeline

```text
Upload PDF
    ↓
Detect PDF type (Text-based vs Scanned)
    ↓
Extract PDF Content (PyMuPDF / PaddleOCR)
    ↓
Detect Headings & Create Sections
    ↓
Split Content into Chunks
    ↓
Generate Embeddings
    ↓
Store Chunks + Embeddings in ChromaDB
    ↓
User Asks a Question
    ↓
Hybrid Search (Text Search + Semantic Search)
    ↓
Combine & Remove Duplicates
    ↓
BGE Reranker
    ↓
Retrieve Top Relevant Chunks
    ↓
Send Context + Question to LLM
    ↓
Generate Answer
```

## How to Run

### 1. Clone this repository

```bash
git clone https://github.com/stupidFLOWERch/PDF-RAG-System.git
cd PDF-RAG-System
```

### 2. Create and activate a virtual environment

Create a Python virtual environment:
```bash
python -m venv .venv
```
Activate the virtual environment on Windows:
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Run the server

Start the FastAPI server using Uvicorn:

```bash
uvicorn src.backend.app:app --reload
```

### 5. Open the chatbot interface

Open the following URL in your browser:

http://localhost:8000/

## Run with Docker

### 1. Clone this repository

```bash
git clone https://github.com/stupidFLOWERch/PDF-RAG-System.git
cd PDF-RAG-System
```

### 2. Build and run with Docker Compose
```bash
docker compose up --build
```

### 3. Open the chatbot interface

Open the following URL in your browser:

http://localhost:8000/
