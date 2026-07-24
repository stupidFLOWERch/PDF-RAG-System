# PDF-Based RAG Chatbot

A RAG chatbot that answers questions based on the content of uploaded PDF documents.

## Prerequisites
- Python 3.12+
- Ollama
- Llama 3 (llama3:latest)
## Tech Stack
- Python – Backend and document processing
- FastAPI – Backend API
- PyMuPDF (fitz) – PDF text extraction
- ChromaDB – Vector database
- Sentence Transformers – Text embeddings
- BGE Reranker – Document reranking
- Ollama – Local LLM inference
- Llama 3 – Answer generation
- Uvicorn – ASGI server
## RAG Pipeline

```text
Upload PDF
    ↓
Extract PDF Content
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
Hybrid Search
(Text Search + Semantic Search)
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
