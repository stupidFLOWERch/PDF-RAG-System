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
| PDF Extraction (Text-based) | PyMuPDF |
| PDF Extraction (Scanned) | PaddleOCR-VL-1.6 / PP-StructureV3 |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (BAAI/bge-base-en-v1.5) |
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

## Evaluation

The RAG system includes a separate evaluation pipeline for measuring retrieval and answer quality using RAGAS.

### Evaluation Pipeline

```text
Ground Truth Questions
    ↓
evaluate_rag.py
    ↓
RAG System
    ↓
Retrieve Relevant Chunks
    ↓
Generate Answers
    ↓
ragas_dataset.json
    ↓
evaluate_ragas.py
    ↓
RAGAS + Qwen
    ↓
Faithfulness / Context Precision / Context Recall
    ↓
ragas_results.json
```

### Generate Evaluation Dataset

`evaluate_rag.py` runs the RAG system against predefined questions and stores the generated answers and retrieved contexts.

```bash
python -m rag_ollama.evaluate_rag --pdf path/to/document.pdf
```

Supported options include:

| Option                | Description                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `--pdf`               | Path to the PDF document used for evaluation.                                            |
| `--ground-truth`      | Path to the ground-truth JSON file containing evaluation questions and expected answers. |
| `--output`            | Path where the generated RAGAS dataset will be saved.                                    |
| `--collection`        | Name of the ChromaDB collection used to store document embeddings.                       |
| `--persist-directory` | Directory where the ChromaDB data is stored.                                             |
| `--top-k`             | Number of relevant chunks retrieved for each question.                                   |
| `--skip-index`        | Skips rebuilding the vector index and uses the existing ChromaDB collection.             |

### Run RAGAS Evaluation

`evaluate_ragas.py` evaluates the generated dataset using RAGAS with **Qwen 2.5:7b model**.

The evaluation uses the following metrics:

| Metric                | Explanation                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| Faithfulness    | Measures whether the generated answer is supported by the retrieved context.                         |
| Context Precision | Measures whether the retrieved contexts are relevant to the question.                                |
| Context Recall  | Measures whether the retrieved information contains the information required to answer the question. |

Run:

```bash
python -m rag_ollama.evaluate_ragas
```

The evaluation results are saved as a JSON file containing per-question scores and average scores.

### Evaluation vs Unit Testing

Unit tests verify individual functions and components, while RAGAS evaluates the quality of the complete RAG pipeline.

RAGAS evaluation is treated as a separate evaluation workflow rather than a regular unit test run on every code change.
