"""Build a RAGAS-ready evaluation dataset from a PDF and ground_truth.json.

Expected project modules (adjust the imports below if your package layout differs):
  pdf_loader.extract_lines, pdf_loader.merge_lines
  chunker.create_sections, chunker.flatten_sections
  db.VectorDB
  rag.RAG, with RAG.ask_with_context() added as shown in RAGAS_EVALUATION_SETUP.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .pdf_loader import extract_lines, merge_lines
from .chunker import create_sections, flatten_sections, chunk_document
from .db import VectorDB  
from .rag import RAG

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GROUND_TRUTH = (
    PROJECT_ROOT / "evaluation" / "PlantPals_ground_truth.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evaluation" / "PlantPals_ragas_dataset.json"
)


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Accept a JSON array or an object containing data/questions/items."""
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next(
            (payload[key] for key in ("data", "questions", "items") if key in payload),
            None,
        )
    else:
        records = None

    if not isinstance(records, list):
        raise ValueError(
            "ground_truth.json must be a JSON array, or contain a data/questions/items array."
        )

    normalized = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not str(record.get("question", "")).strip():
            raise ValueError(f"Item {index} has no non-empty 'question' field.")
        normalized.append(record)
    return normalized


def build_index(pdf_path: Path, collection_name: str, persist_directory: str) -> VectorDB:
    """Process the PDF using the project's existing loader and chunker, then store chunks."""
    db = VectorDB(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # Clear existing collection before indexing
    try:
        db.delete_collection()
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Re-create the database
    db = VectorDB(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    elements = merge_lines(extract_lines(str(pdf_path)))
    sections, title = create_sections(elements)
    documents = flatten_sections(sections, title)
    db.add_documents(documents)
    print(f"Indexed {len(documents)} chunks from {pdf_path.name}.")
    return db


def make_ragas_record(
    source: dict[str, Any], answer: str, retrieved_chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return both standard RAGAS fields and retrieval-debugging metadata."""
    contexts = [chunk.get("text", "") for chunk in retrieved_chunks]
    ground_truth = source.get("ground_truth", source.get("answer", ""))
    return {
        "id": source.get("id"),
        "question": source["question"],
        "ground_truth": ground_truth,
        # RAGAS-compatible names (contexts can be passed as retrieved_contexts in newer RAGAS).
        "answer": answer,
        "response": answer,
        "contexts": contexts,
        # Retain metadata, ranks, scores, headings, etc. returned by VectorDB for diagnosis.
        "retrieved_chunks": retrieved_chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG over all ground-truth questions.")
    parser.add_argument("--pdf", required=True, type=Path, help="Source PDF to index")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Ground-truth JSON",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output RAGAS dataset",
    )
    parser.add_argument("--collection", default="documents_eval")
    parser.add_argument("--persist-directory", default="./chroma_db_eval")
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse an already-populated persistent collection instead of indexing the PDF again.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not args.skip_index and not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if not args.ground_truth.is_file():
        parser.error(f"Ground-truth JSON not found: {args.ground_truth}")

    questions = load_questions(args.ground_truth)
    if args.skip_index:
        db = VectorDB(args.collection, args.persist_directory)
    else:
        db = build_index(args.pdf, args.collection, args.persist_directory)

    rag = RAG(db)
    dataset = []
    for number, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{number}/{len(questions)}] {question}")
        answer, retrieved_chunks = rag.ask_with_context(question, top_k=args.top_k)
        dataset.append(make_ragas_record(item, answer, retrieved_chunks))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(dataset)} records to {args.output}")


if __name__ == "__main__":
    main()
