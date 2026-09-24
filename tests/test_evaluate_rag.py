import json

import pytest

from src.rag_ollama.evaluate_rag import (
    load_questions,
    make_ragas_record,
)


def test_load_questions_accepts_json_array(tmp_path):
    path = tmp_path / "ground_truth.json"

    path.write_text(
        json.dumps([
            {
                "id": 1,
                "question": "What is PlantPals?",
                "ground_truth": "PlantPals is ..."
            }
        ]),
        encoding="utf-8"
    )

    records = load_questions(path)

    assert len(records) == 1
    assert records[0]["question"] == "What is PlantPals?"

def test_load_questions_rejects_missing_question(tmp_path):
    path = tmp_path / "ground_truth.json"

    path.write_text(
        json.dumps([
            {
                "id": 1,
                "ground_truth": "Some answer"
            }
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_questions(path)

def test_make_ragas_record():
    source = {
        "id": 1,
        "question": "What is PlantPals?",
        "ground_truth": "PlantPals is a plant information system."
    }

    retrieved_chunks = [
        {
            "text": "PlantPals is a plant information system.",
            "metadata": {
                "heading": "Introduction"
            }
        }
    ]

    result = make_ragas_record(
        source,
        "PlantPals is a plant information system.",
        retrieved_chunks
    )

    assert result["question"] == "What is PlantPals?"
    assert result["ground_truth"] == \
        "PlantPals is a plant information system."

    assert result["answer"] == \
        "PlantPals is a plant information system."

    assert len(result["contexts"]) == 1
    assert result["contexts"][0] == \
        "PlantPals is a plant information system."
