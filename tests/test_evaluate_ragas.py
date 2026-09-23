import json
import pytest

from src.rag_ollama.evaluate_ragas import (
    prepare_dataset,
    save_results,
)

def test_prepare_dataset():
    records = [
        {
            "question": "What is PlantPals?",
            "answer": "PlantPals is a plant system.",
            "ground_truth": "PlantPals is a plant information system.",
            "contexts": [
                "PlantPals is a plant information system."
            ]
        }
    ]

    dataset = prepare_dataset(records)

    assert len(dataset) == 1
    assert dataset[0]["question"] == "What is PlantPals?"
    assert dataset[0]["answer"] == \
        "PlantPals is a plant system."

    assert dataset[0]["ground_truth"] == \
        "PlantPals is a plant information system."

    assert len(dataset[0]["contexts"]) == 1

def test_save_results(tmp_path):
    original_records = [
        {
            "id": 1,
            "question": "What is PlantPals?",
            "ground_truth": "PlantPals is a plant system.",
            "answer": "PlantPals is a plant system.",
            "contexts": ["PlantPals is a plant system."]
        }
    ]

    scores = [
        {
            "faithfulness": 0.9,
            "context_precision": 0.8,
            "context_recall": 0.85
        }
    ]

    output_path = tmp_path / "results.json"

    results = save_results(
        original_records,
        scores,
        output_path
    )

    assert output_path.exists()
    assert len(results) == 1

    assert results[0]["faithfulness"] == 0.9
    assert results[0]["context_precision"] == 0.8
    assert results[0]["context_recall"] == 0.85

    saved_data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert len(saved_data) == 1
    assert saved_data[0]["question"] == "What is PlantPals?"