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
