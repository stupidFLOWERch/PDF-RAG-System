import pytest
from fastapi import HTTPException

from src.backend.app import validate_documents


def test_validate_documents_rejects_empty_documents():
    """
    Empty document list should be rejected with HTTP 422.
    """
    with pytest.raises(HTTPException) as exc_info:
        validate_documents([])

    assert exc_info.value.status_code == 422


def test_validate_documents_rejects_near_empty_documents():
    """
    Documents with too few words should be rejected.
    """
    documents = [
        {
            "text": "Hello world",
            "metadata": {},
        }
    ]

    with pytest.raises(HTTPException) as exc_info:
        validate_documents(documents)

    assert exc_info.value.status_code == 422


def test_validate_documents_accepts_valid_documents():
    """
    Documents with enough words should pass validation.
    """
    documents = [
        {
            "text": (
                "This is a valid document containing enough "
                "words to pass the minimum word threshold "
                "required by the PDF RAG system."
            ),
            "metadata": {},
        }
    ]

    total_words = validate_documents(documents)

    assert total_words >= 20

