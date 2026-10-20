
import pytest

from src.document_loader import chunk_documents


def test_chunk_documents_preserves_metadata():
    documents = [
        {
            "text": "A" * 1000,
            "page": 2,
            "source": "sample.pdf",
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=400,
        overlap=100,
    )

    assert len(chunks) > 1
    assert chunks[0]["page"] == 2
    assert chunks[0]["source"] == "sample.pdf"
    assert len(chunks[0]["text"]) == 400


def test_short_document_creates_one_chunk():
    documents = [
        {
            "text": "Machine learning is useful.",
            "page": 1,
            "source": "sample.pdf",
        }
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Machine learning is useful."


def test_invalid_chunk_settings_raise_error():
    documents = [
        {
            "text": "Example text",
            "page": 1,
            "source": "sample.pdf",
        }
    ]

    with pytest.raises(ValueError):
        chunk_documents(
            documents,
            chunk_size=100,
            overlap=100,
        )


def test_empty_documents_create_no_chunks():
    assert chunk_documents([]) == []
