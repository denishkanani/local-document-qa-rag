
import fitz


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list of dictionaries containing the text
    and the original page number.
    """
    documents = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append({
                    "text": text,
                    "page": page_number,
                    "source": file_path,
                })

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Split page text into overlapping character chunks.
    Keep the source and page number for citations.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Require chunk_size > overlap >= 0."
        )

    chunks = []

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": document["page"],
                    "source": document["source"],
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks
