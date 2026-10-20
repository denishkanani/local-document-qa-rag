
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Load the embedding model.
        self.model = SentenceTransformer(model_name)

        # These will be populated when documents are indexed.
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]) -> None:
        """Create embeddings and store them in FAISS."""

        if not chunks:
            raise ValueError("No text chunks to index.")

        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # FAISS expects float32 vectors.
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # Inner product of normalized vectors equals cosine similarity.
        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

    def search(
        self,
        question: str,
        top_k: int = 4,
    ) -> list[dict]:
        """Return the most similar chunks for a question."""

        if self.index is None:
            raise ValueError(
                "Build the vector store before searching."
            )

        if not question.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, chunk_index in zip(
            scores[0],
            indices[0],
        ):
            if chunk_index < 0:
                continue

            result = dict(self.chunks[int(chunk_index)])
            result["score"] = float(score)

            results.append(result)

        return results
