from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import Chunk, load_chunks_from_jsonl


class VectorStore:
    def __init__(self):
        self.index = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []

    def add_chunks(self, chunks: list[Chunk]):
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.embed(chunk_texts)
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (dot product)
        elif self.index.d != dimension:
            raise ValueError(
                f"Dimension {dimension} does not match index dimension {self.index.d}"
            )
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def load(self, index_path: Path, chunks_path: Path):
        """Load the index and chunks from the given paths."""
        print(f"Loading index from {index_path}...", end=" ")
        self.index = faiss.read_index(str(index_path))
        print("done")
        self.chunks = load_chunks_from_jsonl(chunks_path)
        print(f"Loaded {len(self.chunks)} chunks from {chunks_path}")

    def save(self, path: Path):
        """Save the index to the given path."""
        if self.index is None:
            raise ValueError("Index not created")
        faiss.write_index(self.index, str(path))

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if self.index is None:
            raise ValueError("Index not created")
        query_embedding = self.embed([query]).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]

    def embed(self, chunks: list[str]) -> np.ndarray:
        # Important: Normalize embeddings to have unit length
        return self.model.encode(chunks, normalize_embeddings=True)