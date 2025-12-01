import nltk
from rank_bm25 import BM25Okapi
from src.utils import Chunk
from typing import Callable

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def nltk_tokenizer(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if any(c.isalnum() for c in t)]


class BM25Retriever:
    def __init__(self, tokenizer: Callable[[str], list[str]] = nltk_tokenizer) -> None:
        self.bm25 = None
        self.chunks = None
        self.tokenizer = tokenizer

    def load_chunks(self, chunks: list[Chunk]) -> None:
        """Load chunks into BM25 index."""
        self.chunks = chunks
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        tokenized_chunks = [self.tokenizer(chunk) for chunk in chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Search for top k chunks using BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 index not created")
        tokenized_query = self.tokenizer(query)
        return self.bm25.get_top_n(tokenized_query, self.chunks, n=k)
