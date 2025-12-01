import heapq
from pathlib import Path

from src.sparse_retrieval import BM25Retriever
from src.utils import Chunk
from src.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, k: int = 60) -> None:
        self.bm25 = BM25Retriever()
        self.vector_store = VectorStore()
        self.k = k

    def load_chunks(self, chunks: list[Chunk]) -> None:
        """Load chunks into BM25 and FAISS indexes."""
        self.bm25.load_chunks(chunks)
        self.vector_store.add_chunks(chunks)

    def load_from_file(self, faiss_index_path: Path, chunks_path: Path) -> None:
        self.vector_store.load(faiss_index_path, chunks_path)
        self.bm25.load_chunks(self.vector_store.chunks)  # TODO: load BM25 from file

    def search_dense(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Search for top k chunks using dense retrieval."""
        return self.vector_store.search(query, top_k)

    def search_sparse(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Search for top k chunks using sparse retrieval."""
        return self.bm25.search(query, top_k)

    def search_hybrid(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Search for top k chunks using hybrid retrieval."""
        retrieve_k = 4 * top_k
        sparse_result = self.bm25.search(query, retrieve_k)
        dense_result = self.vector_store.search(query, retrieve_k)

        # Build rank dictionaries and chunk lookup
        sparse_rank = {
            chunk["chunk_id"]: i + 1 for i, chunk in enumerate(sparse_result)
        }
        dense_rank = {chunk["chunk_id"]: i + 1 for i, chunk in enumerate(dense_result)}
        id_to_chunk = {
            chunk["chunk_id"]: chunk for chunk in sparse_result + dense_result
        }
        rrf_score = {}
        for id in id_to_chunk:
            rrf_score[id] = 0
            if id in sparse_rank:
                rrf_score[id] += 1 / (self.k + sparse_rank[id])
            if id in dense_rank:
                rrf_score[id] += 1 / (self.k + dense_rank[id])

        top_k_ids = heapq.nlargest(top_k, rrf_score, key=rrf_score.get)
        return [id_to_chunk[id] for id in top_k_ids]
