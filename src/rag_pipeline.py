import os
from typing import Literal

from src.generator import Generator
from src.hybrid_search import HybridRetriever
from src.utils import PROCESSED_DATA_DIR, Chunk, get_openai_api_key

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RagAssistant:
    def __init__(self):
        api_key = get_openai_api_key()
        self.generator = Generator(api_key)

        self.retriever = HybridRetriever()
        self.retriever.load_from_file(
            PROCESSED_DATA_DIR / "rag_index.faiss", PROCESSED_DATA_DIR / "chunks.jsonl"
        )

    def query(
        self,
        query: str,
        model: str = "gpt-4o-mini",
        retrieval_mode: Literal["hybrid", "dense", "sparse", "none"] = "sparse",
        top_k: int = 5,
    ) -> tuple[str, list[Chunk]]:
        context = None
        match retrieval_mode:
            case "hybrid":
                context = self.retriever.search_hybrid(query, top_k)
            case "dense":
                context = self.retriever.search_dense(query, top_k)
            case "sparse":
                context = self.retriever.search_sparse(query, top_k)
        answer = self.generator.generate(
            query, context, model=model, retrieval_mode=retrieval_mode
        )
        return answer, context
