"""
Compare the performance of different retrievers. User needs to manually enter whether the retrieved chunks are relevant to the query.
"""

from src.utils import Chunk
from pathlib import Path
from src.hybrid_search import HybridRetriever


def calculate_precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int = 5
) -> float:
    """Calculate precision at k."""
    retrieved_top_k = retrieved_ids[:k]
    true_positives = set(relevant_ids).intersection(retrieved_top_k)
    return len(true_positives) / k


def calculate_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate mean reciprocal rank."""
    for rank, id in enumerate(retrieved_ids, start=1):
        if id in relevant_ids:
            return 1 / rank
    return 0.0


def manual_evaluation(query: str, results: list[Chunk], method_name: str) -> list[str]:
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Method: {method_name}")
    print("=" * 100)
    relevant_ids = []
    for chunk in results:
        print("-" * 100)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Content: {chunk['chunk_text']}")
        is_relevant = input("Is the result relevant? (y/n): ")
        if is_relevant == "y":
            relevant_ids.append(chunk["chunk_id"])
    return relevant_ids


def compare_retrievers():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    retriever = HybridRetriever()
    print("Loading index and chunks...")
    retriever.load_from_file(data_dir / "rag_index.faiss", data_dir / "chunks.jsonl")

    test_queries = [
        "What is attention mechanism in transformers?",
        "How does BERT pretraining work?",
        "What is the difference between GPT and BERT?",
    ]

    results = []
    for query in test_queries:
        print("-" * 100)
        print(f"Query: {query}")
        dense_results = retriever.search_dense(query, 5)
        sparse_results = retriever.search_sparse(query, 5)
        hybrid_results = retriever.search_hybrid(query, 5)

        dense_relevant_ids = manual_evaluation(query, dense_results, "Dense")
        dense_retrieved_ids = [c["chunk_id"] for c in dense_results]
        dense_precision_at_5 = calculate_precision_at_k(
            dense_retrieved_ids, dense_relevant_ids, 5
        )

        sparse_relevant_ids = manual_evaluation(query, sparse_results, "Sparse")
        sparse_retrieved_ids = [c["chunk_id"] for c in sparse_results]
        sparse_precision_at_5 = calculate_precision_at_k(
            sparse_retrieved_ids, sparse_relevant_ids, 5
        )

        hybrid_relevant_ids = manual_evaluation(query, hybrid_results, "Hybrid")
        hybrid_retrieved_ids = [c["chunk_id"] for c in hybrid_results]
        hybrid_precision_at_5 = calculate_precision_at_k(
            hybrid_retrieved_ids, hybrid_relevant_ids, 5
        )

        results.append(
            {
                "query": query,
                "dense_precision_at_5": dense_precision_at_5,
                "sparse_precision_at_5": sparse_precision_at_5,
                "hybrid_precision_at_5": hybrid_precision_at_5,
            }
        )

    print("=" * 100)
    print("Summary:")
    print(
        "| Query | Dense Precision@5 / MRR | Sparse Precision@5 / MRR | Hybrid Precision@5 / MRR |"
    )
    for row in results:
        print(
            f"| {row['query']} | {row['dense_precision_at_5']} | {row['sparse_precision_at_5']} | {row['hybrid_precision_at_5']} |"
        )
        print("|" + "-" * 100 + "|")
    average_dense_precision_at_5 = sum(
        [row["dense_precision_at_5"] for row in results]
    ) / len(results)
    average_sparse_precision_at_5 = sum(
        [row["sparse_precision_at_5"] for row in results]
    ) / len(results)
    average_hybrid_precision_at_5 = sum(
        [row["hybrid_precision_at_5"] for row in results]
    ) / len(results)
    print(
        f"| AVERAGE | {average_dense_precision_at_5} | {average_sparse_precision_at_5} | {average_hybrid_precision_at_5} |"
    )


if __name__ == "__main__":
    compare_retrievers()
