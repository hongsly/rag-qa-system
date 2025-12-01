import pytest
from src.sparse_retrieval import BM25Retriever
from src.utils import Chunk


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk_0",
            chunk_text="attention mechanism is used in transformers for sequence modeling",
            token_count=10,
        ),
        Chunk(
            chunk_id="chunk_1",
            chunk_text="BERT uses bidirectional attention for language understanding",
            token_count=8,
        ),
        Chunk(
            chunk_id="chunk_2",
            chunk_text="GPT uses causal attention for text generation",
            token_count=8,
        ),
        Chunk(
            chunk_id="chunk_3",
            chunk_text="neural networks are used for deep learning tasks",
            token_count=8,
        ),
    ]


@pytest.fixture
def bm25_retriever(sample_chunks):
    retriever = BM25Retriever()
    retriever.load_chunks(sample_chunks)
    return retriever


def test_bm25_index_creation(bm25_retriever):
    assert bm25_retriever.bm25 is not None, "❌ BM25 index not created"
    assert len(bm25_retriever.chunks) == 4, "❌ Incorrect number of chunks loaded"


def test_query_attention_mechanism(bm25_retriever):
    # Test 1: Query matching "attention"
    print("Test 1: Query 'attention mechanism'")
    results = bm25_retriever.search("attention mechanism", k=3)
    print(f"   Retrieved {len(results)} results:")
    for i, chunk in enumerate(results, 1):
        print(f"   {i}. [{chunk['chunk_id']}] {chunk['chunk_text'][:60]}...")

    # Verify: chunk_0 should be rank 1 (has both "attention" and "mechanism")
    assert results[0]["chunk_id"] == "chunk_0", "❌ Expected chunk_0 as top result"
    print("   ✅ Top result is correct (chunk_0 with both terms)\n")


def test_query_bert_language_model(bm25_retriever):
    # Test 2: Query matching "BERT"
    print("Test 2: Query 'BERT language model'")
    results = bm25_retriever.search("BERT language model", k=3)

    print(f"   Retrieved {len(results)} results:")
    for i, chunk in enumerate(results, 1):
        print(f"   {i}. [{chunk['chunk_id']}] {chunk['chunk_text'][:60]}...")

    # Verify: chunk_1 should be rank 1 (mentions BERT)
    assert results[0]["chunk_id"] == "chunk_1", "❌ Expected chunk_1 as top result"
    print("   ✅ Top result is correct (chunk_1 with BERT)\n")


def test_query_no_matches(bm25_retriever):
    # Test 3: Query with no matches
    print("Test 3: Query 'quantum computing'")
    results = bm25_retriever.search("quantum computing", k=3)

    print(f"   Retrieved {len(results)} results (should return closest matches):")
    for i, chunk in enumerate(results, 1):
        print(f"   {i}. [{chunk['chunk_id']}] {chunk['chunk_text'][:60]}...")

    # Should still return k results (closest matches even if not relevant)
    assert len(results) == 3, "❌ Should return k results even with no matches"
    print("   ✅ Returns k results even when no exact matches\n")


def test_bm25_empty_query(bm25_retriever):
    results = bm25_retriever.search("", k=3)
    # Should still return results (BM25 will just return by IDF)
    assert len(results) == 3


def test_bm25_k_larger_than_corpus(bm25_retriever):
    results = bm25_retriever.search("attention", k=10)
    # Should return all 4 chunks (corpus size)
    assert len(results) == 4
