from pathlib import Path

from src.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def test_search():
    vector_store = VectorStore()
    vector_store.load(PROCESSED_DATA_DIR / "rag_index.faiss", PROCESSED_DATA_DIR / "chunks.jsonl")
    chunks = vector_store.search("How does hybrid retrieval work?")
    print(f"Found {len(chunks)} chunks")    
    for chunk in chunks:
        print("-" * 100)
        print("-" * 100)
        print(chunk["chunk_id"])
        print("-" * 100)
        print(chunk["chunk_text"])

if __name__ == "__main__":
    test_search()