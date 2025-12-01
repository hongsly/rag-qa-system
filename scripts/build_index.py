from src.data_loader import CorpusLoader
from src.utils import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.vector_store import VectorStore


def build_index():
    pdf_paths = list(RAW_DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs")
    # pdf_paths = pdf_paths[:2]  # TODO: remove
    corpus_loader = CorpusLoader()
    corpus_loader.parse_pdfs(pdf_paths)
    print(f"Parsing complete\n\n")

    print(f"Number of chunks before filtering: {len(corpus_loader.get_chunks())}")
    corpus_loader.filter_reference_chunks()
    print(f"Number of chunks after filtering: {len(corpus_loader.chunks)}\n\n")

    print(f"Statistics: {corpus_loader.get_statistics()}")

    corpus_loader.save_chunks(PROCESSED_DATA_DIR / "chunks.jsonl")
    print(f"Saved {len(corpus_loader.get_chunks())} chunks")

    chunks = corpus_loader.get_chunks()
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)
    vector_store.save(PROCESSED_DATA_DIR / "rag_index.faiss")
    print(f"Saved index to {PROCESSED_DATA_DIR / 'rag_index.faiss'}")


if __name__ == "__main__":
    build_index()
