import json
import os
from pathlib import Path
from typing import TypedDict

import tiktoken
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_results"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "rag_index.faiss"
CHUNKS_JSONL_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"


class ChunkMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    url: str


class Chunk(TypedDict):
    chunk_id: str
    chunk_text: str
    token_count: int
    metadata: ChunkMetadata


def chunk_text(
    text: str,
    model_name: str = "gpt-3.5-turbo",
    chunk_size: int = 500,
    overlap: int = 50,
    parent_document_name: str = None,
    metadata: ChunkMetadata = None,
) -> list[Chunk]:
    """Chunk text into chunks of chunk_size tokens with overlap"""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)  # return a list of token ids

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        chunk_id = f"chunk_{i}"
        if parent_document_name:
            chunk_id = parent_document_name + ":" + chunk_id
        chunk_text = enc.decode(chunk)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                token_count=len(chunk),
                metadata=metadata
                or ChunkMetadata(arxiv_id="", title="", authors=[], year=0, url=""),
            )
        )
    return chunks


def load_chunks_from_jsonl(chunks_path: Path = CHUNKS_JSONL_PATH) -> list[Chunk]:
    with open(chunks_path, "r") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    return chunks


def get_openai_api_key() -> str:
    """Get the OpenAI API key from the environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found!",
            "Create a .env file with: OPENAI_API_KEY=<your_api_key>",
        )
    return api_key
