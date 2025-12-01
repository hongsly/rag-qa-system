import json
from pathlib import Path

import arxiv
import ollama
import pymupdf4llm
from src.utils import Chunk, ChunkMetadata, chunk_text
from tqdm import tqdm


class _PDFDocument:
    def __init__(self, path: Path):
        """Initialize with a PDF path.
        Converts the PDF to markdown and stores the markdown text.

        Args:
            path: Path to the PDF file.
        """
        self.path = path
        try:
            self.md_text = pymupdf4llm.to_markdown(str(self.path))
        except Exception as e:
            print(f"Error parsing PDF {self.path}: {e}")
            self.md_text = ""

    def get_name(self) -> str:
        return self.path.stem

    def get_arxiv_id(self) -> str:
        return self.get_name().split("_")[0]


class CorpusLoader:
    def __init__(self, ollama_model: str = "qwen2.5-coder:7b"):
        self.chunks = None
        self.arxivClient = arxiv.Client()
        # self.llm = ChatOllama(model=ollama_model, reasoning=False)
        self.ollama_model = ollama_model

    def parse_pdfs(
        self, pdf_paths: list[Path], chunk_size: int = 500, overlap: int = 50
    ):
        """Parse PDFs and create chunks, saving the chunks to self.chunks."""
        pdf_documents = [
            _PDFDocument(path) for path in tqdm(pdf_paths, desc="Parsing PDFs")
        ]
        arxiv_ids = [doc.get_arxiv_id() for doc in pdf_documents]
        metadata_list = [
            self._fetch_arxiv_metadata(id)
            for id in tqdm(arxiv_ids, desc="Fetching arxiv metadata")
        ]
        self.chunks = [
            chunk
            for i, pdf_document in enumerate(pdf_documents)
            for chunk in chunk_text(
                pdf_document.md_text,
                chunk_size=chunk_size,
                overlap=overlap,
                parent_document_name=pdf_document.get_name(),
                metadata=metadata_list[i],
            )
        ]

    def _is_reference_section(self, chunk_text: str) -> bool:
        prompt = f"""
        You are a helpful assistant that determines if a text comes from the reference/bibliography section of a paper,
        i.e. it only contains a list of references/bibliography entries (authors, titles, venue, and publication years).

        - Return "yes" if and only if the text only contains a list of refereces (authors, titles, venue, and publication years).
        - Return "no" if the text contains any actual content, e.g. if it contains both a conclusion paragraph and the start of refereces section.

        The text is: 
        <text>
        {chunk_text}
        </text>

        Return "yes" if it is reference/bibliography only, otherwise return "no". Only return "yes" or "no".
        """
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        return "yes" in response.response.lower()

    def filter_reference_chunks(self):
        self.chunks = [
            chunk
            for chunk in tqdm(self.chunks, desc="Filtering reference chunks")
            if not self._is_reference_section(chunk["chunk_text"])
        ]

    def save_chunks(self, output_path: Path):
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        with open(output_path, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk) + "\n")

    def get_chunks(self) -> list[Chunk]:
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        return self.chunks

    def get_statistics(self) -> dict:
        if self.chunks is None:
            raise ValueError("Chunks not loaded")

        token_counts = [chunk["token_count"] for chunk in self.chunks]
        return {
            "num_chunks": len(self.chunks),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
        }

    def _fetch_arxiv_metadata(self, arxiv_id: str) -> ChunkMetadata:
        search = arxiv.Search(id_list=[arxiv_id])
        r = next(self.arxivClient.results(search))
        if arxiv_id not in r.entry_id:
            print(f"WARINING: ArXiv ID mismatch: {arxiv_id} != {r.entry_id}")
        return {
            "arxiv_id": arxiv_id,
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "url": r.pdf_url,
        }
