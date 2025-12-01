import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from src.utils import EVAL_DATA_DIR, RAW_DATA_DIR, get_openai_api_key


def load_and_clean_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path, mode="single")
    full_text = loader.load()[0].page_content

    # Remove references section
    # NOTE: simple heuristic. Currently discards the appendix section after references
    # Pattern: ^References$ or ^Bibliography$ (case insensitive, multiline)
    ref_header_pattern = r"^\s*(?:\d+\.?\s*)?(References|Bibliography)\s*$"
    ref_match = re.search(
        ref_header_pattern, full_text, flags=re.IGNORECASE | re.MULTILINE
    )

    if ref_match:
        start_idx = ref_match.start()

        # Chop everything after it
        cleaned_text = full_text[:start_idx]

        reference_start = (
            full_text[start_idx : start_idx + 50].replace("\n", " ").strip()
        )
        print(f"✂️  Chopped references from {pdf_path.name}")
        print("reference start: ", reference_start)
    else:
        cleaned_text = full_text
        print(f"⚠️  No references found in {pdf_path.name}")

    return Document(page_content=cleaned_text, metadata={"source": pdf_path})


def load_pdfs() -> list[Document]:
    pdf_paths = list(RAW_DATA_DIR.glob("*.pdf"))
    return [load_and_clean_pdf(pdf_path) for pdf_path in pdf_paths]


def _get_openai_generator(model: str = "gpt-4o-mini") -> TestsetGenerator:
    get_openai_api_key()
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)


def _get_ollama_generator(model: str = "qwen2.5-coder:7b") -> TestsetGenerator:
    ollama_llm = LangchainLLMWrapper(
        ChatOllama(
            model=model,
            reasoning=False,
            temperature=0.0,
            num_ctx=64000,
            keep_alive="5m",
        )
    )
    ollama_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )
    generator = TestsetGenerator(llm=ollama_llm, embedding_model=ollama_embeddings)
    return generator


def generate_testset():
    """Generate a testset of 40 questions from the chunks."""

    # filter out documents that are too long -- potentially over Ollama context window
    documents = [d for d in load_pdfs() if len(d.page_content) < 80000]
    print(f"Loaded {len(documents)} documents")

    generator = _get_ollama_generator()

    dataset = generator.generate_with_langchain_docs(
        documents, testset_size=40, with_debugging_logs=True, raise_exceptions=False
    )
    output_path = EVAL_DATA_DIR / "ragas_testset.jsonl"
    dataset.to_jsonl(str(output_path))


if __name__ == "__main__":
    generate_testset()
