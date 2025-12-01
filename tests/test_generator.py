import pytest
from src.generator import Generator

@pytest.fixture
def generator():
    return Generator(api_key="fake_key")

def test_get_prompt_without_context(generator):
    prompt = generator._get_prompt("What is RAG?", context=None)

    print(f"Prompt: {prompt}")
    assert "<question>What is RAG?</question>" in prompt
    assert "<documents>" not in prompt

def test_get_prompt_with_context(generator):
    context = [
        {"chunk_id": "doc1:chunk0", "chunk_text": "RAG is retrieval-augmented generation.", "metadata": {"title": "RAG", "authors": ["John Doe"], "year": 2025}},
        {"chunk_id": "doc2:chunk1", "chunk_text": "It combines search with LLMs.", "metadata": {"title": "LLMs", "authors": ["Jane Doe"], "year": 2025}},
    ]

    prompt = generator._get_prompt("What is RAG?", context=context)

    print(f"Prompt: {prompt}")
    assert "<question>What is RAG?</question>" in prompt
    assert "<documents>" in prompt
    assert '<document id="doc1:chunk0">' in prompt
    assert '<title>RAG</title>' in prompt
    assert '<authors>John Doe</authors>' in prompt
    assert '<year>2025</year>' in prompt
    assert '<content>RAG is retrieval-augmented generation.</content>' in prompt
    assert '<document id="doc2:chunk1">' in prompt
    assert '<title>LLMs</title>' in prompt
    assert '<authors>Jane Doe</authors>' in prompt
    assert '<year>2025</year>' in prompt
    assert '<content>It combines search with LLMs.</content>' in prompt
