# RAG Q&A System

A production-quality **Retrieval-Augmented Generation (RAG)** system for question-answering over research papers, featuring hybrid retrieval (BM25 + Dense), comprehensive evaluation, and Docker deployment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌐 Live Demo

**Try the live app**: [https://hongsly-rag-qa-system-app-ze0vmi.streamlit.app/](https://hongsly-rag-qa-system-app-ze0vmi.streamlit.app/)

Ask questions about RAG and LLM research papers and see the system retrieve relevant context and generate answers in real-time.

---

## 🎯 Project Overview

This project demonstrates a complete RAG pipeline implementation, from data ingestion to evaluation, with a focus on **system design**, **retrieval quality**, and **rigorous evaluation**. Built as a portfolio piece to showcase ML engineering skills.

**Key Highlights:**
- 📊 **Rigorous Evaluation**: 42-question testset with 5 RAGAS metrics + systematic error analysis across 4 retrieval modes
- 🔍 **Hybrid Retrieval**: BM25 (sparse) + SentenceTransformer (dense) with Reciprocal Rank Fusion (87% context recall)
- 💡 **Key Insight**: Dense embeddings underperform on small technical corpora (3.7× worse retrieval failures vs sparse)
- 🐳 **Production-Ready**: Docker + Streamlit UI with health checks, optimized from 4.3GB → 2.2GB
- 📈 **Testset Quality**: Improved from 5.5/10 to 8.5/10 by regenerating from whole documents vs chunks

---

## 🏗️ Architecture

```
┌─────────────────┐
│  ArXiv Papers   │  32 papers (RAG/LLM research)
│    (PDF)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document        │  PyMuPDF4LLM parser
│ Preprocessing   │  + tiktoken tokenization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chunking      │  500 tokens/chunk, 50 overlap
│                 │  → 1,395 chunks total
└─────┬─────┬─────┘
      │     │
      │     └──────────────────┐
      │                        │
      ▼                        ▼
┌───────────┐          ┌─────────────┐
│   FAISS   │          │    BM25     │
│  (Dense)  │          │  (Sparse)   │
│           │          │             │
│ all-MiniLM│          │ NLTK tokens │
│  -L6-v2   │          │             │
└─────┬─────┘          └──────┬──────┘
      │                       │
      └──────────┬────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  RRF Fusion    │  score = Σ 1/(k + rank_i)
        │    (k=60)      │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │  Top-K Chunks  │  Default K=5
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ GPT-4o-mini    │  Generate answer with citations
        │  or Ollama     │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │     Answer     │  + source citations
        └────────────────┘
```

---

## 📊 Evaluation Results

### Performance Summary (42-question testset)

| Method | Answer Correctness | Context Recall | Faithfulness | Context Precision | Success Rate |
|--------|-------------------|----------------|--------------|-------------------|--------------|
| **HYBRID** | **66.9%** ⭐ | 83.3% | 83.0% | 78.1% | **52.4%** ⭐ |
| **SPARSE** | 61.3% | **87.3%** ⭐ | **83.8%** ⭐ | **78.7%** ⭐ | 42.9% |
| **DENSE** | 51.9% | 69.4% | 72.2% | 66.7% | 23.8% |
| **NONE** | 43.5% | N/A | N/A | N/A | N/A |

**Key Insights:**
- ✅ **HYBRID best for answer quality** (67%) - combining both retrieval methods captures diverse relevant chunks
- ✅ **SPARSE best for recall** (87%) - keyword matching excels on technical corpus
- ❌ **DENSE underperforms** (52%) - small corpus (1.4K chunks) doesn't benefit from embeddings
- 📈 **RAG provides +24% lift** over no-retrieval baseline

### Error Analysis

| Failure Mode | HYBRID | SPARSE | DENSE |
|--------------|--------|--------|-------|
| **Retrieval Failure** | 11.9% | 7.1% ✅ | **26.2%** ❌ |
| **Generation Failure** | 9.5% | 11.9% | 19.0% |
| **Ranking Issue** | 7.1% | 9.5% | 9.5% |
| **Partial Retrieval** | 11.9% | 9.5% | 9.5% |

**Takeaway**: Dense embeddings struggle on small technical corpora (3.7× worse retrieval failures than sparse). Hybrid fusion mitigates this weakness.

---

## 🚀 Quick Start

### Option 1: Try the Live Demo (Easiest)

Visit [https://hongsly-rag-qa-system-app-ze0vmi.streamlit.app/](https://hongsly-rag-qa-system-app-ze0vmi.streamlit.app/) to use the deployed application immediately—no installation required!

### Option 2: Docker (Recommended for Local Setup)

```bash
# 1. Clone and navigate
git clone https://github.com/hongsly/ml-engineering-fundamentals.git
cd ml-engineering-fundamentals/projects/rag-qa-system

# 2. Set OpenAI API key
echo "OPENAI_API_KEY=sk-proj-xxx" > .env

# 3. Start with docker-compose
docker-compose up -d

# 4. Open browser
open http://localhost:8501
```

**With local Ollama** (Mac/Windows):
```bash
# Terminal 1: Start Ollama
ollama serve
ollama pull qwen2.5-coder:7b

# Terminal 2: Start app (connects automatically)
docker-compose up -d
```

### Option 3: Local Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY=sk-proj-xxx

# 4. Run Streamlit app
streamlit run app.py
```

**Note**: Index files (`data/processed/*.faiss`, `chunks.jsonl`) are prebuilt and included in the repo.

---

## 📂 Project Structure

```
rag-qa-system/
├── app.py                      # Streamlit web UI
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev/eval dependencies
│
├── src/                        # Core RAG implementation
│   ├── data_loader.py          # PDF parsing + chunking
│   ├── vector_store.py         # FAISS dense retrieval
│   ├── sparse_retrieval.py     # BM25 sparse retrieval
│   ├── hybrid_search.py        # RRF fusion
│   ├── generator.py            # OpenAI/Ollama generation
│   ├── rag_pipeline.py         # End-to-end pipeline
│   └── utils.py                # Shared utilities
│
├── scripts/                    # Build + evaluation scripts
│   ├── build_index.py          # Create FAISS + BM25 indexes
│   └── generate_testset.py    # RAGAS testset generation
│
├── evaluation/
│   └── evaluate_rag.py         # RAGAS metric evaluation
│
├── experiments/                # Analysis + ablation studies
│   ├── analyze_errors.py       # Failure mode categorization
│   └── smoke_test.py           # Quick validation tests
│
├── data/
│   ├── processed/              # Prebuilt indexes
│   │   ├── rag_index.faiss     # Dense vector index (1,395 chunks)
│   │   └── chunks.jsonl        # Chunk metadata + text
│   └── eval/                   # Test questions
│       ├── ragas_testset.jsonl # 32 auto-generated questions
│       └── test_questions.json # 10 manual test questions
│
├── outputs/
│   └── eval_results/           # Evaluation outputs
│       ├── eval_results_*.json # Metrics by retrieval mode
│       └── error_analysis_summary.json
│
└── tests/                      # Unit tests
    ├── test_sparse_retrieval.py
    └── test_generator.py
```

---

## 🔬 Technical Decisions & Trade-offs

### 1. Hybrid Retrieval Strategy

**Decision**: RRF fusion of BM25 + Dense embeddings

**Why**:
- BM25 excels at keyword matching ("ColBERT", "DPR") → 87% recall
- Dense embeddings capture semantic similarity → better for conceptual queries
- RRF fusion (k=60) balances both without tuning weights

**Result**: 67% answer correctness (vs 61% sparse-only, 52% dense-only)

**Lesson Learned**: Hybrid helps *when both methods contribute*. If one method dominates (e.g., small corpus hurts dense), fusion can degrade performance. Query-corpus alignment matters.

### 2. Chunking Strategy

**Decision**: 500 tokens/chunk, 50 overlap

**Why**:
- 500 tokens fits typical LLM context (5 chunks = 2.5K tokens)
- 50-token overlap prevents information loss at boundaries
- Smaller than semantic-based chunking but more predictable

**Alternative Considered**: Recursive character splitting (LangChain) - rejected due to inconsistent chunk sizes

### 3. Testset Generation

**Initial Mistake**: Generated questions from 500-token chunks → 46% low-quality questions from bibliography sections

**Fix**: Regenerated from whole documents with reference sections filtered → quality improved 5.5/10 → 8.5/10

**Lesson**: Testset generation and retrieval chunking have different requirements. RAGAS needs document structure to generate multi-hop questions.

### 4. Small Corpus Challenges

**Observation**: Dense embeddings underperform on 1.4K chunk corpus (26% retrieval failures vs 7% for BM25)

**Hypothesis**: Dense retrieval benefits from large, diverse corpora to learn representations. Small technical corpus favors exact keyword matching.

**Implication**: For specialized domains with <10K documents, consider sparse-only or hybrid with sparse weighting.

---

## 🧪 Running Evaluations

### Full RAGAS Evaluation

```bash
# Activate dev environment
pip install -r requirements-dev.txt

# Run evaluation (all 4 modes: hybrid, sparse, dense, none)
python evaluation/evaluate_rag.py

# Results saved to outputs/eval_results/eval_results_*.json
```

### Error Analysis

```bash
# Categorize failures by mode
python experiments/analyze_errors.py

# Output: outputs/eval_results/error_analysis_summary.json
```

### Smoke Test (Quick Validation)

```bash
# Test 5 questions × 4 modes
python experiments/smoke_test.py
```

---

## 🎨 Features

### Current Features ✅

- [x] Hybrid retrieval (BM25 + Dense + RRF fusion)
- [x] FAISS vector store with L2 distance
- [x] OpenAI Responses API integration (gpt-4o-mini)
- [x] Ollama local LLM support
- [x] Streamlit web UI with model selection
- [x] RAGAS evaluation framework
- [x] Error analysis categorization
- [x] Docker deployment with health checks
- [x] Comprehensive documentation

### Future Improvements 🚀

**High Priority** (1-2 weeks):
- [ ] **Reranking**: Add cross-encoder (e.g., ms-marco-MiniLM) to improve ranking quality
- [ ] **Query Expansion**: Use LLM to generate query variations
- [ ] **Metadata Filtering**: Filter by paper year, authors, topic
- [ ] **Streaming Responses**: Implement async generation with progress indicators

**Medium Priority** (2-4 weeks):
- [ ] **Semantic Chunking**: Replace fixed 500-token chunks with semantic-based splitting (e.g., paragraph boundaries, sentence embeddings)
- [ ] **Adaptive Retrieval**: Route queries to sparse/dense based on keyword density
- [ ] **Larger Corpus**: Expand to 100+ papers, test scaling
- [ ] **REST API**: FastAPI backend for programmatic access
- [ ] **Observability**: Add logging, tracing (OpenTelemetry)

**Research Directions**:
- [ ] **Late Interaction**: Implement ColBERT-style token-level matching
- [ ] **Self-RAG**: Add retrieval-decision mechanism
- [ ] **Hypothetical Document Embeddings (HyDE)**: Generate synthetic docs for better dense retrieval

---

## 📈 System Characteristics

### Storage (Measured)

- FAISS Index: 2.0MB (1,395 × 384-dim float32 vectors)
- Chunks Metadata: 3.0MB (chunks.jsonl with text + metadata)
- Docker Image: 2.2GB (Python 3.10 + dependencies)

### Performance Notes

- **Latency**: Dominated by LLM generation (retrieval is fast for 1.4K chunks)
- **Scalability**: Current implementation loads BM25 corpus in RAM - suitable for <10K documents
- **Cost**: <$2 total (testset generation + evaluation runs with gpt-4o-mini)

---

## 🧰 Development

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format
black src/ tests/

# Lint
flake8 src/ tests/
```

### Build Index from Scratch

```bash
# Download papers to data/pdfs/ first
python scripts/build_index.py

# Outputs:
# - data/processed/rag_index.faiss
# - data/processed/chunks.jsonl
```

### Generate New Testset

```bash
# Requires Ollama running
python scripts/generate_testset.py

# Output: data/eval/ragas_testset.jsonl
```

---

## 📚 Key Papers & References

**Core RAG Papers**:
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)
- [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) (Izacard & Grave, 2020)

**Evaluation**:
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) (Es et al., 2023)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)

**Retrieval Methods**:
- [BM25 Okapi](https://en.wikipedia.org/wiki/Okapi_BM25) (Robertson & Zaragoza, 2009)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! Please:
1. Open an issue for bugs or feature requests
2. Include evaluation results for proposed changes
3. Follow existing code style (black, flake8)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

**Note on Dependencies**:
- **Runtime code** (src/, app.py): MIT licensed
- **Build scripts** (scripts/): Use [PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF4LLM) (AGPL-3.0) for PDF parsing
- **Prebuilt artifacts** (data/processed/): Created using AGPL tools, but runtime app doesn't execute AGPL code
- If you rebuild indexes from scratch, you'll need PyMuPDF4LLM (see requirements-dev.txt)

---

---

## 🎯 Interview Talking Points

**Q: "Walk me through your RAG system design."**

> I built a hybrid RAG system combining BM25 sparse retrieval with dense embeddings via RRF fusion. The key insight came from evaluation—on a small technical corpus (1.4K chunks), BM25 achieved 87% context recall vs 69% for dense-only, so hybrid fusion improved answer quality to 67%. I used RAGAS for rigorous evaluation across 42 questions with metrics like answer correctness, faithfulness, and context recall. Error analysis revealed dense embeddings struggle on small corpora (26% retrieval failures), teaching me that corpus size and domain specificity impact retrieval method selection.

**Q: "What was your biggest technical challenge?"**

> Testset quality. Initially, I generated questions from 500-token chunks, which produced 46% low-quality questions from bibliography sections because RAGAS couldn't distinguish references from content. I fixed this by regenerating from whole documents with reference filtering, improving quality from 5.5/10 to 8.5/10. This taught me that testset generation and retrieval chunking have different requirements—RAGAS needs document structure for multi-hop reasoning, not isolated chunks.

**Q: "How would you improve this system?"**

> Three priorities: (1) Add cross-encoder reranking to improve precision—error analysis showed 7-10% ranking issues where relevant chunks ranked below top-5. (2) Implement adaptive routing—queries with high keyword density go to BM25, conceptual queries to dense, based on my finding that hybrid helps only when both contribute. (3) Scale the corpus to 100+ papers and test if dense embeddings improve—my hypothesis is the 26% retrieval failure rate would decrease with more diverse data.

---

**Built with ❤️ for rigorous ML engineering**
