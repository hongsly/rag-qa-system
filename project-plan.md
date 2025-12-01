# RAG Q&A System - Project Plan

**Created**: 2025-11-22 (Day 26, Week 4 Day 5)
**Timeline**: Option B - Full Project 4 (~12 hours over 2 weeks)
**Data Source**: ArXiv Papers (RAG/LLM domain)

---

## Problem Statement

Build a production-quality RAG system for question-answering over recent ArXiv papers on RAG and LLM techniques. Demonstrate:
- Hybrid retrieval (dense + sparse + RRF fusion)
- Automated evaluation with Ragas framework
- Docker deployment to cloud
- Complete senior MLE portfolio piece

---

## Architecture Overview

```
ArXiv Papers (PDF) → Chunking (500 tokens, 50 overlap)
                           ↓
              Sentence-BERT Embeddings (384-dim)
                           ↓
                    ┌──────┴──────┐
                    ↓             ↓
              FAISS Index      BM25 Index
             (Dense retrieval) (Sparse retrieval)
                    ↓             ↓
                    └──────┬──────┘
                           ↓
                  RRF Fusion (k=60)
               Score = Σ 1/(k + rank_i)
                           ↓
                    Top-K Documents
                           ↓
              GPT-3.5-turbo + Context
                           ↓
                  Generated Answer + Citations
```

---

## Tech Stack Decisions

### Core Components
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
  - Why: Free, fast, 384-dim works well, proven for RAG
  - Alternative considered: OpenAI embeddings (too expensive for portfolio)

- **Dense Retrieval**: FAISS (local index)
  - Why: Fast, battle-tested, good for <100K docs
  - Alternative: Chroma (heavier dependency)

- **Sparse Retrieval**: rank-bm25 library
  - Why: Pure Python, no server needed, sufficient for portfolio
  - Alternative: Elasticsearch (overkill for 20-30 papers)

- **Fusion**: Hand-coded RRF
  - Why: Simple (5 lines), demonstrates understanding
  - Formula: Score = Σ 1/(60 + rank_i)

- **LLM**: OpenAI API (gpt-3.5-turbo)
  - Why: Reliable, fast, cheap ($0.50 for 5K queries)
  - Alternative: Ollama (slower, local hassle for portfolio)

- **Evaluation**: Ragas + manual metrics
  - Ragas: Context precision, recall, faithfulness, answer relevance
  - Manual: Recall@K, MRR, NDCG for retrieval

- **Deployment**: Docker + Streamlit Cloud
  - Why: Free hosting, easy to share, professional
  - Alternative: AWS Lambda (more complex)

### Development Tools
- **Version Control**: Git + GitHub
- **Environment**: Python 3.10+ with venv
- **CI/CD**: GitHub Actions (linting + tests)
- **Monitoring**: Simple logging to file

---

## Data Source: ArXiv Papers

### Target Papers (20-30 papers on RAG/LLMs)

**Search queries on arxiv.org:**
1. "Retrieval Augmented Generation" (2023-2024)
2. "RAG evaluation" OR "RAG metrics"
3. "Hybrid retrieval" OR "dense sparse retrieval"
4. "Query rewriting" OR "Query decomposition"
5. "LLM hallucination" OR "Faithfulness"

**Recommended papers to download** (pick 20-30):
- FiD (Fusion-in-Decoder) - Izacard et al.
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
- Lost in the Middle (Liu et al., 2023)
- RAFT (Gorilla paper, 2024)
- ColBERT: Efficient and Effective Passage Search (Khattab & Zaharia, 2020)
- Ragas: Automated Evaluation of RAG (2023)
- Self-RAG (Asai et al., 2023)
- RAPTOR: Recursive Abstractive Processing (2024)
- GraphRAG (Microsoft, 2024)
- Recent survey papers on RAG (2024)
- Papers on query rewriting/decomposition
- Papers on reranking strategies
- Papers on long-context vs. RAG

**Download strategy**:
- Use arxiv.org search + filter by date (2023-2024)
- Download PDFs to `data/raw/`
- Total size: ~50-100 MB (acceptable)

---

## Implementation Timeline

### Weekend (Light Sessions)

**Day 26 (Sat, Nov 22) - 30 min** ✅
- [x] Create project structure
- [x] Write this project-plan.md
- [x] Decision: Option B confirmed

**Day 27 (Sun, Nov 23) - 30 min**
- [x] Download 20-30 ArXiv papers (PDFs to `data/raw/`)
- [x] Create folder structure:
  ```
  rag-qa-system/
  ├── data/
  │   ├── raw/              # PDFs
  │   ├── processed/        # Chunks (JSON)
  │   └── eval/             # Test questions
  ├── src/
  ├── evaluation/
  ├── tests/
  └── outputs/
  ```
- [x] Create `requirements.txt` stub (list libraries, don't install yet)

### Main Implementation

** Mon, Nov 24, Week 4 Day 7 - 2 hours**
- [x] `src/data_loader.py`: Parse PDFs, chunk by 500 tokens with 50 overlap
- [x] `src/vector_store.py`: Generate embeddings, build FAISS index, save to disk
- [x] Test: Search for 1 query, verify top-5 results
- [x] Commit: "Add data loading and embedding generation"

** Tue, Nov 25, Week 5 Day 1 (Day 29) - 2.5 hours** ✅ **COMPLETE**
- [x] `src/sparse_retrieval.py`: BM25 with NLTK tokenization
- [x] `src/hybrid_search.py`: RRF fusion (k=60, retrieve 4×k candidates)
- [x] Evaluation framework: Precision@5, MRR
- [x] Test: Compare dense vs BM25 vs hybrid on 2 query sets
  - General NLP queries: Hybrid 40% < Dense 60% (query-corpus mismatch)
  - RAG-focused queries: Hybrid 80% > Dense 67% ✅ (aligned queries)
- [x] Key finding: Query-corpus alignment is critical for BM25 performance
- [x] Folder reorganization: Created `experiments/` for analysis scripts
- [x] Commit: "Add hybrid retrieval with RRF fusion"

**Decision**: Use **Hybrid (RRF)** for production - performs better (80% vs 67%) with RAG-focused queries
**Rationale**: User queries will be RAG-related (e.g., "How does ColBERT work?"), not general NLP
**See**: `references/day29-hybrid-retrieval-findings.md` for details

** Wed, Nov 26, Week 5 Day 2 - 2 hours** ✅ **COMPLETE**
- [x] `src/generator.py`: OpenAI Responses API wrapper (gpt-4o-mini, prompt engineering)
- [x] `src/rag_pipeline.py`: End-to-end pipeline (RagAssistant with 4 modes)
- [x] Create test question set (10 questions in `data/eval/test_questions.json`)
  - 3 simple factual ✅
  - 3 complex reasoning ✅
  - 2 multi-hop ✅
  - 2 negative (not in corpus) ✅
- [x] Smoke test: 5 questions × 4 modes = 20 tests
  - Results: Citations excellent, token usage validated (2700 vs 50)
  - Issue discovered: Negative question handling (retrieval contamination)
- [x] Commit: "Add generation and end-to-end pipeline"

**Key Decision**: Used gpt-4o-mini ($0.15 input / $0.60 output) instead of gpt-3.5-turbo - 3× cheaper + better quality

** Thu, Nov 27, Week 5 Day 3 - 1 hour** ✅ **PLANNING & COST ANALYSIS**
- [x] Add ArXiv metadata to chunks (title, authors, year, URL) - 30 min
- [x] Regenerate chunks with metadata (`scripts/build_index.py`)
- [x] Researched Ragas 0.3.9 API (`generate_with_langchain_docs`, gpt-4o-mini setup)
- [x] Investigated Ollama support (not reliable - missing `agenerate_prompt`)
- [x] Ragas cost underestimate
  - Test run: 200 chunks → 200K tokens → $0.70 (SummaryExtractor phase only) (but was using gpt-4o instead of gpt-4o-mini)
- [x] Analyzed manual vs Ragas test format differences
- [x] Discussed ground truth requirements for metrics
- [x] **Daily knowledge check**: 94% (A) - Excellent overdue item retention
- Implementation deferred to Day 5 (evaluation code, run metrics)
- Decision: Sample 250 representative chunks instead of all 1500 → 8× cost savings ($1.25 vs $10-15)

** Fri, Nov 28, Week 5 Day 4 - 2 hours** ✅ **RETRIEVAL EVALUATION**
- [x] Add reference filtering to `CorpusLoader.filter_reference_chunks()` (Ollama-based)
- [x] Rebuild index with filtered chunks (1395 remaining, 9.5% references removed)
- [x] Implement sampling: `_sample_chunks()` in generate_testset.py
- [x] Generate 42 Ragas questions with Ollama (free, exceeded target of 40)
- [x] Create `evaluation/evaluate_retrieval.py`: Recall@K, MRR, NDCG
- [x] Run retrieval evaluation on 41 questions (3 modes: sparse, dense, hybrid)
- [x] **Critical insight**: Sampled testset → incomplete ground truth (metrics are lower bounds)
- RAG evaluation deferred to Day 6 (use LLM-based context_recall)
- Error analysis deferred to Day 6

**Total cost**: $0 (Ollama for filtering + generation)

** Sat, Nov 29, Week 5 Day 5 - 3 hours** ✅ **RAG EVALUATION & ERROR ANALYSIS**
- [x] Run RAG evaluation on 10 manual + 41 Ragas questions (4 modes: sparse, dense, hybrid, none)
  - Initial results: answer_correctness 0.39-0.58 (unexpectedly low)
- [x] **Question quality analysis**: Discovered 46% of Ragas questions were low-quality
  - Created `experiments/analyze_question_quality.py` (citation pattern detection)
  - Found 19/41 suspicious questions (from bibliography/table/footnote chunks)
  - **Root cause identified**: Generated from 500-token chunks instead of whole documents
  - Ragas needs whole documents to build knowledge graph
- [x] Manual review and filtering
  - Created `experiments/review_suspicious_questions.py` (interactive review)
  - Categorized: 11 definitely bad, 4 moderate, 4 contaminated, 1 false positive
  - Filtered 13 low-quality questions → 28 clean questions (68% retention)
  - Created `experiments/filter_ragas_testset.py` (generates filtered testset)
- [x] Metrics recalculation on filtered testset
  - Created `experiments/filter_and_recalculate.py`
  - ~13% average improvement across all metrics
  - SPARSE: 66.8% answer_correctness, HYBRID: 62.8%, DENSE: 53.0%
- [x] **Error analysis**: Categorize failure modes
  - Created `experiments/analyze_errors.py` (pattern-based categorization)
  - Key finding: **Dense 29.6% retrieval failures vs Sparse 10.7%** (3× worse!)
  - SPARSE success rate: 57.1% (best), HYBRID: 46.4%, DENSE: 25.9%
  - Failure patterns: retrieval failure, generation failure, hallucination, ranking issue
- [x] **Decision**: Default to SPARSE (best performance), keep HYBRID as option (highest recall 92%)

**Key Insights**:
1. Testset generation methodology matters: chunks vs whole documents is critical
2. SPARSE > DENSE for small technical corpus (keyword matching advantage)
3. Question quality filtering improved metrics by ~13%

**Status**: RAG evaluation complete, ready for UI + deployment

** Sun, Nov 30, Week 5 Day 6 - 2.5 hours** ✅ **STREAMLIT UI + DOCKER DEPLOYMENT**
- [x] **Testset regeneration with whole documents**
  - Fixed Day 6 root cause: used PyMuPDFLoader for whole documents (not chunks)
  - Generated 40 questions with Ollama (qwen2.5-coder:7b), filtered to 32 clean
  - Quality: 5.5/10 → 8.5/10, no questions from references (0% vs 46%)
  - Unique: 97.5% (vs 54% before), 1 duplicate removed
- [x] **Added reference answers to manual questions**
  - Updated `data/eval/test_questions.json` with comprehensive references
  - 10 manual questions now have LLM-gradable references
- [x] **RAG evaluation v2 (42 questions: 10 manual + 32 Ragas)**
  - HYBRID best: 66.9% answer_correctness, 52.4% success rate
  - SPARSE: 61.3% correctness, 87.3% recall (highest)
  - DENSE: 51.9% correctness, 23.8% success (worst)
  - Validated Day 6 findings: Dense 3.7× worse retrieval failures
- [x] **Error analysis v2**: Categorized 42 questions by failure patterns
  - Created `experiments/analyze_errors.py` (updated for new file structure)
  - HYBRID: 52% success, SPARSE: 43%, DENSE: 24%
  - Dense retrieval failures: 26% (confirmed 3-4× worse than SPARSE)
- [x] **File structure reorganization**
  - Created `outputs/eval_results/` for clean input/output separation
  - Updated `src/utils.py` with EVAL_OUTPUT_DIR
  - Updated all evaluation scripts to use new structure
- [x] **Streamlit UI complete**
  - Created `app.py` with mode selection, top-K config, example questions
  - Displays answer + retrieved chunks with scores
  - Fixed BM25 multiprocessing issue (pre-tokenize corpus)
  - Sidebar: retrieval mode, top K, model, system info
- [x] **Docker containerization**
  - Created `Dockerfile` (learned Docker basics: FROM, COPY, RUN, CMD, ENV)
  - Created `docker-compose.yml` for orchestration
  - Created `.dockerignore` (exclude venv/, __pycache__, raw PDFs)
  - Split `requirements-dev.txt`
  - Image size: 4.3GB → 2.2GB (-49% optimized)
- [x] **Comprehensive README.md** created
  - Architecture, features, evaluation results, technical decisions
  - Quick start, usage examples, development setup
  - License note (MIT + PyMuPDF4LLM AGPL)
- [x] **Documentation updates**
  - Updated `experiments/README.md` with Day 34 experiments
  - Created `references/Day34-Quick-Reference.md`

**Key Insights**:
1. Whole documents vs chunks critical for testset generation (quality 5.5→8.5)
2. HYBRID best for production (67% correctness, 52% success)
3. Docker optimization: .dockerignore + slim image + dev deps split

**Status**: RAG project production-ready (90% complete)
**Next**: Push to GitHub, optional FastAPI + observability

---

## Code Structure (Full Project 4)

```
rag-qa-system/
├── README.md                      # Comprehensive documentation
├── requirements.txt              # All dependencies with versions
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Optional: multi-service
├── .env.example                  # API keys template
├── .gitignore                    # Don't commit data, .env
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions (lint + test)
├── data/
│   ├── raw/                      # 20-30 ArXiv PDFs
│   ├── processed/                # Chunked docs (JSON lines)
│   └── eval/                     # Test question sets
│       └── test_questions.json   # 10 test questions with ground truth
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # PDF parsing, chunking
│   ├── embeddings.py             # Sentence-BERT wrapper
│   ├── vector_store.py           # FAISS operations
│   ├── retriever.py              # Dense + BM25 + RRF fusion
│   ├── generator.py              # OpenAI API wrapper
│   ├── rag_pipeline.py           # End-to-end pipeline
│   └── api.py                    # FastAPI endpoint (optional)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_retrieval.py    # Recall@K, MRR, NDCG
│   ├── evaluate_rag.py           # Ragas integration
│   ├── error_analysis.py         # Failure categorization
│   └── cost_analysis.py          # API cost tracking
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py          # Unit tests
│   └── test_api.py               # API tests
├── app.py                        # Streamlit UI
├── notebooks/                    # Optional
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_comparison.ipynb
│   └── 03_retrieval_tuning.ipynb
└── outputs/
    ├── eval_results/             # Evaluation metrics and reports
    │   ├── retrieval_metrics.json
    │   ├── ragas_scores.json
    │   └── error_analysis.json
    └── logs/                     # Query logs and monitoring
```

**Estimated lines of code**: ~800 lines (excluding notebooks)

---

## Evaluation Plan

### Test Question Set (10 questions)

**Simple Factual (3 questions)**:
1. "What is Retrieval-Augmented Generation?"
2. "Who proposed the FiD architecture?"
3. "What does RAFT stand for?"

**Complex Reasoning (3 questions)**:
4. "Why does hybrid retrieval (dense + sparse) outperform either approach alone?"
5. "How does ColBERT differ from traditional dense retrieval?"
6. "What are the trade-offs between long-context LLMs and RAG systems?"

**Multi-hop (2 questions)**:
7. "How do GraphRAG and FiD differ in their approach to multi-document reasoning?"
8. "What evaluation metrics are recommended for both retrieval and generation in RAG?"

**Negative (2 questions)**:
9. "What is the capital of France?" (not in corpus)
10. "How do you train a neural network?" (not in corpus)

### Metrics to Track

**Retrieval Metrics** (compare dense, BM25, hybrid):
- Recall@K (K=1,3,5,10): % of questions with correct doc in top-K
- MRR (Mean Reciprocal Rank): 1/rank of first correct doc
- NDCG: Normalized Discounted Cumulative Gain
- Precision@K: % of relevant docs in top-K

**Ragas Metrics** (automated LLM-as-judge):
- Context Precision: Are retrieved contexts relevant to question?
- Context Recall: Does retrieved context contain answer?
- Faithfulness: Is answer grounded in context (no hallucination)?
- Answer Relevance: Does answer address the question?
- Answer Correctness: Semantic similarity with ground truth

**Cost Metrics**:
- Total API calls (embeddings + generation + evaluation)
- Tokens used per query
- Cost per query, cost per 1K queries

### Expected Results

**Retrieval** (based on 99.2% RAG mastery):
- Dense-only: Recall@5 ≈ 70-80%
- BM25-only: Recall@5 ≈ 60-70%
- Hybrid+RRF: Recall@5 ≈ 85-95% ⭐ (best)

**Ragas Scores** (target):
- Context Precision: >0.85
- Context Recall: >0.90
- Faithfulness: >0.90
- Answer Relevance: >0.85

---

## Interview Talking Points

After building this, you can say:

**"I built a production-ready hybrid RAG system with rigorous evaluation, error analysis, and Docker deployment for Q&A over ML research papers."**

**Architecture**:
- **Corpus**: 32 ArXiv papers on RAG/LLMs → 1,395 chunks (500 tokens, 50 overlap)
- **Retrieval**: Hybrid with RRF fusion (k=60)
  - Dense: SentenceBERT (all-MiniLM-L6-v2) in FAISS
  - Sparse: BM25-Okapi with NLTK tokenization
- **Generation**: GPT-4o-mini with structured prompt for citations
- **Key insight**: On small technical corpus (1.4K chunks), BM25 achieved 87% context recall vs 69% dense-only. Hybrid fusion improved answer quality to 67%.

**Evaluation rigor**:
- **Testset**: 42 questions (10 manual + 32 Ragas-generated) with reference answers
- **Metrics**: 5 RAGAS metrics - answer_correctness, context_recall, faithfulness, answer_relevancy, context_precision
- **Results**: HYBRID 66.9% correctness (52.4% success), SPARSE 61.3%, DENSE 51.9%
- **Error analysis**: Categorized failures across modes
  - Retrieval failure: DENSE 3.7× worse (26%) vs SPARSE (7%)
  - Generation failure: 10-19% across all modes
  - Ranking issues: 7-10% (opportunity for cross-encoder reranking)
- **Key lesson**: Dense embeddings struggle on small technical corpora - keyword matching wins

**Production readiness**:
- **Docker**: Containerized with docker-compose, optimized from 4.3GB → 2.2GB
- **UI**: Streamlit with mode selection (sparse/hybrid/dense), top-K config, OpenAI/Ollama support
- **Deployment**: Health checks, environment config, prebuilt indexes included
- **Documentation**: Comprehensive README, experiments log, quick reference sheets

**What I'd improve next**:
1. **Cross-encoder reranking**: Address 7-10% ranking issues (retrieve 20 → rerank → top 5)
2. **Semantic chunking**: Replace fixed 500-token chunks with paragraph-based splitting
3. **Adaptive routing**: Route keyword-heavy queries to BM25, conceptual to dense
4. **Scale corpus**: Test if 100+ papers reduce dense retrieval failures (26% → <10%)

---

## Success Criteria

**Technical** (All Achieved ✅):
- ✅ End-to-end RAG pipeline working (sparse/dense/hybrid/none modes)
- ✅ Hybrid retrieval (dense + BM25 + RRF) implemented correctly with k=60
- ✅ Ragas evaluation framework integrated with 5 metrics
- ✅ Context recall ≥ 85% (SPARSE: 87.3%, HYBRID: 83.3%)
- ✅ Faithfulness score ≥ 0.80 (SPARSE: 83.8%, HYBRID: 83.0%)
- ✅ Dockerized with optimization (4.3GB → 2.2GB, -49%)
- ✅ Streamlit UI with mode selection and Ollama support
- ✅ Comprehensive README with architecture, evaluation results, interview talking points

**Portfolio** (All Achieved ✅):
- ✅ Demonstrates senior MLE skills (evaluation rigor, error analysis, production deployment)
- ✅ Shows RAG mastery (99.2% from Week 4 studies + practical implementation)
- ✅ Interview-ready: Can explain architecture, trade-offs, evaluation, testset quality issues
- ✅ Meets all Project 4 requirements from Project-Ideas.md
- ✅ Prebuilt indexes included for easy demo
- ✅ Error analysis with failure categorization (retrieval/generation/ranking/partial)

**Timeline** (Completed ✅):
- ✅ Day 1 (Nov 25): Data ingestion, chunking, vector store setup
- ✅ Day 2 (Nov 26): Sparse retrieval (BM25), hybrid search (RRF)
- ✅ Day 3 (Nov 27): Generation with GPT-4o-mini, end-to-end pipeline
- ✅ Day 4 (Nov 28): Ragas testset generation (40 questions)
- ✅ Day 5 (Nov 29): RAG evaluation (4 modes), retrieval-only evaluation
- ✅ Day 6 (Nov 30): Testset regeneration (whole docs), re-evaluation (42 questions), error analysis v2, Streamlit UI, Docker deployment
- ✅ **Total time**: ~15 hours (5 days, ~3 hours/day)

**Actual Results**:
- **Answer Correctness**: HYBRID 66.9% (best), SPARSE 61.3%, DENSE 51.9%
- **Context Recall**: SPARSE 87.3% (best), HYBRID 83.3%, DENSE 69.4%
- **Success Rate** (correctness >0.7): HYBRID 52.4%, SPARSE 42.9%, DENSE 23.8%
- **Testset Quality**: 8.5/10 after regeneration from whole documents (was 5.5/10 from chunks)

---

## Notes

- This is Option B: Full Project 4 with all mandatory components (all completed)
- Quality over features approach worked well - focused on rigorous evaluation before advanced features
- Documented everything for interview storytelling (README, experiments log, quick reference sheets)
- Prebuilt indexes and Docker deployment enable easy demo without rebuild
- Actual cost: <$2 for OpenAI API (testset generation + evaluation runs) - very cost-effective

**Key Learnings**:
1. **Testset generation ≠ Retrieval chunking**: Ragas needs whole documents for knowledge graph, not isolated chunks
2. **Dense embeddings need scale**: On small corpus (1.4K chunks), BM25 outperforms (87% vs 69% recall)
3. **Hybrid helps when both contribute**: If one method dominates, fusion may not improve (or degrade)
4. **Error analysis drives improvements**: Identified 7-10% ranking issues → cross-encoder reranking is clear next step

---

**Status**: ✅ **PROJECT COMPLETE** (90% overall, production-ready)

**Remaining Optional Improvements**:
- [ ] Push to GitHub (separate portfolio repo)
- [ ] Optional: Deploy to Streamlit Cloud
- [ ] Optional: FastAPI backend + observability (LangSmith/Langfuse)
- [ ] Optional: Regenerate testset with GPT-4o-mini (higher quality than Ollama)

**Ready for interviews**: Can explain architecture, evaluation methodology, error analysis, and production deployment in 5-10 minutes. 🚀
