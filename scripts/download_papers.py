"""
Download ArXiv papers for RAG Q&A system corpus
Usage: python download_papers.py
"""

import time
import urllib.request
from pathlib import Path

# Target directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ArXiv paper IDs and short names
PAPERS = [
    # Core RAG Papers (Must-have)
    ("2005.11401", "rag_lewis"),
    ("2007.01282", "fid_izacard"),
    ("2307.03172", "lost_in_middle_liu"),
    ("2310.11511", "self_rag_asai"),
    ("2401.18059", "raptor_sarthi"),
    ("2403.10131", "raft_zhang"),
    ("2404.16130", "graphrag_edge"),

    # Retrieval Methods
    ("2004.12832", "colbert_khattab"),
    ("2004.04906", "dpr_karpukhin"),
    ("2107.05720", "splade_formal"),
    ("2104.05740", "hybrid_retrieval_ma"),

    # Evaluation & Metrics
    ("2309.15217", "ragas_es"),
    ("2311.09476", "ares_saad_falcon"),
    ("2309.01431", "rgb_benchmark_chen"),

    # Query Processing
    ("2305.14283", "query_rewriting_ma"),
    ("2310.06117", "step_back_zheng"),
    ("2212.10496", "hyde_gao"),

    # Advanced RAG
    ("2305.06983", "active_rag_jiang"),
    ("2212.10509", "cot_retrieval_trivedi"),
    ("2007.15651", "reranking_nogueira"),

    # Long Context vs RAG
    ("2402.13116", "long_context_rag_xu"),

    # Survey Papers
    ("2312.10997", "rag_survey_gao"),
    ("2202.01110", "rag_survey_li"),
    ("2312.00752", "rag_roadmap_asai"),

    # Multi-hop & Reasoning
    ("2009.12756", "multihop_qi"),
    ("2210.03629", "react_yao"),

    # Document Processing
    ("2410.02525", "contextual_embeddings_nussbaum"),

    # Hallucination
    ("2305.14552", "hallucination_manakul"),

    # 2025
    ("2411.06037", "sufficient_context_joren"),
    ("2508.21038", "embedding_limit_weller"),
    ("2412.12881", "deliberative_rag_jiang"),
    ("2410.10594", "vision_rag_yu")
]

def download_paper(arxiv_id, short_name):
    """Download a paper from ArXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{arxiv_id}_{short_name}.pdf"
    filepath = DATA_DIR / filename

    # Skip if already downloaded
    if filepath.exists():
        print(f"✓ Already exists: {filename}")
        return True

    try:
        print(f"Downloading: {filename}...", end=" ")
        urllib.request.urlretrieve(url, filepath)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def main():
    print(f"Starting download of {len(PAPERS)} papers to {DATA_DIR}/")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for arxiv_id, short_name in PAPERS:
        if download_paper(arxiv_id, short_name):
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting: be nice to ArXiv servers
        time.sleep(1)

    print("=" * 60)
    print(f"Download complete!")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  Total: {success_count + fail_count}")
    print(f"\nPapers saved to: {DATA_DIR.absolute()}")

if __name__ == "__main__":
    main()
