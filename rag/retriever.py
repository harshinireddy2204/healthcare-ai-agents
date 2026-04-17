"""
rag/retriever.py

Retrieval interface for clinical guidelines — uses OpenAI embeddings.

Memory footprint:
  - No torch (~300 MB saved)
  - No sentence-transformers (~90 MB saved)
  - No cross-encoder reranker (~85 MB saved)
  - No transformers tokenizers (~200 MB saved)
  Total: ~675 MB saved vs the previous version that OOM-killed on Railway.

Retrieval quality:
  OpenAI text-embedding-3-small is MTEB-competitive with MiniLM for retrieval.
  We keep the bidirectional synonym expansion (critical for short queries
  like "diabetes" or "insulin") and rely on cosine similarity alone — the
  cross-encoder reranker is gone to save memory, and it wasn't essential.
"""
import logging
import os
import warnings
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_guidelines"
COLLECTION_NAME = "clinical_guidelines"
EMBEDDING_MODEL = "text-embedding-3-small"

_client = None
_collection = None
_openai_client = None


# ── Clinical synonym expansion ────────────────────────────────────────────────
CLINICAL_SYNONYM_GROUPS: list[list[str]] = [
    ["diabetes", "diabetes mellitus", "DM", "T2DM", "T1DM", "Type 2 Diabetes", "Type 1 Diabetes", "hyperglycemia"],
    ["hba1c", "hemoglobin A1c", "A1c", "glycated hemoglobin", "glycemic control", "glycemic goals"],
    ["insulin", "insulin therapy", "insulin pump", "CSII", "basal bolus", "insulin delivery", "MDI"],
    ["metformin", "biguanide", "glucophage", "first-line diabetes"],
    ["glp-1", "glp1", "semaglutide", "liraglutide", "ozempic", "wegovy", "GLP-1 agonist"],
    ["sglt2", "sglt-2", "empagliflozin", "dapagliflozin", "jardiance", "SGLT2 inhibitor"],
    ["cgm", "continuous glucose monitor", "glucose sensor", "dexcom", "libre"],
    ["ckd", "chronic kidney disease", "renal insufficiency", "kidney disease", "nephropathy"],
    ["egfr", "estimated glomerular filtration rate", "GFR", "renal function", "kidney function"],
    ["nephrology", "kidney specialist", "nephrologist", "renal clinic"],
    ["dialysis", "hemodialysis", "peritoneal dialysis", "ESRD", "RRT", "renal replacement therapy"],
    ["proteinuria", "albuminuria", "UACR", "urine albumin", "microalbuminuria"],
    ["hypertension", "high blood pressure", "elevated BP", "HTN", "systolic", "diastolic"],
    ["heart failure", "HF", "HFrEF", "HFpEF", "cardiomyopathy", "reduced ejection fraction", "EF"],
    ["atrial fibrillation", "afib", "AF", "arrhythmia"],
    ["coronary artery disease", "CAD", "angina", "chest pain", "ischemic heart disease"],
    ["stroke", "TIA", "ischemic stroke", "cerebrovascular"],
    ["cholesterol", "LDL", "HDL", "lipid", "lipid panel", "hyperlipidemia", "dyslipidemia"],
    ["statin", "atorvastatin", "rosuvastatin", "simvastatin", "lipid lowering"],
    ["anticoagulation", "warfarin", "DOAC", "apixaban", "rivaroxaban", "blood thinner"],
    ["aspirin", "antiplatelet", "ASA"],
    ["mammogram", "mammography", "breast cancer screening", "breast imaging", "biennial"],
    ["colonoscopy", "colorectal screening", "colon cancer", "FIT test", "fecal immunochemical"],
    ["pap smear", "cervical cancer", "HPV test", "colposcopy"],
    ["psa", "prostate specific antigen", "prostate cancer"],
    ["dexa", "bone density", "bone mineral density", "osteoporosis screening", "T-score"],
    ["flu vaccine", "influenza", "annual flu shot", "influenza vaccine"],
    ["pneumonia vaccine", "pneumococcal", "PCV", "PPSV23", "Prevnar"],
    ["shingles vaccine", "shingrix", "zoster vaccine", "herpes zoster"],
    ["covid vaccine", "coronavirus vaccine", "covid-19 vaccine"],
    ["depression", "MDD", "depressive disorder", "PHQ-9", "PHQ9", "major depressive", "SSRI", "antidepressant"],
    ["anxiety", "GAD", "GAD-7", "generalized anxiety", "panic"],
    ["obesity", "overweight", "BMI 30", "BMI 35", "weight management", "bariatric"],
    ["weight loss", "weight reduction", "caloric deficit"],
    ["copd", "chronic obstructive", "emphysema", "chronic bronchitis"],
    ["asthma", "bronchospasm", "inhaled corticosteroid"],
    ["spirometry", "FEV1", "FVC", "pulmonary function"],
    ["breast cancer", "tamoxifen", "HER2", "aromatase inhibitor", "lumpectomy"],
    ["prostate cancer", "bicalutamide", "ADT", "androgen deprivation"],
    ["rheumatoid arthritis", "RA", "methotrexate", "DMARD", "TNF inhibitor", "biologic"],
    ["lupus", "SLE", "hydroxychloroquine", "ANA"],
    ["osteoporosis", "bone loss", "bisphosphonate", "alendronate", "fracture risk"],
]


def expand_query(query: str) -> str:
    """Bidirectional synonym expansion."""
    query_lower = query.lower()
    additions: list[str] = []

    for group in CLINICAL_SYNONYM_GROUPS:
        matched = any(term.lower() in query_lower for term in group)
        if matched:
            for term in group:
                if term.lower() not in query_lower:
                    additions.append(term)

    if additions:
        seen = set()
        unique = []
        for t in additions:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        return query + " " + " ".join(unique)

    return query


# ── Lazy loaders ──────────────────────────────────────────────────────────────

def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _embed_query(query: str) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve_guidelines(
    query: str,
    category: Optional[str] = None,
    n_results: int = 3,
    min_relevance: float = 0.20,
    expand: bool = True,
    use_reranker: bool = False,  # no-op — kept for backward compat
) -> list[dict]:
    """
    Retrieve relevant clinical guideline chunks for a query.
    Uses OpenAI text-embedding-3-small for semantic search against ChromaDB.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return [{
            "text": "No guidelines loaded yet. Guidelines seed embedding may be in progress.",
            "source_name": "System",
            "url": "",
            "category": "",
            "scraped_at": "",
            "relevance_score": 0.0,
            "rerank_score": 0.0
        }]

    search_query = expand_query(query) if expand else query

    try:
        query_embedding = _embed_query(search_query)
    except Exception as e:
        return [{
            "text": f"Embedding API error: {e}",
            "source_name": "Error",
            "url": "", "category": "", "scraped_at": "",
            "relevance_score": 0.0, "rerank_score": 0.0
        }]

    where = {"category": category} if category else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * 2, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return [{
            "text": f"ChromaDB query error: {e}",
            "source_name": "Error",
            "url": "", "category": "", "scraped_at": "",
            "relevance_score": 0.0, "rerank_score": 0.0
        }]

    if not results["documents"] or not results["documents"][0]:
        return []

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        cosine_score = 1 - dist
        if cosine_score >= min_relevance:
            candidates.append({
                "text": doc,
                "source_name": meta.get("source_name", "Unknown"),
                "url": meta.get("url", ""),
                "category": meta.get("category", ""),
                "scraped_at": meta.get("scraped_at", "")[:10],
                "relevance_score": round(cosine_score, 3),
                "rerank_score": cosine_score
            })

    # Fallback: if threshold filtered everything, return top result
    if not candidates and results["documents"][0]:
        doc = results["documents"][0][0]
        meta = results["metadatas"][0][0]
        dist = results["distances"][0][0]
        candidates = [{
            "text": doc,
            "source_name": meta.get("source_name", "Unknown"),
            "url": meta.get("url", ""),
            "category": meta.get("category", ""),
            "scraped_at": meta.get("scraped_at", "")[:10],
            "relevance_score": round(1 - dist, 3),
            "rerank_score": 1 - dist
        }]

    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    return candidates[:n_results]


def format_citations(results: list[dict]) -> str:
    if not results:
        return "No relevant guidelines found."

    lines = ["**Clinical Guideline References:**\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. **{r['source_name']}** (retrieved {r['scraped_at']})\n"
            f"   Semantic match: {r['relevance_score']:.0%} | "
            f"[Source]({r['url']})\n"
            f"   > {r['text'][:300]}...\n"
        )
    return "\n".join(lines)


@tool
def search_clinical_guidelines(query: str, category: str = "") -> str:
    """
    Search the clinical guidelines knowledge base for evidence-based recommendations.
    Use this tool whenever you need to cite a specific guideline for a care gap,
    prior authorization criteria, or risk management decision.

    query: natural language search query (e.g. "HbA1c target type 2 diabetes")
    category: optional filter — preventive_screening, diabetes, cardiovascular,
              ckd, immunization, mental_health, etc.

    Returns formatted citations with source names, relevance scores, and URLs.
    """
    results = retrieve_guidelines(
        query=query,
        category=category if category else None,
        n_results=3
    )
    return format_citations(results)


GUIDELINE_TOOLS = [search_clinical_guidelines]


if __name__ == "__main__":
    test_queries = [
        ("diabetes", None),
        ("insulin", None),
        ("mammogram screening", "preventive_screening"),
        ("HbA1c target type 2 diabetes", "diabetes"),
    ]

    print("\n=== Guideline Retrieval Test (OpenAI embeddings) ===\n")
    for query, cat in test_queries:
        print(f"Query: '{query}' (category: {cat})")
        results = retrieve_guidelines(query, category=cat, n_results=2)
        if results:
            for r in results:
                print(f"  [{r['relevance_score']:.0%}] {r['source_name']}")
                print(f"  {r['text'][:150]}...")
        else:
            print("  No results")
        print()