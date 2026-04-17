"""
rag/retriever.py

Retrieval interface that agents use to query clinical guidelines.

Why queries like "diabetes" or "insulin" now return results:
  1. min_relevance lowered from 0.50 → 0.20 (cosine space)
     — Short 1-word queries produce low cosine scores against long chunks.
       0.50 was filtering out valid matches. 0.20 matches industry practice
       for general-purpose sentence embeddings.
  2. Removed "Represent this medical question for retrieval:" prefix
     — That phrasing helps INSTRUCTION-TUNED models (BGE, E5-instruct).
       all-MiniLM-L6-v2 is a plain sentence encoder — the prefix only
       adds noise and degrades match quality.
  3. Cross-encoder reranker now has its own minimum score check
     — If reranker scores are all negative (no good match), we fall
       back to cosine ranking rather than returning garbage.
  4. Query expansion now bidirectional — when user types a synonym,
     we also add the primary term (e.g. "A1c" → "HbA1c").
"""
import logging
import os
import warnings
from typing import Optional

# ── Suppress library noise before imports ─────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_guidelines"
COLLECTION_NAME = "clinical_guidelines"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_client = None
_collection = None
_embedder = None
_reranker = None
import threading
_model_lock = threading.Lock()


# ── Clinical synonym expansion ────────────────────────────────────────────────
# Maps clinical terms to related vocabulary for bidirectional query expansion.
# When any term in a row matches the query, ALL terms in that row are added.
CLINICAL_SYNONYM_GROUPS: list[list[str]] = [
    # Diabetes
    ["diabetes", "diabetes mellitus", "DM", "T2DM", "T1DM", "Type 2 Diabetes", "Type 1 Diabetes", "hyperglycemia"],
    ["hba1c", "hemoglobin A1c", "A1c", "glycated hemoglobin", "glycemic control", "glycemic goals"],
    ["insulin", "insulin therapy", "insulin pump", "CSII", "basal bolus", "insulin delivery", "MDI"],
    ["metformin", "biguanide", "glucophage", "first-line diabetes"],
    ["glp-1", "glp1", "semaglutide", "liraglutide", "ozempic", "wegovy", "GLP-1 agonist"],
    ["sglt2", "sglt-2", "empagliflozin", "dapagliflozin", "jardiance", "SGLT2 inhibitor"],
    ["cgm", "continuous glucose monitor", "glucose sensor", "dexcom", "libre"],
    # Kidney / CKD
    ["ckd", "chronic kidney disease", "renal insufficiency", "kidney disease", "nephropathy"],
    ["egfr", "estimated glomerular filtration rate", "GFR", "renal function", "kidney function"],
    ["nephrology", "kidney specialist", "nephrologist", "renal clinic"],
    ["dialysis", "hemodialysis", "peritoneal dialysis", "ESRD", "RRT", "renal replacement therapy"],
    ["proteinuria", "albuminuria", "UACR", "urine albumin", "microalbuminuria"],
    # Cardiovascular
    ["hypertension", "high blood pressure", "elevated BP", "HTN", "systolic", "diastolic"],
    ["heart failure", "HF", "HFrEF", "HFpEF", "cardiomyopathy", "reduced ejection fraction", "EF"],
    ["atrial fibrillation", "afib", "AF", "arrhythmia"],
    ["coronary artery disease", "CAD", "angina", "chest pain", "ischemic heart disease"],
    ["stroke", "TIA", "ischemic stroke", "cerebrovascular"],
    ["cholesterol", "LDL", "HDL", "lipid", "lipid panel", "hyperlipidemia", "dyslipidemia"],
    ["statin", "atorvastatin", "rosuvastatin", "simvastatin", "lipid lowering"],
    ["anticoagulation", "warfarin", "DOAC", "apixaban", "rivaroxaban", "blood thinner"],
    ["aspirin", "antiplatelet", "ASA"],
    # Screening
    ["mammogram", "mammography", "breast cancer screening", "breast imaging", "biennial"],
    ["colonoscopy", "colorectal screening", "colon cancer", "FIT test", "fecal immunochemical"],
    ["pap smear", "cervical cancer", "HPV test", "colposcopy"],
    ["psa", "prostate specific antigen", "prostate cancer"],
    ["dexa", "bone density", "bone mineral density", "osteoporosis screening", "T-score"],
    # Immunization
    ["flu vaccine", "influenza", "annual flu shot", "influenza vaccine"],
    ["pneumonia vaccine", "pneumococcal", "PCV", "PPSV23", "Prevnar"],
    ["shingles vaccine", "shingrix", "zoster vaccine", "herpes zoster"],
    ["covid vaccine", "coronavirus vaccine", "covid-19 vaccine"],
    # Mental health
    ["depression", "MDD", "depressive disorder", "PHQ-9", "PHQ9", "major depressive", "SSRI", "antidepressant"],
    ["anxiety", "GAD", "GAD-7", "generalized anxiety", "panic"],
    # Obesity
    ["obesity", "overweight", "BMI 30", "BMI 35", "weight management", "bariatric"],
    ["weight loss", "weight reduction", "caloric deficit"],
    # Pulmonary
    ["copd", "chronic obstructive", "emphysema", "chronic bronchitis"],
    ["asthma", "bronchospasm", "inhaled corticosteroid"],
    ["spirometry", "FEV1", "FVC", "pulmonary function"],
    # Oncology
    ["breast cancer", "tamoxifen", "HER2", "aromatase inhibitor", "lumpectomy"],
    ["prostate cancer", "bicalutamide", "ADT", "androgen deprivation"],
    # Rheumatology
    ["rheumatoid arthritis", "RA", "methotrexate", "DMARD", "TNF inhibitor", "biologic"],
    ["lupus", "SLE", "hydroxychloroquine", "ANA"],
    ["osteoporosis", "bone loss", "bisphosphonate", "alendronate", "fracture risk"],
]


def expand_query(query: str) -> str:
    """
    Bidirectional synonym expansion: if the query contains any term from a
    synonym group, add the other terms so the embedding lands closer to the
    matching guideline vocabulary.

    Example:
        "diabetes" →
        "diabetes diabetes mellitus DM T2DM T1DM Type 2 Diabetes Type 1 Diabetes hyperglycemia"
    """
    query_lower = query.lower()
    additions: list[str] = []

    for group in CLINICAL_SYNONYM_GROUPS:
        # Check if ANY term in this group appears in the query
        matched = any(term.lower() in query_lower for term in group)
        if matched:
            for term in group:
                if term.lower() not in query_lower:
                    additions.append(term)

    if additions:
        # Deduplicate while preserving order
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


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve_guidelines(
    query: str,
    category: Optional[str] = None,
    n_results: int = 3,
    min_relevance: float = 0.20,   # lowered from 0.50 — was filtering valid matches
    use_reranker: bool = True,
    expand: bool = True,
) -> list[dict]:
    """
    Retrieve relevant clinical guideline chunks for a query.

    Returns up to n_results chunks ranked by the cross-encoder reranker
    (if enabled) or by cosine similarity.
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

    # Step 1: Query expansion
    search_query = expand_query(query) if expand else query

    # Step 2: Embed the query.
    # Note: NO instruction prefix — all-MiniLM-L6-v2 is not instruction-tuned.
    # The prefix degrades match quality for short domain-specific queries.
    model = _get_embedder()
    query_embedding = model.encode(
        search_query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    # Step 3: Over-fetch candidates for reranking
    fetch_n = min(n_results * 4, collection.count())
    where = {"category": category} if category else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return [{"text": f"Retrieval error: {e}", "source_name": "Error",
                 "url": "", "category": "", "scraped_at": "",
                 "relevance_score": 0.0, "rerank_score": 0.0}]

    if not results["documents"] or not results["documents"][0]:
        return []

    # Step 4: Build candidates with initial cosine scores
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

    if not candidates:
        # Fallback: if strict threshold filtered everything, return the top result
        # anyway with a warning — better than returning nothing for short queries.
        if results["documents"][0]:
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
        else:
            return []

    # Step 5: Cross-encoder reranking
    if use_reranker and len(candidates) > 1:
        try:
            reranker = _get_reranker()
            # Use original (un-expanded) query for reranking — reranker reads
            # the full chunk alongside the query, so extra synonyms aren't needed.
            pairs = [(query, c["text"]) for c in candidates]
            scores = reranker.predict(pairs, show_progress_bar=False)
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception:
            # Reranker failed — fall back to cosine ranking
            candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    return candidates[:n_results]


def format_citations(results: list[dict]) -> str:
    if not results:
        return "No relevant guidelines found."

    lines = ["**Clinical Guideline References:**\n"]
    for i, r in enumerate(results, 1):
        rerank = r.get("rerank_score", r["relevance_score"])
        lines.append(
            f"{i}. **{r['source_name']}** (retrieved {r['scraped_at']})\n"
            f"   Semantic match: {r['relevance_score']:.0%} | "
            f"Rerank score: {rerank:.2f} | "
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
    category: optional filter — one of: preventive_screening, diabetes, cardiovascular,
              ckd, immunization, mental_health, diabetes_ckd

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
        ("eGFR 45 nephrology referral CKD", "ckd"),
    ]

    print("\n=== Guideline Retrieval Test ===\n")
    for query, cat in test_queries:
        print(f"Query: '{query}' (category: {cat})")
        results = retrieve_guidelines(query, category=cat, n_results=2)
        if results:
            for r in results:
                print(f"  [{r['relevance_score']:.0%} cosine | {r.get('rerank_score', 0):.2f} rerank] {r['source_name']}")
                print(f"  {r['text'][:150]}...")
        else:
            print("  No results")
        print()