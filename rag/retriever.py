"""
rag/retriever.py

Retrieval interface that agents use to query clinical guidelines.

Accuracy improvements over v1:
  1. Clinical query expansion  — adds synonyms before embedding
     ("HbA1c" → also searches "hemoglobin A1c", "glycated hemoglobin")
  2. Over-fetch + cross-encoder reranking
     — fetches 12 chunks, reranks with a cross-encoder, returns top 3
     — cross-encoder reads query+chunk together (much more accurate than cosine alone)
  3. min_relevance raised from 0.3 → 0.5 (stops garbage from appearing)
  4. Clinical query prefix for the embedding model
     — prepends "Represent this medical question for retrieval:" to improve
        similarity scores for clinical vocabulary
  5. All library logs suppressed at import time
"""
import logging
import os
import warnings
from typing import Optional

# ── Suppress all library noise before any imports ─────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_guidelines"
COLLECTION_NAME = "clinical_guidelines"

# Primary embedding model — same as embedder.py (must match!)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder reranker — reads query+document together, much more accurate
# than cosine similarity alone. Downloads ~85MB once, then cached.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_client = None
_collection = None
_embedder = None
_reranker = None

# ── Clinical synonym expansion map ────────────────────────────────────────────
# Adds known medical synonyms to the query before embedding.
# This bridges vocabulary gaps between patient notes and guideline text.
CLINICAL_SYNONYMS: dict[str, list[str]] = {
    # Diabetes
    "hba1c":            ["hemoglobin A1c", "glycated hemoglobin", "A1c", "HbA1c", "glycemic control"],
    "diabetes":         ["diabetes mellitus", "DM", "T2DM", "T1DM", "Type 2 Diabetes", "hyperglycemia"],
    "metformin":        ["biguanide", "first-line diabetes therapy", "glucophage"],
    "insulin pump":     ["CSII", "continuous subcutaneous insulin infusion", "insulin delivery device"],
    "cgm":              ["continuous glucose monitor", "glucose sensor", "dexcom", "libre"],
    "glp-1":            ["semaglutide", "liraglutide", "ozempic", "wegovy", "GLP-1 agonist"],
    "sglt2":            ["empagliflozin", "dapagliflozin", "jardiance", "SGLT2 inhibitor"],
    # Cardiovascular
    "egfr":             ["estimated glomerular filtration rate", "kidney function", "GFR", "renal function"],
    "ckd":              ["chronic kidney disease", "renal insufficiency", "nephropathy", "kidney disease"],
    "ldl":              ["LDL cholesterol", "low-density lipoprotein", "bad cholesterol", "lipid"],
    "hdl":              ["HDL cholesterol", "high-density lipoprotein", "good cholesterol"],
    "hypertension":     ["high blood pressure", "elevated BP", "HTN", "systolic", "diastolic"],
    "statin":           ["atorvastatin", "rosuvastatin", "simvastatin", "lipid lowering", "cholesterol"],
    "heart failure":    ["HF", "HFrEF", "HFpEF", "cardiomyopathy", "reduced ejection fraction"],
    "atrial fibrillation": ["AFib", "AF", "arrhythmia", "anticoagulation", "warfarin"],
    "coronary artery":  ["CAD", "angina", "chest pain", "ischemic heart disease", "cardiac"],
    # Kidney
    "nephrology":       ["kidney specialist", "renal", "nephrology referral", "nephrologist"],
    "dialysis":         ["hemodialysis", "peritoneal dialysis", "ESRD", "kidney failure"],
    "proteinuria":      ["UACR", "albuminuria", "protein in urine", "microalbuminuria"],
    # Screening
    "mammogram":        ["mammography", "breast cancer screening", "breast imaging", "biennial"],
    "colonoscopy":      ["colorectal screening", "colon cancer screening", "FIT test", "colonoscopy"],
    "pap smear":        ["cervical cancer screening", "HPV test", "colposcopy", "cervical"],
    "psa":              ["prostate specific antigen", "prostate cancer screening"],
    "dexa":             ["bone density", "bone mineral density", "osteoporosis screening"],
    # Immunization
    "flu vaccine":      ["influenza vaccine", "influenza immunization", "annual flu shot", "trivalent"],
    "pneumonia vaccine":["pneumococcal", "PCV", "PPSV23", "Prevnar", "pneumonia immunization"],
    "shingles vaccine": ["shingrix", "zoster vaccine", "herpes zoster", "varicella"],
    # Mental health
    "depression":       ["PHQ-9", "depressive disorder", "MDD", "antidepressant", "SSRI"],
    "anxiety":          ["GAD-7", "generalized anxiety", "panic", "anxious", "SSRI"],
    # Obesity
    "bmi":              ["body mass index", "obesity", "overweight", "weight", "adiposity"],
    "obesity":          ["BMI 30", "BMI 35", "overweight", "weight management", "bariatric"],
    # Pulmonary
    "copd":             ["chronic obstructive pulmonary disease", "FEV1", "spirometry", "emphysema"],
    "asthma":           ["bronchospasm", "inhaled corticosteroid", "albuterol", "peak flow"],
    # Oncology
    "breast cancer":    ["tamoxifen", "HER2", "hormone receptor", "mastectomy", "lumpectomy"],
    "prostate cancer":  ["bicalutamide", "ADT", "PSA", "radical prostatectomy", "gleason"],
    # Rheumatology
    "rheumatoid arthritis": ["RA", "methotrexate", "DMARD", "TNF inhibitor", "biologic", "DAS28"],
    "lupus":            ["SLE", "hydroxychloroquine", "ANA", "anti-dsDNA", "lupus nephritis"],
    "osteoporosis":     ["bone density", "T-score", "bisphosphonate", "alendronate", "FRAX"],
    # Neurology
    "stroke":           ["TIA", "ischemic stroke", "anticoagulation", "post-stroke", "clopidogrel"],
    "parkinson":        ["levodopa", "carbidopa", "dopamine", "tremor", "bradykinesia"],
}


def expand_query(query: str) -> str:
    """
    Add clinical synonyms to the query string so the embedding model
    can match guideline text that uses different terminology.

    Example:
        "eGFR 45 CKD referral" →
        "eGFR 45 CKD referral estimated glomerular filtration rate kidney function
         chronic kidney disease renal insufficiency nephropathy"
    """
    query_lower = query.lower()
    additions = []

    for term, synonyms in CLINICAL_SYNONYMS.items():
        if term in query_lower:
            additions.extend(synonyms)

    if additions:
        # Deduplicate and append
        seen = set(query_lower.split())
        new_terms = [t for t in additions if t.lower() not in seen]
        return query + " " + " ".join(new_terms)

    return query


# ── Lazy loaders ──────────────────────────────────────────────────────────────

def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve_guidelines(
    query: str,
    category: Optional[str] = None,
    n_results: int = 3,
    min_relevance: float = 0.50,    # raised from 0.3 — stops garbage chunks
    use_reranker: bool = True,       # cross-encoder reranking
    expand: bool = True,             # clinical synonym expansion
) -> list[dict]:
    """
    Retrieve relevant clinical guideline chunks for a query.

    Pipeline:
      1. Expand query with clinical synonyms
      2. Embed expanded query
      3. Over-fetch (n_results * 4) from ChromaDB
      4. Cross-encoder rerank to get true top n_results
      5. Filter by min_relevance

    Args:
        query:          Natural language clinical query
        category:       Optional ChromaDB filter by category
        n_results:      Final number of chunks to return after reranking
        min_relevance:  Min cosine similarity for initial candidates (0-1)
        use_reranker:   Whether to apply cross-encoder reranking
        expand:         Whether to expand query with clinical synonyms

    Returns:
        List of dicts: text, source_name, url, category, scraped_at,
                       relevance_score, rerank_score
    """
    collection = _get_collection()

    if collection.count() == 0:
        return [{
            "text": "No guidelines loaded yet. Run: python rag/refresh_flow.py",
            "source_name": "System",
            "url": "",
            "category": "",
            "scraped_at": "",
            "relevance_score": 0.0,
            "rerank_score": 0.0
        }]

    # Step 1: Query expansion
    search_query = expand_query(query) if expand else query

    # Step 2: Embed with a clinical prefix
    # Adding this prefix significantly improves similarity scores for medical vocab
    prefixed_query = f"Represent this medical question for retrieval: {search_query}"
    model = _get_embedder()
    query_embedding = model.encode(prefixed_query, normalize_embeddings=True).tolist()

    # Step 3: Over-fetch candidates (4x more than we need, reranker picks the best)
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
                "rerank_score": cosine_score  # default if reranker skipped
            })

    if not candidates:
        return []

    # Step 5: Cross-encoder reranking
    # The cross-encoder reads query + document together (not separate embeddings)
    # This is much more accurate for clinical nuance like "HbA1c > 9%" vs "> 8.5%"
    if use_reranker and len(candidates) > 1:
        try:
            reranker = _get_reranker()
            pairs = [(query, c["text"]) for c in candidates]  # use original query
            scores = reranker.predict(pairs)
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)
            # Sort by reranker score (descending)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            # Reranker failed — fall back to cosine ranking
            candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    return candidates[:n_results]


def format_citations(results: list[dict]) -> str:
    """
    Format retrieved guideline chunks into a citation block agents embed in output.
    """
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


# ── LangChain tool ────────────────────────────────────────────────────────────

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


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        ("mammogram screening 67 year old woman", "preventive_screening"),
        ("HbA1c target type 2 diabetes", "diabetes"),
        ("eGFR 45 nephrology referral CKD", "ckd"),
        ("flu vaccine annual adult immunization", "immunization"),
    ]

    print("\n=== Guideline Retrieval Test (with reranking) ===\n")
    for query, cat in test_queries:
        print(f"Query: '{query}' (category: {cat})")
        results = retrieve_guidelines(query, category=cat, n_results=2)
        if results:
            for r in results:
                print(f"  [{r['relevance_score']:.0%} cosine | {r.get('rerank_score', 0):.2f} rerank] {r['source_name']}")
                print(f"  {r['text'][:150]}...")
        else:
            print("  No results (collection may be empty — run refresh_flow.py first)")
        print()