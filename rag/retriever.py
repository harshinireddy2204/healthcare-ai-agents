"""
rag/retriever.py

Retrieval interface that agents use to query clinical guidelines.
Returns relevant chunks with source citations.

Agents call retrieve_guidelines() as an MCP-style tool.
Every result includes the guideline name, URL, and date so
the agent can cite it explicitly in its output.
"""
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Reuse the same paths as embedder
from pathlib import Path
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_guidelines"
COLLECTION_NAME = "clinical_guidelines"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_client = None
_collection = None
_embedder = None


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


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def retrieve_guidelines(
    query: str,
    category: Optional[str] = None,
    n_results: int = 4,
    min_relevance: float = 0.3
) -> list[dict]:
    """
    Retrieve relevant clinical guideline chunks for a query.

    Args:
        query:          Natural language query (e.g. "mammogram screening age 67 female")
        category:       Optional filter by category (e.g. "preventive_screening", "diabetes")
        n_results:      How many chunks to retrieve
        min_relevance:  Minimum cosine similarity score (0-1)

    Returns:
        List of dicts with: text, source_name, url, category, scraped_at, relevance_score
    """
    collection = _get_collection()
    model = _get_embedder()

    if collection.count() == 0:
        return [{
            "text": "No guidelines loaded yet. Run the refresh flow first.",
            "source_name": "System",
            "url": "",
            "category": "",
            "scraped_at": "",
            "relevance_score": 0.0
        }]

    # Embed the query
    query_embedding = model.encode(query).tolist()

    # Build filter
    where = {"category": category} if category else None

    # Query ChromaDB
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return [{"text": f"Retrieval error: {e}", "source_name": "Error",
                 "url": "", "category": "", "scraped_at": "", "relevance_score": 0.0}]

    if not results["documents"] or not results["documents"][0]:
        return []

    # Convert cosine distance to similarity score (ChromaDB returns distance)
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        relevance = 1 - dist  # cosine distance → similarity
        if relevance >= min_relevance:
            output.append({
                "text": doc,
                "source_name": meta.get("source_name", "Unknown"),
                "url": meta.get("url", ""),
                "category": meta.get("category", ""),
                "scraped_at": meta.get("scraped_at", "")[:10],  # date only
                "relevance_score": round(relevance, 3)
            })

    return output


def format_citations(results: list[dict]) -> str:
    """
    Format retrieved guideline chunks into a citation block
    that agents can include in their output.
    """
    if not results:
        return "No relevant guidelines found."

    lines = ["**Clinical Guideline References:**\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. **{r['source_name']}** (retrieved {r['scraped_at']})\n"
            f"   Relevance: {r['relevance_score']:.0%} | "
            f"[Source]({r['url']})\n"
            f"   > {r['text'][:300]}...\n"
        )
    return "\n".join(lines)


# ── LangChain tool wrapper ────────────────────────────────────────────────────

@tool
def search_clinical_guidelines(query: str, category: str = "") -> str:
    """
    Search the clinical guidelines knowledge base for evidence-based recommendations.
    Use this tool whenever you need to cite a specific guideline for a care gap,
    prior authorization criteria, or risk management decision.

    query: natural language search query
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


# Export as tool list for agent binding
GUIDELINE_TOOLS = [search_clinical_guidelines]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        ("mammogram screening 67 year old woman", "preventive_screening"),
        ("HbA1c target type 2 diabetes", "diabetes"),
        ("eGFR 45 nephrology referral CKD", "ckd"),
        ("flu vaccine annual adult immunization", "immunization"),
    ]

    print("\n=== Guideline Retrieval Test ===\n")
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
