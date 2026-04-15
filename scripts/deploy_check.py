"""
scripts/deploy_check.py

Pre-deployment verification script.
Run this before pushing to production to catch common issues.

Usage:
    python scripts/deploy_check.py              # check local setup
    python scripts/deploy_check.py --api-url https://your-app.railway.app
"""
import os
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "

results = []

def check(label: str, passed: bool, detail: str = "", warn_only: bool = False):
    icon = PASS if passed else (WARN if warn_only else FAIL)
    results.append((icon, label, detail, passed or warn_only))
    print(f"  {icon}  {label}" + (f"\n       {detail}" if detail else ""))


print("\n══════════════════════════════════════════════")
print("  Healthcare AI — Deployment Check")
print("══════════════════════════════════════════════\n")

# ── 1. Environment variables ──────────────────────────────────────────────────
print("【1】 Environment Variables")

from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY", "")
check("OPENAI_API_KEY set",
      bool(openai_key and openai_key.startswith("sk-")),
      "Add OPENAI_API_KEY=sk-... to your .env file")

langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
check("LANGCHAIN_API_KEY set (LangSmith)",
      bool(langchain_key),
      "Optional but recommended — add LANGCHAIN_API_KEY for observability",
      warn_only=True)

check("MODEL_NAME set",
      bool(os.getenv("MODEL_NAME")),
      "Defaults to gpt-4o-mini — fine for production",
      warn_only=True)

check("DATABASE_URL set",
      True,  # has a default
      f"Using: {os.getenv('DATABASE_URL', 'sqlite:///./healthcare_agents.db')}")

# ── 2. Python imports ─────────────────────────────────────────────────────────
print("\n【2】 Python Dependencies")

required_packages = [
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("crewai", "crewai"),
    ("fastapi", "fastapi"),
    ("streamlit", "streamlit"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("networkx", "networkx"),
    ("httpx", "httpx"),
    ("prefect", "prefect"),
    ("sqlalchemy", "sqlalchemy"),
]

for module, pip_name in required_packages:
    try:
        __import__(module)
        check(f"{pip_name} installed", True)
    except ImportError:
        check(f"{pip_name} installed", False, f"pip install {pip_name}")

# ── 3. Data files ─────────────────────────────────────────────────────────────
print("\n【3】 Data Files")

data_files = [
    ("data/synthetic_patients.json", "20 synthetic patients"),
    ("data/payer_policies.json",     "Payer policy data"),
]

for filepath, label in data_files:
    path = Path(filepath)
    check(f"{label} ({filepath})", path.exists(),
          f"File missing — check git commit")

# Check patient count
try:
    with open("data/synthetic_patients.json") as f:
        patients = json.load(f)
    check("20 patients loaded", len(patients) == 20,
          f"Found {len(patients)} patients")
except Exception as e:
    check("Patient data readable", False, str(e))

# ── 4. Database ───────────────────────────────────────────────────────────────
print("\n【4】 Database")

try:
    from api.main import init_db, engine
    from sqlalchemy import text
    init_db()
    with engine.connect() as conn:
        audit_count = conn.execute(text("SELECT COUNT(*) FROM audit_log")).scalar()
        review_count = conn.execute(text("SELECT COUNT(*) FROM review_queue")).scalar()
    check("Database tables exist", True)
    check("Demo data seeded",
          audit_count >= 5,
          f"audit_log: {audit_count} rows, review_queue: {review_count} rows — run: python scripts/reset_demo_data.py",
          warn_only=(audit_count == 0))
except Exception as e:
    check("Database accessible", False, str(e))

# ── 5. Knowledge graph ────────────────────────────────────────────────────────
print("\n【5】 Knowledge Graph")

try:
    from knowledge_graph.clinical_graph import get_graph_stats
    stats = get_graph_stats()
    check("Knowledge graph loads",
          stats["total_nodes"] >= 90,
          f"{stats['total_nodes']} nodes, {stats['total_edges']} edges")
except Exception as e:
    check("Knowledge graph loads", False, str(e))

# ── 6. RAG / ChromaDB ─────────────────────────────────────────────────────────
print("\n【6】 RAG Knowledge Base")

try:
    from rag.embedder import get_collection_stats
    rag_stats = get_collection_stats()
    chunks = rag_stats.get("total_chunks", 0)
    check("ChromaDB accessible", True)
    check("Guidelines loaded",
          chunks > 100,
          f"{chunks} chunks across {rag_stats.get('total_sources', 0)} sources — run: python rag/refresh_flow.py",
          warn_only=(chunks == 0))
except Exception as e:
    check("ChromaDB accessible", False, str(e))

# ── 7. FastAPI health ─────────────────────────────────────────────────────────
print("\n【7】 FastAPI Endpoints (local)")

try:
    import httpx
    with httpx.Client(timeout=5) as client:
        r = client.get("http://localhost:8000/health")
        r.raise_for_status()
        data = r.json()
    check("API health check", data.get("status") == "healthy",
          f"Response: {data}")

    r2 = client.get("http://localhost:8000/audit-log?limit=1")
    check("Audit log endpoint", r2.status_code == 200)

    r3 = client.get("http://localhost:8000/pending-reviews")
    check("Review queue endpoint", r3.status_code == 200)

    r4 = client.get("http://localhost:8000/guidelines-status")
    check("Guidelines status endpoint", r4.status_code == 200)

except httpx.ConnectError:
    check("API running locally", False,
          "Start the API first: uvicorn api.main:app --reload --port 8000",
          warn_only=True)
except Exception as e:
    check("API health check", False, str(e))

# ── 8. Remote API check (if URL provided) ────────────────────────────────────
remote_url = None
for arg in sys.argv[1:]:
    if arg.startswith("--api-url="):
        remote_url = arg.replace("--api-url=", "")
    elif arg == "--api-url" and len(sys.argv) > sys.argv.index(arg) + 1:
        remote_url = sys.argv[sys.argv.index(arg) + 1]

if remote_url:
    print(f"\n【8】 Remote API Check ({remote_url})")
    try:
        import httpx
        with httpx.Client(timeout=15) as client:
            r = client.get(f"{remote_url}/health")
            r.raise_for_status()
            data = r.json()
        check("Remote API health", data.get("status") == "healthy",
              f"Version: {data.get('version')}, URL: {remote_url}")

        r2 = client.get(f"{remote_url}/audit-log?limit=1")
        check("Remote audit log",  r2.status_code == 200)

        r3 = client.get(f"{remote_url}/pending-reviews")
        check("Remote review queue", r3.status_code == 200)

        r4 = client.get(f"{remote_url}/guidelines-status")
        g = r4.json()
        chunks = g.get("collection", {}).get("total_chunks", 0)
        check("Remote guidelines", chunks > 0,
              f"{chunks} chunks loaded",
              warn_only=(chunks == 0))

    except Exception as e:
        check("Remote API reachable", False, str(e))

# ── 9. Docker (if present) ────────────────────────────────────────────────────
print("\n【9】 Docker (for cloud deployment)")

try:
    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    check("Docker installed", result.returncode == 0,
          result.stdout.strip())
    check("Dockerfile present", Path("Dockerfile").exists())
    check("railway.toml present", Path("railway.toml").exists())
    check("render.yaml present",  Path("render.yaml").exists())
except FileNotFoundError:
    check("Docker installed", False,
          "Not required for Railway/Render deployment",
          warn_only=True)

# ── 10. Streamlit config ──────────────────────────────────────────────────────
print("\n【10】 Streamlit Cloud Config")

check(".streamlit/config.toml present",
      Path(".streamlit/config.toml").exists())
check(".streamlit/secrets.toml.example present",
      Path(".streamlit/secrets.toml.example").exists())
check("frontend/app.py reads API_BASE from secrets",
      "st.secrets" in Path("frontend/app.py").read_text())

# ── 11. Git hygiene ───────────────────────────────────────────────────────────
print("\n【11】 Git / Repository")

try:
    result = subprocess.run(["git", "status", "--porcelain"],
                            capture_output=True, text=True)
    uncommitted = result.stdout.strip()
    check("No uncommitted changes",
          not uncommitted,
          f"Uncommitted: {uncommitted[:200]}" if uncommitted else "",
          warn_only=bool(uncommitted))

    # Check .env is not committed
    result2 = subprocess.run(["git", "ls-files", ".env"],
                              capture_output=True, text=True)
    env_committed = result2.stdout.strip()
    check(".env NOT in git", not env_committed,
          "DANGER: .env is tracked by git — run: git rm --cached .env")

    # Check gitignore has key entries
    gitignore = Path(".gitignore").read_text() if Path(".gitignore").exists() else ""
    check(".env in .gitignore", ".env" in gitignore)
    check("chroma_guidelines in .gitignore",
          "chroma" in gitignore,
          "Large vector DB should not be committed",
          warn_only=("chroma" not in gitignore))

except Exception as e:
    check("Git accessible", False, str(e), warn_only=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
total   = len(results)
passed  = sum(1 for _, _, _, ok in results if ok)
failed  = sum(1 for icon, _, _, ok in results if not ok)
warnings = sum(1 for icon, _, _, ok in results if icon == WARN and ok)

print(f"  Total checks : {total}")
print(f"  {PASS} Passed    : {passed - warnings}")
print(f"  {WARN} Warnings  : {warnings}")
print(f"  {FAIL} Failures  : {failed}")

if failed == 0:
    print("\n  ✅ All checks passed — ready to deploy!")
    print("\n  Next steps:")
    print("  1. git add . && git commit -m 'feat: production deployment config'")
    print("  2. git push origin main")
    print("  3. Deploy API: railway up  (or connect repo to Render)")
    print("  4. Deploy dashboard: connect repo to share.streamlit.io")
    print("  5. Set API_BASE_URL secret in Streamlit Cloud settings")
    print("  6. Run: python scripts/deploy_check.py --api-url https://your-app.railway.app")
else:
    print(f"\n  ❌ Fix {failed} failure(s) before deploying.")

print("══════════════════════════════════════════════\n")
sys.exit(0 if failed == 0 else 1)