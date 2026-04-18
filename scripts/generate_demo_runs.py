"""
scripts/generate_demo_runs.py

Generate pre-computed workflow results for the public portfolio demo.

Problem this solves:
  The public Railway deployment has a tiny API budget (~$2-5 in OpenAI credits).
  Each live workflow costs $0.03-0.05 and takes 2-3 minutes.
  If 10 recruiters load the app simultaneously, 9 get HTTP 409 and the budget
  evaporates in an hour.

Solution:
  Run each workflow ONCE locally (where you have credits and patience), capture
  the full output, and check the JSON into git. The public app serves these
  cached results instantly to anyone visiting the demo — zero API calls,
  zero latency, unlimited concurrent viewers.

How to use:
  # Local machine, with OPENAI_API_KEY set:
  cd healthcare-ai-agents
  python scripts/generate_demo_runs.py

  # Commits:
  git add data/demo_runs.json
  git commit -m "Regenerate demo runs"
  git push

  # On Railway: no config needed — main.py loads data/demo_runs.json at startup.

The generated JSON is indexed by patient_id and contains the complete workflow
output including care_gap_report, knowledge_graph findings, prior_auth decisions,
and ICT clinical synthesis for HIGH-complexity cases.

Patients to run:
  P001 — LOW complexity  (simple preventive care gaps)
  P004 — HIGH complexity (CKD4 + anticoagulation, escalates to HITL review)
  P007 — LOW complexity  (standard auth approval)
  P014 — HIGH complexity (oncology + multi-comorbidity, escalates)
  P018 — MODERATE        (multi-factor case, MDT collaboration)
  P020 — MODERATE        (diabetes + cardiovascular, standard MDT)

These 6 patients showcase all three pathways (LOW/MODERATE/HIGH), both outcomes
(COMPLETED and PENDING_REVIEW), and all three agents (care gap, prior auth, ICT).
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root so OPENAI_API_KEY (and other env vars) are
# available. Without this, the script errors out even when .env is populated,
# because os.getenv only reads already-exported environment variables.
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Patients we want to pre-compute. These cover all complexity tiers and both
# completion paths, giving visitors a representative tour of the system.
DEMO_PATIENTS = [
    ("P001", "full"),  # LOW complexity, completes cleanly
    ("P007", "full"),  # LOW complexity, auth approval
    ("P018", "full"),  # MODERATE, MDT collaboration, escalates
    ("P020", "full"),  # MODERATE, diabetes+CV, completes
    ("P004", "full"),  # HIGH complexity, ICT synthesis, escalates
    ("P014", "full"),  # HIGH complexity, oncology, escalates
]

OUTPUT_PATH = PROJECT_ROOT / "data" / "demo_runs.json"


def run_one_workflow(patient_id: str, mode: str) -> dict:
    """
    Run a single workflow and capture its full output.
    Uses the same supervisor graph as the live API (agents.triage_supervisor).

    The returned dict must match the shape the frontend renders:
      - crew_output: string, parsed into KG/CARE GAPS/PRIOR AUTH/SYNTHESIS sections
      - complexity_tier, complexity_score, complexity_rationale
      - status, workflow_results
    That matches what `api/main.py`'s /demo-runs/{patient_id} endpoint serves
    and what render_crew_output() in frontend/app.py parses.
    """
    print(f"\n{'='*60}")
    print(f"  Running {patient_id} (mode={mode})")
    print(f"{'='*60}")

    # Import here so --dry-run can work without full agent stack
    from agents.triage_supervisor import run_triage

    start = time.time()
    try:
        # Live agents only support the full pathway when pre-computing cache.
        # If you want auth_only or care_gap_only cached, extend this block.
        if mode != "full":
            print(f"  (note: mode={mode} requested — running full triage anyway "
                  f"so the cache renders in all dashboard sections)")

        result = run_triage(patient_id)
        elapsed = round(time.time() - start, 1)
        print(f"  ✅ Completed in {elapsed}s — {result.get('status', 'COMPLETED')}")

        # The result from run_triage already has everything we need.
        # We just add a demo marker and timestamp.
        result["_is_demo"] = True
        result["_generated_at"] = datetime.utcnow().isoformat() + "Z"
        result["_elapsed_seconds"] = elapsed
        result["mode"] = "full"
        return result

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"  ❌ Failed after {elapsed}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            "patient_id": patient_id,
            "mode": mode,
            "status": "ERROR",
            "error": str(e),
            "_elapsed_seconds": elapsed,
            "_generated_at": datetime.utcnow().isoformat() + "Z",
            "_is_demo": True,
        }


def main():
    # Validate env before burning budget
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to your local .env first.")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing runs so we can re-run individual patients without redoing all of them
    existing = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            print(f"[Init] Found existing demo_runs.json with {len(existing)} patients")
        except Exception as e:
            print(f"[Init] Could not parse existing file: {e}. Starting fresh.")

    # Allow re-running only specific patients: python generate_demo_runs.py P004 P018
    target_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    patients_to_run = (
        [p for p in DEMO_PATIENTS if p[0] in target_ids]
        if target_ids else DEMO_PATIENTS
    )

    if not patients_to_run:
        print(f"ERROR: No matching patient IDs in {target_ids}. Valid: {[p[0] for p in DEMO_PATIENTS]}")
        sys.exit(1)

    print(f"[Plan] Running {len(patients_to_run)} workflow(s): {[p[0] for p in patients_to_run]}")
    print(f"[Plan] Estimated cost: ${len(patients_to_run) * 0.05:.2f}")
    print(f"[Plan] Estimated time: {len(patients_to_run) * 3} minutes\n")

    # Give a chance to abort
    try:
        input("Press Enter to start, or Ctrl+C to abort...")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    # Run each workflow, saving after each one so a crash doesn't lose progress
    for patient_id, mode in patients_to_run:
        result = run_one_workflow(patient_id, mode)
        existing[patient_id] = result

        # Save incrementally
        OUTPUT_PATH.write_text(json.dumps(existing, indent=2, default=str))
        print(f"  💾 Saved progress to {OUTPUT_PATH}")

        # Pause between runs to avoid hitting TPM ceiling on subsequent calls
        if patient_id != patients_to_run[-1][0]:
            print("  ⏳ Waiting 120s for TPM bucket to fully refill...")
            time.sleep(120)

    # Final summary
    successful = sum(1 for v in existing.values() if v.get("status") != "ERROR")
    print(f"\n{'='*60}")
    print(f"  ✅ Done. {successful}/{len(existing)} runs captured.")
    print(f"  📁 {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  git add data/demo_runs.json")
    print("  git commit -m 'Regenerate demo runs'")
    print("  git push")


if __name__ == "__main__":
    main()