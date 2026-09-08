"""Reproduce the entire analysis end-to-end from the raw NHTSA flat files.

    python analysis/run_all.py

Every deliverable under results/ and Data/processed/ is regenerated. qa.md is reset
first so the gate log always reflects exactly one clean run.
"""

from __future__ import annotations

import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import AUDIT_DIR   # noqa: E402

STEPS = [
    ("01_build_complaints.py", "Step 1  -  complaint-level build"),
    ("02_delay_analysis.py", "Step 2  -  delay distributions and hazard model"),
    ("03_triangle_nowcast.py", "Step 3  -  triangles, chain ladder, unit tests"),
    ("04_backtest.py", "Step 4  -  rolling-origin backtest"),
    ("05_case_studies.py", "Step 5  -  early-warning case studies"),
    ("06_figures.py", "Step 6  -  figures 1-4"),
    ("07_tables.py", "Step 7  -  LaTeX tables and compile proof"),
    ("08_robustness.py", "Step 8  -  robustness battery"),
    ("09_summary.py", "Step 9  -  RESULTS_SUMMARY.md"),
    ("11_tables34.py", "Step 11 - tables 3 and 4 from the robustness battery"),
    ("10_qa_audit.py", "Self-QA  -  formula-vs-code and style audit"),
]


def main():
    qa = os.path.join(AUDIT_DIR, "qa.md")
    if os.path.exists(qa):
        os.remove(qa)

    t0 = time.time()
    for script, title in STEPS:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
        t = time.time()
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
        print(f"  [{time.time() - t:.1f}s]", flush=True)
    print(f"\n{'=' * 72}\nAll steps complete in {time.time() - t0:.0f}s.")
    print(f"Gate log: {qa}")


if __name__ == "__main__":
    main()
