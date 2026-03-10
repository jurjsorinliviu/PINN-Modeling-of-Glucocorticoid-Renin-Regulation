"""
Reproduce the deterministic manuscript pipeline and reviewer-facing analyses.

This wrapper runs the main training, comparison, supplementary, and reviewer
experiment scripts in the order expected by the final codebase.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PIPELINE = [
    "1_setup_and_data_check.py",
    "2_train_ode_baseline.py",
    "8_unified_pipeline.py",
    "9_ensemble_synthetic_03.py",
    "10_compare_ensembles.py",
    "11_supplementary_experiments.py",
    "12_pure_nn_baseline.py",
    "13_reviewer_requested_experiments.py",
    "wilcoxon_test.py",
    "5_comprehensive_ieee_analysis.py",
    "6_generate_missing_reports.py",
]


def run_script(script_name: str) -> None:
    script_path = Path(script_name)
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_name}")

    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print("=" * 80)

    start = time.time()
    subprocess.run([sys.executable, script_name], check=True)
    elapsed = time.time() - start

    print(f"[OK] {script_name} completed in {elapsed / 60:.1f} minutes")


def main() -> None:
    print("=" * 80)
    print("FULL REPRODUCTION PIPELINE")
    print("=" * 80)
    print("This will regenerate the deterministic manuscript results and the")
    print("reviewer-requested comparison package.")

    total_start = time.time()
    for script_name in PIPELINE:
        run_script(script_name)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("REPRODUCTION COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {total_elapsed / 3600:.2f} hours")


if __name__ == "__main__":
    main()
