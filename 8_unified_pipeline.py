"""
End-to-end unified training pipeline:
1. Calibration sweeps.
2. Ensemble training with unified loss.
3. Report generation.
"""

from src.calibration_runs import run_all_calibrations
from src.unified_ensemble import train_ensemble
from src.unified_report import generate_report


def main():
    print("=== Running calibration sweeps ===")
    run_all_calibrations()

    print("\n=== Training unified ensemble ===")
    train_ensemble()

    print("\n=== Generating unified report ===")
    generate_report()

    print("\nUnified pipeline complete.")


if __name__ == "__main__":
    main()
