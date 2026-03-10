"""
Generate the corrected strict LODO figure for the manuscript.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "supplementary_experiments"
OUTPUT = RESULTS / "figures" / "experiment_2_cross_validation_corrected.png"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    results = load_json(RESULTS / "experiment_2_results.json")
    folds = results["folds"]

    fold_labels = [f"Fold {f['fold']}\n({f['held_out_dose']:g} mg/dl)" for f in folds]
    train_r2 = [float(f["train_r2"]) for f in folds]
    test_err = [float(f["test_error"]) for f in folds]
    passed = [bool(f["passed_plausibility"]) for f in folds]

    mean_train = float(np.mean(train_r2))
    mean_test = float(np.mean(test_err))
    std_test = float(np.std(test_err, ddof=1))

    pass_colors = ["#67c18d" if p else "#e67e73" for p in passed]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.8))

    # (a) Training performance
    bars = ax1.bar(fold_labels, train_r2, color=pass_colors, edgecolor="black", linewidth=1.8)
    ax1.axhline(mean_train, color="#4f6bed", linestyle="--", linewidth=2, label=f"Mean: {mean_train:.3f}")
    ax1.set_ylabel("Training R²", fontweight="bold", fontsize=12)
    ax1.set_title("a) Training Performance", fontweight="bold", fontsize=14)
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)
    for bar, val in zip(bars, train_r2):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.03, f"{val:.3f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=10)

    # (b) Held-out error
    bars = ax2.bar(fold_labels, test_err, color=pass_colors, edgecolor="black", linewidth=1.8)
    ax2.axhline(mean_test, color="#4f6bed", linestyle="--", linewidth=2,
                label=f"Mean: {mean_test:.3f} ± {std_test:.3f}")
    ax2.set_ylabel("Held-out Error (normalized units)", fontweight="bold", fontsize=12)
    ax2.set_title("b) Held-out Dose Prediction Error", fontweight="bold", fontsize=14)
    ax2.set_ylim(0, max(test_err) * 1.18)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=10)
    for bar, val in zip(bars, test_err):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{val:.3f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=10)

    # (c) Plausibility
    y = np.arange(len(folds))
    status_vals = [1] * len(folds)
    ax3.barh(y, status_vals, color=pass_colors, edgecolor="black", linewidth=1.8)
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"Fold {f['fold']}\n({f['held_out_dose']:g} mg/dl)" for f in folds])
    ax3.set_xticks([])
    ax3.set_xlim(0, 1.25)
    ax3.set_title("c) Biological Plausibility", fontweight="bold", fontsize=14)
    for yi, p in zip(y, passed):
        ax3.text(0.62, yi, "[OK] Pass" if p else "[X] Fail", ha="center", va="center",
                 fontsize=12, fontweight="bold", color="white")
    ax3.invert_yaxis()
    pass_rate = 100.0 * sum(1 for p in passed if p) / len(passed)
    ax3.text(0.5, -0.12, f"Overall: {pass_rate:.0f}% pass rate ({sum(passed)}/{len(passed)} folds)",
             transform=ax3.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")

    fig.suptitle("Experiment 2: Leave-One-Dose-Out Cross-Validation", fontsize=20, fontweight="bold", y=1.03)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
