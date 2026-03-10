"""
Generate manuscript-style versions of supplementary figures 6-8.
"""

from pathlib import Path
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "supplementary_experiments"
FIGURES = RESULTS / "figures"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_figure6():
    results = load_json(RESULTS / "experiment_1_results.json")
    constant = results["constant"]
    ramped = results["ramped"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    configs = ["Constant", "Ramped"]
    success = [constant["success_rate"], ramped["success_rate"]]
    colors = ["#e67e73", "#67c18d"]
    bars = ax1.bar(configs, success, color=colors, edgecolor="black", linewidth=1.8)
    ax1.set_ylabel("Success Rate (%)", fontweight="bold", fontsize=12)
    ax1.set_title("a) Plausibility Success Rate", fontweight="bold", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, success):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    mean_r2 = [constant["mean_r2"], ramped["mean_r2"]]
    bars = ax2.bar(configs, mean_r2, color=colors, edgecolor="black", linewidth=1.8)
    ax2.set_ylabel("Mean R² (Passed Models)", fontweight="bold", fontsize=12)
    ax2.set_title("b) Model Accuracy", fontweight="bold", fontsize=14)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, mean_r2):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Constant vs. Ramped High-Dose Weighting", fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES / "experiment_1_ramp_ablation_corrected.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


def plot_figure7():
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

    bars = ax1.bar(fold_labels, train_r2, color=pass_colors, edgecolor="black", linewidth=1.8)
    ax1.axhline(mean_train, color="#4f6bed", linestyle="--", linewidth=2, label=f"Mean: {mean_train:.3f}")
    ax1.set_ylabel("Training R²", fontweight="bold", fontsize=12)
    ax1.set_title("a) Training Performance", fontweight="bold", fontsize=14)
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)
    for bar, val in zip(bars, train_r2):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.03, f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    bars = ax2.bar(fold_labels, test_err, color=pass_colors, edgecolor="black", linewidth=1.8)
    ax2.axhline(mean_test, color="#4f6bed", linestyle="--", linewidth=2, label=f"Mean: {mean_test:.3f} ± {std_test:.3f}")
    ax2.set_ylabel("Held-out Error (normalized units)", fontweight="bold", fontsize=12)
    ax2.set_title("b) Held-out Dose Prediction Error", fontweight="bold", fontsize=14)
    ax2.set_ylim(0, max(test_err) * 1.18)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=10)
    for bar, val in zip(bars, test_err):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    y = np.arange(len(folds))
    ax3.barh(y, [1] * len(folds), color=pass_colors, edgecolor="black", linewidth=1.8)
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"Fold {f['fold']}\n({f['held_out_dose']:g} mg/dl)" for f in folds])
    ax3.set_xticks([])
    ax3.set_xlim(0, 1.25)
    ax3.set_title("c) Biological Plausibility", fontweight="bold", fontsize=14)
    for yi, p in zip(y, passed):
        ax3.text(0.62, yi, "[OK] Pass" if p else "[X] Fail", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax3.invert_yaxis()
    pass_rate = 100.0 * sum(1 for p in passed if p) / len(passed)
    ax3.text(0.5, -0.12, f"Overall: {pass_rate:.0f}% pass rate ({sum(passed)}/{len(passed)} folds)", transform=ax3.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")

    fig.suptitle("Leave-One-Dose-Out Cross-Validation", fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES / "experiment_2_cross_validation_corrected.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


def plot_figure8():
    results = load_json(RESULTS / "experiment_3_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    arch_layers = [r["n_layers"] for r in results["architecture"]]
    arch_success = [r["success_rate"] for r in results["architecture"]]
    arch_r2 = [r["mean_r2"] for r in results["architecture"]]
    ax1.plot(arch_layers, arch_success, marker="o", markersize=10, linewidth=2.8, color="#3498db", label="Success Rate")
    ax1.set_title("a) Architecture Sensitivity", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Number of Layers", fontweight="bold")
    ax1.set_ylabel("Success Rate (%)", fontweight="bold", fontsize=12)
    ax1.set_xticks(arch_layers)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left", framealpha=0.95)
    for x, y, r2 in zip(arch_layers, arch_success, arch_r2):
        ax1.text(x, y + 2, f"R²={r2:.3f}", ha="center", fontweight="bold")

    colloc_points = [r["n_points"] for r in results["collocation"]]
    colloc_success = [r["success_rate"] for r in results["collocation"]]
    colloc_r2 = [r["mean_r2"] for r in results["collocation"]]
    ax2.plot(colloc_points, colloc_success, marker="s", markersize=10, linewidth=2.8, color="#e67e22", label="Success Rate")
    ax2.set_title("b) Collocation Point Sensitivity", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Collocation Points", fontweight="bold")
    ax2.set_ylabel("Success Rate (%)", fontweight="bold", fontsize=12)
    ax2.set_xticks(colloc_points)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", framealpha=0.95)
    for x, y, r2 in zip(colloc_points, colloc_success, colloc_r2):
        ax2.text(x, y + 2, f"R²={r2:.3f}", ha="center", fontweight="bold")

    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES / "experiment_3_hyperparameter_sensitivity_corrected.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


def main():
    outs = [plot_figure6(), plot_figure7(), plot_figure8()]
    for out in outs:
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
