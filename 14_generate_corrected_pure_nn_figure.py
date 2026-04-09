"""
Generate the corrected manuscript figure comparing Pure NN and PINN.

This figure uses finalized saved results only. It does not retrain any models.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "pure_nn_baseline" / "figures" / "pure_nn_vs_pinn_comparison_corrected.png"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    pure_nn = load_json(RESULTS / "pure_nn_baseline" / "pure_nn_results.json")
    pinn_main = load_json(RESULTS / "unified_03" / "unified_ensemble_03_results.json")
    pinn_lodo = load_json(RESULTS / "supplementary_experiments" / "experiment_2_results.json")
    ode = load_json(RESULTS / "ode_baseline_results.json")

    ode_r2 = ode.get("metrics", {}).get("r_squared", ode.get("r2", -0.220))
    ode_rmse = ode.get("metrics", {}).get("rmse", ode.get("rmse", 0.060))

    pure_train_r2 = pure_nn["training_metrics"]["mean_r2"]
    pure_train_rmse = pure_nn["training_metrics"]["mean_rmse"]
    pure_lodo_rmse = pure_nn["cross_validation"]["mean_test_rmse"]
    pure_folds = pure_nn["cross_validation"]["folds"]

    pinn_r2 = pinn_main["ensemble_metrics"]["r2"]
    pinn_rmse = pinn_main["ensemble_metrics"]["rmse"]
    pinn_lodo_rmse = pinn_lodo["avg_test_error"]
    pinn_folds = pinn_lodo["folds"]

    methods = ["ODE\nBaseline", "Pure NN\n(no physics)", "PINN\n(SW=0.3)"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # (a) Main 24 h fit R²
    r2_values = [ode_r2, pure_train_r2, pinn_r2]
    bars = ax1.bar(methods, r2_values, color=colors, alpha=0.78, edgecolor="black", linewidth=2)
    ax1.set_ylabel("R² Score", fontweight="bold", fontsize=12)
    ax1.set_title("(a) Main 24 h Fit Comparison", fontweight="bold", fontsize=13)
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax1.set_ylim([-0.5, 1.1])
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, max(val, 0) + 0.05, f"{val:.3f}",
                 ha="center", va="bottom", fontweight="bold")

    # (b) Strict held-out RMSE
    heldout_methods = ["Pure NN\n(LODO)", "PINN\n(LODO)"]
    heldout_values = [pure_lodo_rmse, pinn_lodo_rmse]
    bars = ax2.bar(heldout_methods, heldout_values, color=[colors[1], colors[2]],
                   alpha=0.78, edgecolor="black", linewidth=2)
    ax2.set_ylabel("Held-out RMSE", fontweight="bold", fontsize=12)
    ax2.set_title("(b) Strict Leave-One-Dose-Out Error", fontweight="bold", fontsize=13)
    ax2.set_ylim([0, max(heldout_values) * 1.25])
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, heldout_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontweight="bold")
    ax2.text(
        0.5,
        0.96,
        "Lower is better; this is a stringent stress test",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    # (c) Fold-wise held-out RMSE by dose
    doses = [float(fold["held_out_dose"]) for fold in pure_folds]
    dose_labels = [f"{dose:g}" for dose in doses]
    pure_fold_rmse = [float(fold["test_rmse"]) for fold in pure_folds]
    pinn_fold_rmse = []
    for dose in doses:
        match = next((fold for fold in pinn_folds if float(fold["held_out_dose"]) == dose), None)
        pinn_fold_rmse.append(float(match["test_error"]) if match is not None else np.nan)

    x = np.arange(len(dose_labels))
    width = 0.36
    bars1 = ax3.bar(x - width / 2, pure_fold_rmse, width, label="Pure NN",
                    color=colors[1], alpha=0.78, edgecolor="black", linewidth=2)
    bars2 = ax3.bar(x + width / 2, pinn_fold_rmse, width, label="PINN",
                    color=colors[2], alpha=0.78, edgecolor="black", linewidth=2)
    ax3.set_ylabel("Held-out RMSE", fontweight="bold", fontsize=12)
    ax3.set_title("(c) Fold-wise Held-out Error by Dose", fontweight="bold", fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(dose_labels)
    ax3.set_xlabel("Held-out dose (mg/dl)", fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax3.grid(axis="y", alpha=0.3)
    for bar_group in (bars1, bars2):
        for bar in bar_group:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}",
                     ha="center", va="bottom", fontsize=8)

    # (d) Main 24 h RMSE
    rmse_values = [ode_rmse, pure_train_rmse, pinn_rmse]
    bars = ax4.bar(methods, rmse_values, color=colors, alpha=0.78, edgecolor="black", linewidth=2)
    ax4.set_ylabel("RMSE (Normalized Units)", fontweight="bold", fontsize=12)
    ax4.set_title("(d) Main 24 h Prediction Error", fontweight="bold", fontsize=13)
    ax4.set_ylim([0, max(rmse_values) * 1.25])
    ax4.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rmse_values):
        ax4.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.002, f"{val:.3f}",
                 ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Pure NN vs. PINN: Main Fit Quality and Held-out Dose Performance",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
