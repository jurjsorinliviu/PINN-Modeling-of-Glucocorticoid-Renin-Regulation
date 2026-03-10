"""
Generate the corrected manuscript dose-response figure.

This figure includes all four measured doses, including the 0 mg/dl control,
while retaining the continuous ensemble curve and descriptive spread.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from src.data import prepare_training_data


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "unified_03" / "figures" / "dose_response_corrected.png"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ensemble = load_json(RESULTS / "unified_03" / "unified_ensemble_03_results.json")
    data = prepare_training_data(dataset="elisa", use_log_scale=False)

    curve = ensemble["dose_response_curve"]
    member_results = ensemble["member_results"]

    # Measured-dose predictions from accepted members.
    measured_predictions = np.array([member["predictions"] for member in member_results], dtype=float)
    measured_mean = measured_predictions.mean(axis=0)
    measured_std = measured_predictions.std(axis=0, ddof=1)

    # Continuous curve from saved report; prepend the control point so 0 mg/dl appears on the plot.
    doses_curve = np.array(curve["dex_range"], dtype=float)
    mean_curve = np.array(curve["mean"], dtype=float)
    std_curve = np.array(curve["std"], dtype=float)

    doses_plot = np.concatenate([[0.0], doses_curve])
    mean_plot = np.concatenate([[measured_mean[0]], mean_curve])
    std_plot = np.concatenate([[measured_std[0]], std_curve])

    obs_doses = np.array(data["dex_concentration"], dtype=float)
    obs_values = np.array(data["renin_normalized"], dtype=float)
    obs_std = np.array(data["renin_std"], dtype=float)

    plt.figure(figsize=(10.5, 6.5))
    ax = plt.gca()

    lower = mean_plot - std_plot
    upper = mean_plot + std_plot

    ax.plot(
        doses_plot,
        mean_plot,
        color="#7f8c8d",
        linewidth=2,
        linestyle="--",
        label="Model-based continuous curve",
        zorder=1,
    )

    ax.fill_between(
        obs_doses,
        measured_mean - measured_std,
        measured_mean + measured_std,
        color="#5DA5DA",
        alpha=0.25,
        label="Accepted ensemble ±1 SD at measured doses",
        zorder=2,
    )
    ax.plot(
        obs_doses,
        measured_mean,
        color="#1f77b4",
        linewidth=3,
        marker="s",
        markersize=7,
        markerfacecolor="#1f77b4",
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="Accepted ensemble mean at measured doses",
        zorder=3,
    )

    ax.errorbar(
        obs_doses,
        obs_values,
        yerr=obs_std,
        fmt="o",
        color="#D62728",
        ecolor="#D62728",
        elinewidth=2,
        capsize=6,
        markersize=9,
        label="Experimental data",
        zorder=3,
    )

    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(-0.02, 100)
    ax.set_ylim(0.72, 1.14)
    ax.set_xlabel("Dexamethasone (mg/dl)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized renin secretion", fontsize=14, fontweight="bold")
    ax.set_title("Dose-response with accepted ensemble variability", fontsize=18, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Clean tick labels so the control is visible.
    ax.set_xticks([0, 0.3, 3, 30, 100])
    ax.set_xticklabels(["0", "0.3", "3", "30", "100"])

    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
