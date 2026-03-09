"""
Compare available ensemble configurations across synthetic-weight settings.

The script loads the baseline SW=0.5 ensemble, the final SW=0.3 deterministic
ensemble, and the exploratory SW=0.2 ensemble when present. It then generates
comparison figures, LaTeX tables, and a text summary without assuming that all
three result files exist.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAIN_RESULTS = Path("results/unified/unified_ensemble_results.json")
MID_RESULTS = Path("results/unified_03/unified_ensemble_03_results.json")
ALT_RESULTS = Path("results/unified_02/unified_ensemble_02_results.json")
OUTPUT_DIR = Path("results/comparison")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def load_results(path: Path) -> dict:
    """Load ensemble results from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def configuration_label(metrics: dict) -> str:
    """Return a short label for a configuration."""
    sw = metrics.get("synthetic_weight", None)
    if sw == 0.3:
        return "Final"
    if sw == 0.5:
        return "Baseline"
    if sw == 0.2:
        return "Exploratory"
    return "Config"


def extract_metrics(results: dict) -> dict:
    """Extract key metrics from ensemble results."""
    ensemble_metrics = results.get("ensemble_metrics", {})
    members = results.get("member_results", [])

    return {
        "name": results.get("configuration", "Unknown"),
        "n_members": results.get("n_members", 0),
        "synthetic_weight": results.get("hyperparameters", {}).get("synthetic_weight", 0),
        "ensemble_r2": ensemble_metrics.get("r2", 0),
        "ensemble_rmse": ensemble_metrics.get("rmse", 0),
        "ensemble_mae": ensemble_metrics.get("mae", 0),
        "ic50_mean": ensemble_metrics.get("ic50_mean", 0),
        "ic50_std": ensemble_metrics.get("ic50_std", 0),
        "hill_mean": ensemble_metrics.get("hill_mean", 0),
        "hill_std": ensemble_metrics.get("hill_std", 0),
        "ic50_gap_mean": ensemble_metrics.get("ic50_gap_mean", 0),
        "ic50_gap_std": ensemble_metrics.get("ic50_gap_std", 0),
        "hill_gap_mean": ensemble_metrics.get("hill_gap_mean", 0),
        "hill_gap_std": ensemble_metrics.get("hill_gap_std", 0),
        "member_r2": [
            member.get("metrics", {}).get("r2")
            for member in members
            if member.get("metrics", {}).get("r2") is not None
        ],
    }


def parameter_distance(metrics: dict) -> float:
    """Euclidean distance in the IC50/Hill gap plane."""
    return float(
        np.sqrt(metrics["ic50_gap_mean"] ** 2 + metrics["hill_gap_mean"] ** 2)
    )


def plot_comparison(configs: list[dict], output_path: Path):
    """Create a compact comparison figure for the available ensembles."""
    if len(configs) < 2:
        return

    labels = [
        f"{configuration_label(cfg)}\n(SW={cfg['synthetic_weight']})" for cfg in configs
    ]
    colors = ["#2E86AB", "#A23B72", "#3E885B"][: len(configs)]
    x_pos = np.arange(len(configs))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    r2_values = [cfg["ensemble_r2"] for cfg in configs]
    axes[0].bar(x_pos, r2_values, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("R^2 score")
    axes[0].set_title("Accuracy")
    axes[0].grid(axis="y", alpha=0.3)

    distances = [parameter_distance(cfg) for cfg in configs]
    axes[1].bar(x_pos, distances, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Parameter distance")
    axes[1].set_title("Alignment to targets")
    axes[1].grid(axis="y", alpha=0.3)

    member_counts = [cfg["n_members"] for cfg in configs]
    axes[2].bar(x_pos, member_counts, color=colors, alpha=0.85, edgecolor="black")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Accepted members")
    axes[2].set_title("Sampling depth")
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Ensemble configuration comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Comparison plot saved: {output_path}")


def generate_comparison_table(configs: list[dict], output_path: Path):
    """Generate a LaTeX comparison table for the available ensembles."""
    if not configs:
        raise ValueError("No ensemble configurations provided.")

    headers = " & ".join(
        f"\\textbf{{{configuration_label(cfg)}}} (SW={cfg['synthetic_weight']}, n={cfg['n_members']})"
        for cfg in configs
    )
    column_format = "l" + "c" * len(configs)

    def row(metric_name: str, formatter) -> str:
        values = " & ".join(formatter(cfg) for cfg in configs)
        return f"{metric_name} & {values} \\\\\n"

    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Ensemble configuration comparison}\n"
        "\\label{tab:ensemble_comparison}\n"
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n"
        f"\\textbf{{Metric}} & {headers} \\\\\n"
        "\\midrule\n"
        "\\textbf{Model Accuracy}" + " & " * len(configs) + "\\\\\n"
    )
    latex += row("R^2 Score", lambda cfg: f"{cfg['ensemble_r2']:.3f}")
    latex += row("RMSE", lambda cfg: f"{cfg['ensemble_rmse']:.3f}")
    latex += row("MAE", lambda cfg: f"{cfg['ensemble_mae']:.3f}")
    latex += "\\midrule\n"
    latex += "\\textbf{Parameter Alignment}" + " & " * len(configs) + "\\\\\n"
    latex += row("IC50 Gap (log)", lambda cfg: f"{cfg['ic50_gap_mean']:.3f}+-{cfg['ic50_gap_std']:.3f}")
    latex += row("Hill Gap (log)", lambda cfg: f"{cfg['hill_gap_mean']:.3f}+-{cfg['hill_gap_std']:.3f}")
    latex += row("Total Distance", lambda cfg: f"{parameter_distance(cfg):.3f}")
    latex += "\\midrule\n"
    latex += "\\textbf{Estimated Parameters}" + " & " * len(configs) + "\\\\\n"
    latex += row("IC50 (nM)", lambda cfg: f"{cfg['ic50_mean']:.1f}+-{cfg['ic50_std']:.1f}")
    latex += row("Hill Coefficient", lambda cfg: f"{cfg['hill_mean']:.2f}+-{cfg['hill_std']:.2f}")
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] LaTeX table saved: {output_path}")


def generate_text_summary(configs: list[dict], output_path: Path):
    """Generate a human-readable text summary for the available ensembles."""
    if not configs:
        raise ValueError("No ensemble configurations provided.")

    configs = sorted(configs, key=lambda cfg: cfg.get("synthetic_weight", 999))
    lines = [
        "=" * 80,
        "ENSEMBLE CONFIGURATION COMPARISON",
        "=" * 80,
        "",
        "Configuration Details:",
        "---------------------",
    ]

    for cfg in configs:
        sufficient = cfg["n_members"] >= 3
        lines.extend(
            [
                f"{configuration_label(cfg)} Ensemble (SW={cfg['synthetic_weight']}):",
                f"  - Members: {cfg['n_members']} {'[VALID] - sufficient for descriptive statistics' if sufficient else '[CAUTION] - limited sampling'}",
                f"  - Ensemble R^2: {cfg['ensemble_r2']:.3f}",
                f"  - RMSE: {cfg['ensemble_rmse']:.3f}",
                f"  - Total parameter distance: {parameter_distance(cfg):.3f}",
            ]
        )

    best_accuracy = max(configs, key=lambda cfg: cfg["ensemble_r2"])
    best_alignment = min(configs, key=parameter_distance)
    preferred = next(
        (
            cfg
            for cfg in configs
            if cfg.get("synthetic_weight") == 0.3 and cfg["n_members"] >= 3
        ),
        best_accuracy,
    )

    lines.extend(
        [
            "",
            "Key Findings:",
            "------------",
            f"1. Highest ensemble accuracy among loaded configurations: SW={best_accuracy['synthetic_weight']} (R^2={best_accuracy['ensemble_r2']:.3f}).",
            f"2. Tightest parameter alignment among loaded configurations: SW={best_alignment['synthetic_weight']} (distance={parameter_distance(best_alignment):.3f}).",
            "3. Statistical confidence remains limited by the small number of accepted ensemble members.",
            f"4. Recommended manuscript default from the available deterministic runs: SW={preferred['synthetic_weight']}.",
            "",
            "Recommendation:",
            "---------------",
            f"Use SW={preferred['synthetic_weight']} as the primary reported ensemble in manuscript-facing summaries,",
            "and present the other synthetic-weight settings as calibration comparisons rather than alternative paper defaults.",
            "",
            "=" * 80,
            "",
        ]
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Text summary saved: {output_path}")


def main():
    """Run the ensemble comparison workflow."""
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPARISON ANALYSIS")
    print("=" * 80 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    available_configs = []

    print("Loading results...")
    for path in [MAIN_RESULTS, MID_RESULTS, ALT_RESULTS]:
        try:
            results = load_results(path)
            metrics = extract_metrics(results)
            available_configs.append(metrics)
            print(
                f"[OK] Loaded {path}: n={metrics['n_members']}, R^2={metrics['ensemble_r2']:.3f}"
            )
        except FileNotFoundError:
            print(f"[WARN] Missing results: {path}")

    if len(available_configs) < 2:
        print("\n[ERROR] Need at least two ensembles to compare.")
        return

    print("\nGenerating comparison outputs...")
    plot_comparison(available_configs, FIGURES_DIR / "ensemble_comparison.png")
    generate_comparison_table(available_configs, TABLES_DIR / "comparison_table.tex")
    generate_text_summary(available_configs, OUTPUT_DIR / "comparison_summary.txt")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Figure: {FIGURES_DIR / 'ensemble_comparison.png'}")
    print(f"  - LaTeX table: {TABLES_DIR / 'comparison_table.tex'}")
    print(f"  - Text summary: {OUTPUT_DIR / 'comparison_summary.txt'}")
    print()


if __name__ == "__main__":
    main()
