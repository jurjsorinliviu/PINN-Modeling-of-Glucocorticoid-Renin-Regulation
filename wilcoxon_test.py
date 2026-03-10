"""
Statistical Analysis: Mann-Whitney U Test for Ensemble Comparison

This script compares the SW=0.3 and SW=0.5 deterministic PINN ensembles using:
- Mann-Whitney U tests (Wilcoxon rank-sum)
- Cohen's d effect sizes
- Bonferroni correction
- Bootstrap confidence intervals for mean differences
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_ensemble_results(filepath):
    with open(filepath, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_member_metrics(results):
    members = results["member_results"]
    metrics = {
        "r2": [],
        "rmse": [],
        "mae": [],
        "ic50_gap": [],
        "hill_gap": [],
    }

    for member in members:
        metrics["r2"].append(member["metrics"]["r2"])
        metrics["rmse"].append(member["metrics"]["rmse"])
        metrics["mae"].append(member["metrics"]["mae"])
        metrics["ic50_gap"].append(member["metrics"]["ic50_gap"])
        metrics["hill_gap"].append(member["metrics"]["hill_gap"])

    return metrics


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(group1, group2, n_bootstrap=10000, alpha=0.05):
    differences = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        differences.append(np.mean(sample1) - np.mean(sample2))

    differences = np.asarray(differences)
    lower = np.percentile(differences, alpha / 2 * 100)
    upper = np.percentile(differences, (1 - alpha / 2) * 100)
    return lower, upper


def mann_whitney_test(group1, group2, metric_name):
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    effect_size = cohens_d(group1, group2)
    ci_lower, ci_upper = bootstrap_ci(group1, group2)

    return {
        "metric": metric_name,
        "n_group1": len(group1),
        "n_group2": len(group2),
        "mean_group1": float(np.mean(group1)),
        "std_group1": float(np.std(group1, ddof=1)),
        "mean_group2": float(np.mean(group2)),
        "std_group2": float(np.std(group2, ddof=1)),
        "mann_whitney_U": float(statistic),
        "p_value": float(p_value),
        "cohens_d": float(effect_size),
        "difference_means": float(np.mean(group1) - np.mean(group2)),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
    }


def interpret_effect_size(d_value):
    abs_d = abs(d_value)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def generate_latex_table(results):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Comparison of PINN Ensembles: SW=0.3 vs SW=0.5}",
        r"\label{tab:ensemble_comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{SW=0.3} & \textbf{SW=0.5} & \textbf{Diff.} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Effect} \\",
        r"                & $(n=5)$         & $(n=4)$         & (95\% CI)      & (corrected)        &                     & Size \\",
        r"\hline",
    ]

    display_names = {
        "r2": r"$R^2$",
        "rmse": "RMSE",
        "mae": "MAE",
        "ic50_gap": r"IC$_{50}$ Gap",
        "hill_gap": "Hill Gap",
    }

    for result in results:
        p_value = result["p_value_corrected"]
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = ""

        lines.append(
            f"{display_names.get(result['metric'], result['metric'])} & "
            f"{result['mean_group1']:.4f}$\\pm${result['std_group1']:.4f} & "
            f"{result['mean_group2']:.4f}$\\pm${result['std_group2']:.4f} & "
            f"{result['difference_means']:.4f} ({result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}) & "
            f"{p_value:.4f}{sig} & {result['cohens_d']:.2f} & {interpret_effect_size(result['cohens_d'])} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\multicolumn{7}{l}{\small $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (Bonferroni-corrected)} \\",
        r"\multicolumn{7}{l}{\small Mann-Whitney U test with bootstrap 95\% CI (10,000 iterations)} \\",
        r"\multicolumn{7}{l}{\small Effect size interpretation: |d|<0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), >0.8 (large)} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def generate_text_summary(results, alpha_corrected):
    summary = f"""
================================================================================
STATISTICAL COMPARISON: SW=0.3 vs SW=0.5 ENSEMBLES
================================================================================

SAMPLE SIZES:
  - SW=0.3: n = 5 ensemble members
  - SW=0.5: n = 4 ensemble members

STATISTICAL METHOD:
  - Mann-Whitney U test (Wilcoxon rank-sum test)
  - Non-parametric test appropriate for small samples
  - Two-sided alternative hypothesis
  - Bonferroni correction for multiple comparisons (alpha = {alpha_corrected:.4f})

RESULTS:
--------------------------------------------------------------------------------
"""

    for result in results:
        metric = result["metric"].upper()
        summary += f"\n{metric}:\n"
        summary += f"  SW=0.3: {result['mean_group1']:.6f} +/- {result['std_group1']:.6f}\n"
        summary += f"  SW=0.5: {result['mean_group2']:.6f} +/- {result['std_group2']:.6f}\n"
        summary += f"  Difference: {result['difference_means']:.6f}\n"
        summary += f"  95% CI: ({result['ci_95_lower']:.6f}, {result['ci_95_upper']:.6f})\n"
        summary += f"  Mann-Whitney U: {result['mann_whitney_U']:.2f}\n"
        summary += f"  p-value (raw): {result['p_value']:.6f}\n"
        summary += f"  p-value (corrected): {result['p_value_corrected']:.6f}\n"
        summary += f"  Cohen's d: {result['cohens_d']:.4f} ({interpret_effect_size(result['cohens_d'])})\n"
        if result["p_value_corrected"] < 0.05:
            summary += f"  *** SIGNIFICANT at alpha={alpha_corrected:.4f} level ***\n"
        else:
            summary += f"  Not significant at alpha={alpha_corrected:.4f} level\n"

    summary += """
--------------------------------------------------------------------------------
INTERPRETATION NOTES:
1. Small sample sizes (n=4, n=5) limit statistical power
2. Non-parametric test does not assume normality
3. Bonferroni correction controls family-wise error rate
4. Cohen's d provides effect size independent of sample size
5. Bootstrap CI provides robust uncertainty quantification

LIMITATIONS:
- Limited power to detect small effects with n<10 per group
- Type II error (false negative) risk is elevated
- Results should be interpreted alongside domain knowledge
- Larger ensembles (n>=10) recommended for confirmatory studies
================================================================================
"""
    return summary


def main():
    print("=" * 80)
    print("Statistical Analysis: Mann-Whitney U Test for Ensemble Comparison")
    print("=" * 80)

    print("\n[1] Loading ensemble results...")
    sw03_path = Path("results/unified_03/unified_ensemble_03_results.json")
    sw05_path = Path("results/unified/unified_ensemble_results.json")

    if not sw03_path.exists() or not sw05_path.exists():
        print("ERROR: Result files not found.")
        print(f"  Missing: {sw03_path}")
        print(f"  Missing: {sw05_path}")
        return

    sw03_results = load_ensemble_results(sw03_path)
    sw05_results = load_ensemble_results(sw05_path)

    print("[2] Extracting member-level metrics...")
    sw03_metrics = extract_member_metrics(sw03_results)
    sw05_metrics = extract_member_metrics(sw05_results)
    print(f"  SW=0.3: {len(sw03_metrics['r2'])} members")
    print(f"  SW=0.5: {len(sw05_metrics['r2'])} members")

    metrics_to_test = ["r2", "rmse", "ic50_gap"]
    n_tests = len(metrics_to_test)
    alpha = 0.05
    alpha_corrected = alpha / n_tests

    print("\n[3] Performing statistical tests...")
    print(f"  Number of tests: {n_tests}")
    print(f"  Original alpha: {alpha:.4f}")
    print(f"  Bonferroni-corrected alpha: {alpha_corrected:.4f}")

    test_results = []
    for metric in metrics_to_test:
        print(f"\n  Testing {metric.upper()}...")
        result = mann_whitney_test(sw03_metrics[metric], sw05_metrics[metric], metric)
        result["p_value_corrected"] = min(result["p_value"] * n_tests, 1.0)
        test_results.append(result)
        print(f"    Mean diff: {result['difference_means']:.6f}")
        print(f"    p-value (corrected): {result['p_value_corrected']:.6f}")
        print(f"    Cohen's d: {result['cohens_d']:.4f} ({interpret_effect_size(result['cohens_d'])})")

    print("\n[4] Generating outputs...")
    output_json = {
        "analysis_info": {
            "method": "Mann-Whitney U test (Wilcoxon rank-sum test)",
            "alpha": alpha,
            "alpha_corrected": alpha_corrected,
            "n_tests": n_tests,
            "correction_method": "Bonferroni",
            "sw03_n": len(sw03_metrics["r2"]),
            "sw05_n": len(sw05_metrics["r2"]),
            "bootstrap_iterations": 10000,
        },
        "test_results": test_results,
        "raw_data": {
            "sw03": {key: [float(v) for v in values] for key, values in sw03_metrics.items()},
            "sw05": {key: [float(v) for v in values] for key, values in sw05_metrics.items()},
        },
    }

    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "wilcoxon_test_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output_json, handle, indent=2)
    print(f"  [OK] JSON results saved: {json_path}")

    latex_table = generate_latex_table(test_results)
    latex_path = output_dir / "comparison_table.tex"
    latex_path.write_text(latex_table, encoding="utf-8")
    print(f"  [OK] LaTeX table saved: {latex_path}")

    text_summary = generate_text_summary(test_results, alpha_corrected)
    summary_path = output_dir / "wilcoxon_test_summary.txt"
    summary_path.write_text(text_summary, encoding="utf-8")
    print(f"  [OK] Text summary saved: {summary_path}")

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)
    significant = sum(1 for result in test_results if result["p_value_corrected"] < 0.05)
    print(f"Significant comparisons after correction: {significant}/{n_tests}")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
