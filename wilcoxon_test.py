"""
Statistical Analysis: Mann-Whitney U Test for Ensemble Comparison
================================================================

This script performs rigorous statistical comparison between PINN ensembles
with different synthetic weight (SW) hyperparameters:
- SW=0.3 (n=5 ensemble members)
- SW=0.5 (n=4 ensemble members)

Tests performed:
- Mann-Whitney U test (Wilcoxon rank-sum test) for non-parametric comparison
- Cohen's d effect size calculation
- Bonferroni correction for multiple testing
- 95% confidence intervals via bootstrap

Author: Sorin Liviu Jurj - Generated for PINN Glucocorticoid-Renin Regulation Study
Date: 2025
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_ensemble_results(filepath):
    """Load ensemble results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_member_metrics(results):
    """Extract individual member metrics from ensemble results."""
    members = results['member_results']
    
    metrics = {
        'r2': [],
        'rmse': [],
        'mae': [],
        'ic50_gap': [],
        'hill_gap': []
    }
    
    for member in members:
        metrics['r2'].append(member['metrics']['r2'])
        metrics['rmse'].append(member['metrics']['rmse'])
        metrics['mae'].append(member['metrics']['mae'])
        metrics['ic50_gap'].append(member['metrics']['ic50_gap'])
        metrics['hill_gap'].append(member['metrics']['hill_gap'])
    
    return metrics


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Cohen's d interpretation:
    - Small: |d| = 0.2
    - Medium: |d| = 0.5
    - Large: |d| = 0.8
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def bootstrap_ci(group1, group2, n_bootstrap=10000, alpha=0.05):
    """
    Calculate bootstrap confidence interval for difference in means.
    """
    differences = []
    
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        differences.append(np.mean(sample1) - np.mean(sample2))
    
    differences = np.array(differences)
    lower = np.percentile(differences, alpha/2 * 100)
    upper = np.percentile(differences, (1 - alpha/2) * 100)
    
    return lower, upper


def mann_whitney_test(group1, group2, metric_name):
    """
    Perform Mann-Whitney U test (Wilcoxon rank-sum test).
    
    This non-parametric test is appropriate for small sample sizes
    and does not assume normal distribution.
    """
    # Mann-Whitney U test (two-sided)
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Calculate effect size (Cohen's d)
    effect_size = cohens_d(group1, group2)
    
    # Bootstrap confidence interval
    ci_lower, ci_upper = bootstrap_ci(group1, group2)
    
    return {
        'metric': metric_name,
        'n_group1': len(group1),
        'n_group2': len(group2),
        'mean_group1': float(np.mean(group1)),
        'std_group1': float(np.std(group1, ddof=1)),
        'mean_group2': float(np.mean(group2)),
        'std_group2': float(np.std(group2, ddof=1)),
        'mann_whitney_U': float(statistic),
        'p_value': float(p_value),
        'cohens_d': float(effect_size),
        'difference_means': float(np.mean(group1) - np.mean(group2)),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper)
    }


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def generate_latex_table(results, alpha_corrected):
    """Generate LaTeX table for manuscript."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Statistical Comparison of PINN Ensembles: SW=0.3 vs SW=0.5}
\label{tab:ensemble_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Metric} & \textbf{SW=0.3} & \textbf{SW=0.5} & \textbf{Diff.} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Effect} \\
                & $(n=5)$         & $(n=4)$         & (95\% CI)      & (corrected)        &                     & Size \\
\hline
"""
    
    for result in results:
        metric = result['metric']
        mean1 = result['mean_group1']
        std1 = result['std_group1']
        mean2 = result['mean_group2']
        std2 = result['std_group2']
        diff = result['difference_means']
        ci_low = result['ci_95_lower']
        ci_high = result['ci_95_upper']
        p_val = result['p_value_corrected']
        cohens = result['cohens_d']
        effect = interpret_effect_size(cohens)
        
        # Format significance stars
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""
        
        # Format metric name
        metric_display = {
            'r2': r'$R^2$',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'ic50_gap': r'IC$_{50}$ Gap',
            'hill_gap': 'Hill Gap'
        }.get(metric, metric)
        
        latex += f"{metric_display} & "
        latex += f"{mean1:.4f}$\\pm${std1:.4f} & "
        latex += f"{mean2:.4f}$\\pm${std2:.4f} & "
        latex += f"{diff:.4f} ({ci_low:.4f}, {ci_high:.4f}) & "
        latex += f"{p_val:.4f}{sig} & "
        latex += f"{cohens:.2f} & "
        latex += f"{effect} \\\\\n"
    
    latex += r"""\hline
\multicolumn{7}{l}{\small $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (Bonferroni-corrected)} \\
\multicolumn{7}{l}{\small Mann-Whitney U test with bootstrap 95\% CI (10,000 iterations)} \\
\multicolumn{7}{l}{\small Effect size interpretation: |d|<0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), >0.8 (large)} \\
\end{tabular}
\end{table}
"""
    
    return latex


def generate_text_summary(results, alpha_corrected):
    """Generate human-readable text summary."""
    summary = """
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
  - Bonferroni correction for multiple comparisons (α = {:.4f})

RESULTS:
--------------------------------------------------------------------------------
""".format(alpha_corrected)
    
    for result in results:
        metric = result['metric'].upper()
        summary += f"\n{metric}:\n"
        summary += f"  SW=0.3: {result['mean_group1']:.6f} ± {result['std_group1']:.6f}\n"
        summary += f"  SW=0.5: {result['mean_group2']:.6f} ± {result['std_group2']:.6f}\n"
        summary += f"  Difference: {result['difference_means']:.6f}\n"
        summary += f"  95% CI: ({result['ci_95_lower']:.6f}, {result['ci_95_upper']:.6f})\n"
        summary += f"  Mann-Whitney U: {result['mann_whitney_U']:.2f}\n"
        summary += f"  p-value (raw): {result['p_value']:.6f}\n"
        summary += f"  p-value (corrected): {result['p_value_corrected']:.6f}\n"
        summary += f"  Cohen's d: {result['cohens_d']:.4f} ({interpret_effect_size(result['cohens_d'])})\n"
        
        if result['p_value_corrected'] < 0.05:
            summary += f"  *** SIGNIFICANT at α={alpha_corrected:.4f} level ***\n"
        else:
            summary += f"  Not significant at α={alpha_corrected:.4f} level\n"
    
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
- Larger ensembles (n≥10) recommended for confirmatory studies
================================================================================
"""
    
    return summary


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("Statistical Analysis: Mann-Whitney U Test for Ensemble Comparison")
    print("="*80)
    
    # Load data
    print("\n[1] Loading ensemble results...")
    sw03_path = Path("results/unified_03/unified_ensemble_03_results.json")
    sw05_path = Path("results/unified/unified_ensemble_results.json")
    
    if not sw03_path.exists() or not sw05_path.exists():
        print("ERROR: Result files not found!")
        print(f"  Looking for: {sw03_path}")
        print(f"  Looking for: {sw05_path}")
        return
    
    sw03_results = load_ensemble_results(sw03_path)
    sw05_results = load_ensemble_results(sw05_path)
    
    # Extract metrics
    print("[2] Extracting member-level metrics...")
    sw03_metrics = extract_member_metrics(sw03_results)
    sw05_metrics = extract_member_metrics(sw05_results)
    
    print(f"  SW=0.3: {len(sw03_metrics['r2'])} members")
    print(f"  SW=0.5: {len(sw05_metrics['r2'])} members")
    
    # Define metrics to test
    metrics_to_test = ['r2', 'rmse', 'ic50_gap']
    n_tests = len(metrics_to_test)
    alpha = 0.05
    alpha_corrected = alpha / n_tests  # Bonferroni correction
    
    print(f"\n[3] Performing statistical tests...")
    print(f"  Number of tests: {n_tests}")
    print(f"  Original α: {alpha:.4f}")
    print(f"  Bonferroni-corrected α: {alpha_corrected:.4f}")
    
    # Perform tests
    test_results = []
    for metric in metrics_to_test:
        print(f"\n  Testing {metric.upper()}...")
        result = mann_whitney_test(
            sw03_metrics[metric],
            sw05_metrics[metric],
            metric
        )
        result['p_value_corrected'] = result['p_value'] * n_tests
        # Ensure corrected p-value doesn't exceed 1.0
        result['p_value_corrected'] = min(result['p_value_corrected'], 1.0)
        test_results.append(result)
        
        print(f"    Mean diff: {result['difference_means']:.6f}")
        print(f"    p-value (corrected): {result['p_value_corrected']:.6f}")
        print(f"    Cohen's d: {result['cohens_d']:.4f} ({interpret_effect_size(result['cohens_d'])})")
    
    # Generate outputs
    print("\n[4] Generating outputs...")
    
    # Save JSON results
    output_json = {
        'analysis_info': {
            'method': 'Mann-Whitney U test (Wilcoxon rank-sum test)',
            'alpha': alpha,
            'alpha_corrected': alpha_corrected,
            'n_tests': n_tests,
            'correction_method': 'Bonferroni',
            'sw03_n': len(sw03_metrics['r2']),
            'sw05_n': len(sw05_metrics['r2']),
            'bootstrap_iterations': 10000
        },
        'test_results': test_results,
        'raw_data': {
            'sw03': {k: [float(v) for v in vals] for k, vals in sw03_metrics.items()},
            'sw05': {k: [float(v) for v in vals] for k, vals in sw05_metrics.items()}
        }
    }
    
    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = output_dir / "wilcoxon_test_results.json"
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"  ✓ JSON results: {json_file}")
    
    # Save LaTeX table
    latex_table = generate_latex_table(test_results, alpha_corrected)
    latex_file = output_dir / "comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"  ✓ LaTeX table: {latex_file}")
    
    # Save text summary
    text_summary = generate_text_summary(test_results, alpha_corrected)
    text_file = output_dir / "wilcoxon_test_summary.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_summary)
    print(f"  ✓ Text summary: {text_file}")
    
    # Print summary to console
    print("\n" + text_summary)
    
    print("\n[5] Analysis complete!")
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nKey findings:")
    for result in test_results:
        metric = result['metric'].upper()
        p_corr = result['p_value_corrected']
        d = result['cohens_d']
        sig = "SIGNIFICANT" if p_corr < alpha_corrected else "not significant"
        print(f"  {metric}: p={p_corr:.4f} ({sig}), d={d:.3f} ({interpret_effect_size(d)})")


if __name__ == "__main__":
    main()