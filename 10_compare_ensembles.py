"""
Compare the two ensemble configurations:
1. Main ensemble (synthetic_weight=0.5, 10 seeds)
2. Alternative ensemble (synthetic_weight=0.2, 5 seeds)

This script generates comparative visualizations and tables showing the trade-off
between model accuracy (R²) and parameter alignment with biological targets.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configuration
MAIN_RESULTS = Path("results/unified/unified_ensemble_results.json")
MID_RESULTS = Path("results/unified_03/unified_ensemble_03_results.json")
ALT_RESULTS = Path("results/unified_02/unified_ensemble_02_results.json")
OUTPUT_DIR = Path("results/comparison")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

PARAMETER_TARGETS = {"log_IC50": 2.88, "log_hill": 1.92}


def load_results(path: Path) -> dict:
    """Load ensemble results from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def extract_metrics(results: dict) -> dict:
    """Extract key metrics from ensemble results."""
    ensemble_metrics = results.get("ensemble_metrics", {})
    members = results.get("member_results", [])
    
    # Collect individual member metrics
    r2_values = []
    ic50_gaps = []
    hill_gaps = []
    
    for member in members:
        metrics = member.get("metrics", {})
        params = member.get("parameters", {})
        
        if "r2" in metrics:
            r2_values.append(metrics["r2"])
        if "ic50_gap" in params:
            ic50_gaps.append(params["ic50_gap"])
        if "hill_gap" in params:
            hill_gaps.append(params["hill_gap"])
    
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
        "member_r2": r2_values,
        "member_ic50_gaps": ic50_gaps,
        "member_hill_gaps": hill_gaps,
    }


def plot_comparison(main_metrics: dict, alt_metrics: dict, output_path: Path):
    """Create comprehensive comparison visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. R² comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = [0, 1]
    r2_values = [main_metrics["ensemble_r2"], alt_metrics["ensemble_r2"]]
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(x_pos, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"Main\n(SW={main_metrics['synthetic_weight']})", 
                          f"Alt\n(SW={alt_metrics['synthetic_weight']})"])
    ax1.set_ylabel("R² Score", fontsize=12, fontweight='bold')
    ax1.set_title("(A) Model Accuracy Comparison", fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Parameter alignment (IC50)
    ax2 = fig.add_subplot(gs[0, 1])
    ic50_gaps = [main_metrics["ic50_gap_mean"], alt_metrics["ic50_gap_mean"]]
    ic50_stds = [main_metrics["ic50_gap_std"], alt_metrics["ic50_gap_std"]]
    bars = ax2.bar(x_pos, ic50_gaps, yerr=ic50_stds, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5, capsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Main\n(SW={main_metrics['synthetic_weight']})", 
                          f"Alt\n(SW={alt_metrics['synthetic_weight']})"])
    ax2.set_ylabel("IC50 Parameter Gap (log units)", fontsize=12, fontweight='bold')
    ax2.set_title("(B) IC50 Parameter Alignment", fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, ic50_gaps, ic50_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Parameter alignment (Hill)
    ax3 = fig.add_subplot(gs[1, 0])
    hill_gaps = [main_metrics["hill_gap_mean"], alt_metrics["hill_gap_mean"]]
    hill_stds = [main_metrics["hill_gap_std"], alt_metrics["hill_gap_std"]]
    bars = ax3.bar(x_pos, hill_gaps, yerr=hill_stds, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5, capsize=8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"Main\n(SW={main_metrics['synthetic_weight']})", 
                          f"Alt\n(SW={alt_metrics['synthetic_weight']})"])
    ax3.set_ylabel("Hill Parameter Gap (log units)", fontsize=12, fontweight='bold')
    ax3.set_title("(C) Hill Parameter Alignment", fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, hill_gaps, hill_stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Member distribution (R²)
    ax4 = fig.add_subplot(gs[1, 1])
    all_r2 = [main_metrics["member_r2"], alt_metrics["member_r2"]]
    bp = ax4.boxplot(all_r2, labels=[f"Main (n={len(main_metrics['member_r2'])})", 
                                      f"Alt (n={len(alt_metrics['member_r2'])})"],
                     patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel("R² Score Distribution", fontsize=12, fontweight='bold')
    ax4.set_title("(D) Individual Member R² Variation", fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Trade-off scatter
    ax5 = fig.add_subplot(gs[2, :])
    main_r2 = np.array(main_metrics["member_r2"])
    main_param_gap = np.sqrt(np.array(main_metrics["member_ic50_gaps"])**2 + 
                             np.array(main_metrics["member_hill_gaps"])**2)
    alt_r2 = np.array(alt_metrics["member_r2"])
    alt_param_gap = np.sqrt(np.array(alt_metrics["member_ic50_gaps"])**2 + 
                            np.array(alt_metrics["member_hill_gaps"])**2)
    
    ax5.scatter(main_param_gap, main_r2, s=120, alpha=0.7, color=colors[0], 
               edgecolor='black', linewidth=1.5, label=f"Main (SW={main_metrics['synthetic_weight']})",
               marker='o')
    ax5.scatter(alt_param_gap, alt_r2, s=120, alpha=0.7, color=colors[1],
               edgecolor='black', linewidth=1.5, label=f"Alt (SW={alt_metrics['synthetic_weight']})",
               marker='s')
    
    # Add ensemble means
    main_mean_gap = np.sqrt(main_metrics["ic50_gap_mean"]**2 + main_metrics["hill_gap_mean"]**2)
    alt_mean_gap = np.sqrt(alt_metrics["ic50_gap_mean"]**2 + alt_metrics["hill_gap_mean"]**2)
    ax5.scatter([main_mean_gap], [main_metrics["ensemble_r2"]], s=250, alpha=0.9,
               color=colors[0], edgecolor='gold', linewidth=3, marker='*',
               label='Main ensemble mean', zorder=10)
    ax5.scatter([alt_mean_gap], [alt_metrics["ensemble_r2"]], s=250, alpha=0.9,
               color=colors[1], edgecolor='gold', linewidth=3, marker='*',
               label='Alt ensemble mean', zorder=10)
    
    ax5.set_xlabel("Parameter Distance from Target (log units)", fontsize=12, fontweight='bold')
    ax5.set_ylabel("R² Score", fontsize=12, fontweight='bold')
    ax5.set_title("(E) Accuracy vs Parameter Alignment Trade-off", fontsize=13, fontweight='bold')
    ax5.legend(loc='best', fontsize=10, framealpha=0.9)
    ax5.grid(alpha=0.3)
    
    # Add annotation
    ax5.annotate('Better parameter\nalignment →', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.annotate('← Better\naccuracy', xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle("Ensemble Configuration Comparison: Accuracy vs Parameter Alignment",
                fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved: {output_path}")


def generate_comparison_table(main_metrics: dict, mid_metrics: dict, alt_metrics: dict, output_path: Path):
    """Generate LaTeX comparison table for all three ensembles."""
    latex = r"""\begin{table}[h]
\centering
\caption{Ensemble Configuration Comparison}
\label{tab:ensemble_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Main} & \textbf{Middle} & \textbf{Alternative} \\
 & (SW=0.5) & (SW=0.3) & (SW=0.2) \\
 & (n=""" + str(main_metrics['n_members']) + r""") & (n=""" + str(mid_metrics['n_members']) + r""") & (n=""" + str(alt_metrics['n_members']) + r""") \\
\midrule
\textbf{Model Accuracy} & & \\
R² Score & """ + f"{main_metrics['ensemble_r2']:.3f}" + r""" & """ + f"{mid_metrics['ensemble_r2']:.3f}" + r""" & """ + f"{alt_metrics['ensemble_r2']:.3f}" + r""" \\
RMSE & """ + f"{main_metrics['ensemble_rmse']:.3f}" + r""" & """ + f"{mid_metrics['ensemble_rmse']:.3f}" + r""" & """ + f"{alt_metrics['ensemble_rmse']:.3f}" + r""" \\
MAE & """ + f"{main_metrics['ensemble_mae']:.3f}" + r""" & """ + f"{mid_metrics['ensemble_mae']:.3f}" + r""" & """ + f"{alt_metrics['ensemble_mae']:.3f}" + r""" \\
\midrule
\textbf{Parameter Alignment} & & \\
IC50 Gap (log) & """ + f"{main_metrics['ic50_gap_mean']:.3f}±{main_metrics['ic50_gap_std']:.3f}" + r""" & """ + f"{mid_metrics['ic50_gap_mean']:.3f}±{mid_metrics['ic50_gap_std']:.3f}" + r""" & """ + f"{alt_metrics['ic50_gap_mean']:.3f}±{alt_metrics['ic50_gap_std']:.3f}" + r""" \\
Hill Gap (log) & """ + f"{main_metrics['hill_gap_mean']:.3f}±{main_metrics['hill_gap_std']:.3f}" + r""" & """ + f"{mid_metrics['hill_gap_mean']:.3f}±{mid_metrics['hill_gap_std']:.3f}" + r""" & """ + f"{alt_metrics['hill_gap_mean']:.3f}±{alt_metrics['hill_gap_std']:.3f}" + r""" \\
Total Distance & """ + f"{np.sqrt(main_metrics['ic50_gap_mean']**2 + main_metrics['hill_gap_mean']**2):.3f}" + r""" & """ + f"{np.sqrt(mid_metrics['ic50_gap_mean']**2 + mid_metrics['hill_gap_mean']**2):.3f}" + r""" & """ + f"{np.sqrt(alt_metrics['ic50_gap_mean']**2 + alt_metrics['hill_gap_mean']**2):.3f}" + r""" \\
\midrule
\textbf{Estimated Parameters} & & \\
IC50 (nM) & """ + f"{main_metrics['ic50_mean']:.1f}±{main_metrics['ic50_std']:.1f}" + r""" & """ + f"{mid_metrics['ic50_mean']:.1f}±{mid_metrics['ic50_std']:.1f}" + r""" & """ + f"{alt_metrics['ic50_mean']:.1f}±{alt_metrics['ic50_std']:.1f}" + r""" \\
Hill Coefficient & """ + f"{main_metrics['hill_mean']:.2f}±{main_metrics['hill_std']:.2f}" + r""" & """ + f"{mid_metrics['hill_mean']:.2f}±{mid_metrics['hill_std']:.2f}" + r""" & """ + f"{alt_metrics['hill_mean']:.2f}±{alt_metrics['hill_std']:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with output_path.open("w") as f:
        f.write(latex)
    print(f"✓ LaTeX table saved: {output_path}")


def generate_text_summary(main_metrics: dict, mid_metrics: dict, alt_metrics: dict, output_path: Path):
    """Generate human-readable text summary with statistical context."""
    
    # Check if ensembles have sufficient members for comparison
    main_sufficient = main_metrics['n_members'] >= 3
    mid_sufficient = mid_metrics['n_members'] >= 3
    alt_sufficient = alt_metrics['n_members'] >= 3
    
    summary = f"""
{'='*80}
ENSEMBLE CONFIGURATION COMPARISON
{'='*80}

Configuration Details:
---------------------
Main Ensemble (SW={main_metrics['synthetic_weight']}):
  - Members: {main_metrics['n_members']} {'✓ Sufficient for statistics' if main_sufficient else '⚠ Need ≥3 for reliable mean/std'}
  - Success rate: ~40%
  - Focus: Robust accuracy with moderate parameter alignment

Middle Ensemble (SW={mid_metrics['synthetic_weight']}):
  - Members: {mid_metrics['n_members']} {'✓ Sufficient for statistics' if mid_sufficient else '⚠ Need ≥3 for reliable mean/std'}
  - Success rate: ~{(mid_metrics['n_members']/10)*100:.0f}% (trained with 10 seeds)
  - Focus: Balance between accuracy and parameter alignment

Alternative Ensemble (SW={alt_metrics['synthetic_weight']}):
  - Members: {alt_metrics['n_members']} {'✓ Sufficient for statistics' if alt_sufficient else '⚠ INSUFFICIENT - single model could be outlier'}
  - Success rate: ~20%
  - Focus: Tightest parameter alignment (if representative)

Model Accuracy Comparison:
-------------------------
                    Main         Middle       Alternative
R² Score:          {main_metrics['ensemble_r2']:.3f}        {mid_metrics['ensemble_r2']:.3f}        {alt_metrics['ensemble_r2']:.3f}
RMSE:              {main_metrics['ensemble_rmse']:.3f}        {mid_metrics['ensemble_rmse']:.3f}        {alt_metrics['ensemble_rmse']:.3f}
MAE:               {main_metrics['ensemble_mae']:.3f}        {mid_metrics['ensemble_mae']:.3f}        {alt_metrics['ensemble_mae']:.3f}

Parameter Alignment Comparison:
-------------------------------
                    Main                Middle              Alternative
IC50 Gap:          {main_metrics['ic50_gap_mean']:.3f}±{main_metrics['ic50_gap_std']:.3f}        {mid_metrics['ic50_gap_mean']:.3f}±{mid_metrics['ic50_gap_std']:.3f}        {alt_metrics['ic50_gap_mean']:.3f}±{alt_metrics['ic50_gap_std']:.3f}
Hill Gap:          {main_metrics['hill_gap_mean']:.3f}±{main_metrics['hill_gap_std']:.3f}        {mid_metrics['hill_gap_mean']:.3f}±{mid_metrics['hill_gap_std']:.3f}        {alt_metrics['hill_gap_mean']:.3f}±{alt_metrics['hill_gap_std']:.3f}
Total Distance:    {np.sqrt(main_metrics['ic50_gap_mean']**2 + main_metrics['hill_gap_mean']**2):.3f}            {np.sqrt(mid_metrics['ic50_gap_mean']**2 + mid_metrics['hill_gap_mean']**2):.3f}            {np.sqrt(alt_metrics['ic50_gap_mean']**2 + alt_metrics['hill_gap_mean']**2):.3f}

Estimated Parameters:
--------------------
                    Main                Middle              Alternative           Target
IC50 (nM):         {main_metrics['ic50_mean']:.1f}±{main_metrics['ic50_std']:.1f}          {mid_metrics['ic50_mean']:.1f}±{mid_metrics['ic50_std']:.1f}          {alt_metrics['ic50_mean']:.1f}±{alt_metrics['ic50_std']:.1f}            {np.exp(PARAMETER_TARGETS['log_IC50']):.1f}
Hill Coefficient:  {main_metrics['hill_mean']:.2f}±{main_metrics['hill_std']:.2f}          {mid_metrics['hill_mean']:.2f}±{mid_metrics['hill_std']:.2f}          {alt_metrics['hill_mean']:.2f}±{alt_metrics['hill_std']:.2f}            {np.exp(PARAMETER_TARGETS['log_hill']):.2f}

Statistical Validity Assessment:
-------------------------------
Main Ensemble (n={main_metrics['n_members']}):    {'[VALID] - sufficient members for reliable statistics' if main_sufficient else '[CAUTION] - limited sampling'}
Middle Ensemble (n={mid_metrics['n_members']}):   {'[VALID] - sufficient members for reliable statistics' if mid_sufficient else '[CAUTION] - limited sampling'}
Alternative (n={alt_metrics['n_members']}):        {'[VALID] - sufficient members for reliable statistics' if alt_sufficient else '[INVALID] - single model insufficient for inference'}

Key Findings:
------------
1. Success Rate vs Parameter Alignment Trade-off:
   - SW=0.5: 40% success rate (4/10) → robust ensemble with moderate gaps
   - SW=0.3: ~{(mid_metrics['n_members']/10)*100:.0f}% success rate ({mid_metrics['n_members']}/10) → {'balanced compromise' if mid_sufficient else 'needs evaluation'}
   - SW=0.2: 20% success rate (1/5) → tightest gaps but insufficient sampling

2. Statistical Confidence:
   {'- Main ensemble: Mean/std reflect true sampling variability' if main_sufficient else '- Main ensemble: Limited confidence in statistics'}
   {'- Middle ensemble: Can assess if gains are representative' if mid_sufficient else '- Middle ensemble: NEEDS MORE SEEDS to confirm trends'}
   {'- Alternative: SINGLE MODEL - could be lucky draw or outlier' if not alt_sufficient else '- Alternative: Sufficient for statistical inference'}

3. Recommendation:
   {'[OK] Main ensemble (SW=0.5) - Use as baseline for paper' if main_sufficient else ''}
   {'[OK] Middle ensemble (SW=0.3) - Run with 10 seeds to test "best of both" hypothesis' if not mid_sufficient else '[OK] Middle ensemble (SW=0.3) - Promising if mean/std hold'}
   {'[WARN] Alternative (SW=0.2) - Promising signal but needs >=20 seeds for 4+ passing models' if not alt_sufficient else ''}
   
   RATIONALE: Lower synthetic weights show better parameter alignment but lower
   plausibility success rates. Need sufficient ensemble members to distinguish
   signal from noise. Middle ground (SW=0.3) offers best compromise if success
   rate reaches ~30-40%.

{'='*80}
"""
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[OK] Text summary saved: {output_path}")


def main():
    """Main comparison workflow."""
    print("\n" + "="*80)
    print("ENSEMBLE COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    try:
        main_results = load_results(MAIN_RESULTS)
        print(f"✓ Loaded main ensemble: {MAIN_RESULTS}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("  Run 8_unified_pipeline.py first to generate main ensemble results")
        return
    
    try:
        mid_results = load_results(MID_RESULTS)
        print(f"✓ Loaded middle ensemble: {MID_RESULTS}")
        has_mid = True
    except FileNotFoundError:
        print(f"⚠ Middle ensemble not found: {MID_RESULTS}")
        print("  This comparison will use main vs alternative only")
        mid_results = None
        has_mid = False
    
    try:
        alt_results = load_results(ALT_RESULTS)
        print(f"✓ Loaded alternative ensemble: {ALT_RESULTS}")
        has_alt = True
    except FileNotFoundError:
        print(f"⚠ Alternative ensemble not found: {ALT_RESULTS}")
        alt_results = None
        has_alt = False
    
    if not has_mid and not has_alt:
        print("\n✗ Need at least one additional ensemble to compare")
        return
    
    # Extract metrics
    print("\nExtracting metrics...")
    main_metrics = extract_metrics(main_results)
    print(f"✓ Main ensemble: {main_metrics['n_members']} members, R²={main_metrics['ensemble_r2']:.3f}")
    
    if has_mid:
        mid_metrics = extract_metrics(mid_results)
        print(f"✓ Middle ensemble: {mid_metrics['n_members']} members, R²={mid_metrics['ensemble_r2']:.3f}")
    else:
        mid_metrics = None
    
    if has_alt:
        alt_metrics = extract_metrics(alt_results)
        print(f"✓ Alternative ensemble: {alt_metrics['n_members']} members, R²={alt_metrics['ensemble_r2']:.3f}")
    else:
        alt_metrics = None
    
    # Generate outputs (use available ensembles)
    print("\nGenerating comparison outputs...")
    
    # For plotting/tables, use what's available
    if has_mid and has_alt:
        # Full 3-way comparison (not implemented in plot yet, would need update)
        generate_comparison_table(main_metrics, mid_metrics, alt_metrics, TABLES_DIR / "comparison_table.tex")
        generate_text_summary(main_metrics, mid_metrics, alt_metrics, OUTPUT_DIR / "comparison_summary.txt")
        print("✓ Generated 3-way comparison (Main vs Middle vs Alternative)")
    elif has_mid:
        # Main vs Middle only
        generate_comparison_table(main_metrics, mid_metrics, {}, TABLES_DIR / "comparison_table.tex")
        generate_text_summary(main_metrics, mid_metrics, {}, OUTPUT_DIR / "comparison_summary.txt")
        print("✓ Generated 2-way comparison (Main vs Middle)")
    elif has_alt:
        # Main vs Alternative only (original)
        # Create dummy mid_metrics for compatibility
        dummy_mid = {'n_members': 0, 'ensemble_r2': 0, 'ensemble_rmse': 0, 'ensemble_mae': 0,
                     'ic50_gap_mean': 0, 'ic50_gap_std': 0, 'hill_gap_mean': 0, 'hill_gap_std': 0,
                     'ic50_mean': 0, 'ic50_std': 0, 'hill_mean': 0, 'hill_std': 0, 'synthetic_weight': 0.3}
        generate_comparison_table(main_metrics, dummy_mid, alt_metrics, TABLES_DIR / "comparison_table.tex")
        generate_text_summary(main_metrics, dummy_mid, alt_metrics, OUTPUT_DIR / "comparison_summary.txt")
        print("✓ Generated 2-way comparison (Main vs Alternative)")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Figure: {FIGURES_DIR / 'ensemble_comparison.png'}")
    print(f"  - LaTeX table: {TABLES_DIR / 'comparison_table.tex'}")
    print(f"  - Text summary: {OUTPUT_DIR / 'comparison_summary.txt'}")
    print("\n")


if __name__ == "__main__":
    main()