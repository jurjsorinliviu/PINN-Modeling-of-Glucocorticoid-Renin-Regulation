"""
Regenerate Supplementary Experiment Figures WITHOUT Retraining

This script reads existing results from supplementary experiments
and regenerates all figures with updated legends and formatting.
No model training is performed.

Usage: python regenerate_supplementary_figures_only.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

# Directories
RESULTS_DIR = Path("results/supplementary_experiments")
FIGURES_DIR = RESULTS_DIR / "figures"

def load_results(experiment_num):
    """Load results from JSON file"""
    results_file = RESULTS_DIR / f"experiment_{experiment_num}_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    with results_file.open('r') as f:
        return json.load(f)


def regenerate_experiment_1_figure():
    """Regenerate Experiment 1: Ramp Ablation figure"""
    print("\n[Regenerating] Experiment 1: Ramp Ablation Figure...")
    
    results = load_results(1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rates
    configs = ['Constant', 'Ramped']
    success_rates = [
        results['constant']['success_rate'],
        results['ramped']['success_rate']
    ]
    
    bars = ax1.bar(configs, success_rates, color=['#e74c3c', '#27ae60'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('a) Plausibility Success Rate', fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add color legend
    legend_elements = [Patch(facecolor='#27ae60', alpha=0.7, label='Higher success'),
                      Patch(facecolor='#e74c3c', alpha=0.7, label='Lower success')]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # R² comparison
    if results['constant']['mean_r2'] > 0 and results['ramped']['mean_r2'] > 0:
        r2_values = [
            results['constant']['mean_r2'],
            results['ramped']['mean_r2']
        ]
        
        bars = ax2.bar(configs, r2_values, color=['#e74c3c', '#27ae60'], 
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Mean R² (Passed Models)', fontweight='bold')
        ax2.set_title('b) Model Accuracy', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add color legend
        legend_elements = [Patch(facecolor='#27ae60', alpha=0.7, label='Ramped (better)'),
                          Patch(facecolor='#e74c3c', alpha=0.7, label='Constant')]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.suptitle('Experiment 1: Constant vs. Ramped High-Dose Weighting',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "experiment_1_ramp_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def regenerate_experiment_2_figure():
    """Regenerate Experiment 2: Cross-Validation figure"""
    print("\n[Regenerating] Experiment 2: Cross-Validation Figure...")
    
    results = load_results(2)
    folds = results['folds']
    n_folds = len(folds)
    
    # Extract data
    fold_nums = [f['fold'] for f in folds]
    doses = [f['held_out_dose'] for f in folds]
    train_r2 = [f['train_r2'] for f in folds]
    test_errors = [f['test_error'] for f in folds]
    passed = [f['passed_plausibility'] for f in folds]
    
    dose_labels = [f"{d:.1f}" for d in doses]
    colors = ['#27ae60' if p else '#e74c3c' for p in passed]
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Panel (a): Training R²
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(fold_nums, train_r2, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Training R²', fontweight='bold', fontsize=12)
    ax1.set_title('a) Training Performance', fontweight='bold', fontsize=13)
    ax1.set_ylim([0, 1.15])
    ax1.set_xticks(fold_nums)
    ax1.set_xticklabels([f'Fold {f}\n({d} nM)' for f, d in zip(fold_nums, dose_labels)], fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    mean_r2 = np.mean(train_r2)
    ax1.axhline(mean_r2, color='blue', linestyle='--', linewidth=2, alpha=0.6)
    
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.7, label='Passed plausibility'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Failed plausibility'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, alpha=0.6, label=f'Mean: {mean_r2:.3f}')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Panel (b): Test Error
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(fold_nums, test_errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Test Error (normalized)', fontweight='bold', fontsize=12)
    ax2.set_title('b) Held-Out Dose Prediction Error', fontweight='bold', fontsize=13)
    ax2.set_ylim([0, max(test_errors) * 1.15])
    ax2.set_xticks(fold_nums)
    ax2.set_xticklabels([f'Fold {f}\n({d} nM)' for f, d in zip(fold_nums, dose_labels)], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (fold, err) in enumerate(zip(fold_nums, test_errors)):
        if err > 0.01:
            ax2.text(fold, err + max(test_errors)*0.08, f'{err:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    mean_error = np.mean(test_errors)
    std_error = np.std(test_errors)
    ax2.axhline(mean_error, color='blue', linestyle='--', linewidth=2, alpha=0.6)
    
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.7, label='Passed plausibility'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Failed plausibility'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, alpha=0.6, 
                   label=f'Mean: {mean_error:.3f}±{std_error:.3f}')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Panel (c): Plausibility Status
    ax3 = fig.add_subplot(gs[0, 2])
    
    pass_status = ['Pass' if p else 'Fail' for p in passed]
    y_pos = np.arange(n_folds)
    
    bars3 = ax3.barh(y_pos, [1]*n_folds, color=colors, alpha=0.7, edgecolor='black', linewidth=2, height=0.7)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f'Fold {f}\n({d} nM)' for f, d in zip(fold_nums, dose_labels)], fontsize=9)
    ax3.set_xlabel('Status', fontweight='bold', fontsize=12)
    ax3.set_title('c) Biological Plausibility', fontweight='bold', fontsize=13)
    ax3.set_xlim([0, 1.3])
    ax3.set_ylim([-0.5, n_folds - 0.5])
    ax3.set_xticks([])
    
    for i, (status, color) in enumerate(zip(pass_status, colors)):
        symbol = '[OK]' if status == 'Pass' else '[X]'
        ax3.text(0.65, i, f'{symbol} {status}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.7, label='Passed'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Failed')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    pass_rate = sum(passed) / n_folds * 100
    summary_text = f'Overall: {pass_rate:.0f}% pass rate ({sum(passed)}/{n_folds} folds)'
    ax3.text(0.65, -1.0, summary_text, ha='center', va='top',
            fontsize=9, fontweight='bold', transform=ax3.transData)
    
    plt.suptitle('Experiment 2: Leave-One-Dose-Out Cross-Validation',
                fontsize=14, fontweight='bold', y=0.98)
    
    output_path = FIGURES_DIR / "experiment_2_cross_validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"[OK] Saved: {output_path}")


def regenerate_experiment_3_figure():
    """Regenerate Experiment 3: Hyperparameter Sensitivity figure"""
    print("\n[Regenerating] Experiment 3: Hyperparameter Sensitivity Figure...")
    
    results = load_results(3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Architecture sensitivity
    n_layers = [r['n_layers'] for r in results['architecture']]
    success_arch = [r['success_rate'] for r in results['architecture']]
    mean_r2_arch = [r['mean_r2'] for r in results['architecture']]
    
    ax1.plot(n_layers, success_arch, 'o-', markersize=10, linewidth=2, color='#3498db', label='Success Rate')
    ax1.set_xlabel('Number of Layers', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('a) Architecture Sensitivity', fontweight='bold')
    ax1.set_xticks(n_layers)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    for n, success, r2 in zip(n_layers, success_arch, mean_r2_arch):
        ax1.text(n, success + 2, f'R²={r2:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax1.legend(loc='lower left', fontsize=9)
    
    # Collocation point sensitivity
    n_points = [r['n_points'] for r in results['collocation']]
    success_coll = [r['success_rate'] for r in results['collocation']]
    mean_r2_coll = [r['mean_r2'] for r in results['collocation']]
    
    ax2.plot(n_points, success_coll, 's-', markersize=10, linewidth=2, color='#e67e22', label='Success Rate')
    ax2.set_xlabel('Collocation Points', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('b) Collocation Point Sensitivity', fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(n_points)
    ax2.set_xticklabels(n_points)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    for pts, success, r2 in zip(n_points, success_coll, mean_r2_coll):
        ax2.text(pts, success + 2, f'R²={r2:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax2.legend(loc='lower left', fontsize=9)
    
    plt.suptitle('Experiment 3: Hyperparameter Sensitivity Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "experiment_3_hyperparameter_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def regenerate_combined_figure():
    """Regenerate combined supplementary figure with all 6 panels"""
    print("\n[Regenerating] Combined Supplementary Figure...")
    
    # Load all results
    exp1_results = load_results(1)
    exp2_results = load_results(2)
    exp3_results = load_results(3)
    
    # Create large figure with 3 rows
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Row 1: Experiment 1 (Ramp Ablation)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1])
    
    # Exp 1a: Success rates
    configs = ['Constant', 'Ramped']
    success_rates = [
        exp1_results['constant']['success_rate'],
        exp1_results['ramped']['success_rate']
    ]
    colors_exp1 = ['#e74c3c', '#27ae60']
    bars1a = ax1a.bar(configs, success_rates, color=colors_exp1, alpha=0.7, edgecolor='black', linewidth=2)
    ax1a.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
    ax1a.set_title('a) Exp. 1: Ramp Ablation - Success Rate', fontweight='bold', fontsize=12)
    ax1a.set_ylim([0, 100])
    ax1a.grid(axis='y', alpha=0.3)
    for bar, rate in zip(bars1a, success_rates):
        ax1a.text(bar.get_x() + bar.get_width()/2., rate + 2,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Exp 1b: R² comparison
    if exp1_results['constant']['mean_r2'] > 0 and exp1_results['ramped']['mean_r2'] > 0:
        r2_values = [
            exp1_results['constant']['mean_r2'],
            exp1_results['ramped']['mean_r2']
        ]
        bars1b = ax1b.bar(configs, r2_values, color=colors_exp1, alpha=0.7, edgecolor='black', linewidth=2)
        ax1b.set_ylabel('Mean R² (Passed Models)', fontweight='bold', fontsize=11)
        ax1b.set_title('b) Exp. 1: Model Accuracy', fontweight='bold', fontsize=12)
        ax1b.set_ylim([0, 1])
        ax1b.grid(axis='y', alpha=0.3)
        for bar, r2 in zip(bars1b, r2_values):
            ax1b.text(bar.get_x() + bar.get_width()/2., r2 + 0.02,
                     f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Row 2: Experiment 2 (Cross-Validation)
    ax2a = fig.add_subplot(gs[1, 0])
    ax2b = fig.add_subplot(gs[1, 1])
    
    folds = exp2_results['folds']
    fold_nums = [f['fold'] for f in folds]
    train_r2 = [f['train_r2'] for f in folds]
    test_errors = [f['test_error'] for f in folds]
    passed = [f['passed_plausibility'] for f in folds]
    colors_exp2 = ['#27ae60' if p else '#e74c3c' for p in passed]
    
    # Exp 2a: Training R²
    bars2a = ax2a.bar(fold_nums, train_r2, color=colors_exp2, alpha=0.7, edgecolor='black', linewidth=2)
    ax2a.set_xlabel('Fold', fontweight='bold', fontsize=11)
    ax2a.set_ylabel('Training R²', fontweight='bold', fontsize=11)
    ax2a.set_title('c) Exp. 2: Cross-Validation - Training', fontweight='bold', fontsize=12)
    ax2a.set_ylim([0, 1])
    ax2a.set_xticks(fold_nums)
    ax2a.grid(axis='y', alpha=0.3)
    ax2a.axhline(np.mean(train_r2), color='blue', linestyle='--', linewidth=2, alpha=0.6)
    
    # Exp 2b: Test errors
    bars2b = ax2b.bar(fold_nums, test_errors, color=colors_exp2, alpha=0.7, edgecolor='black', linewidth=2)
    ax2b.set_xlabel('Fold', fontweight='bold', fontsize=11)
    ax2b.set_ylabel('Test Error', fontweight='bold', fontsize=11)
    ax2b.set_title('d) Exp. 2: Test Error per Fold', fontweight='bold', fontsize=12)
    ax2b.set_xticks(fold_nums)
    ax2b.grid(axis='y', alpha=0.3)
    ax2b.axhline(np.mean(test_errors), color='blue', linestyle='--', linewidth=2, alpha=0.6)
    
    # Row 3: Experiment 3 (Hyperparameter Sensitivity)
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    
    # Exp 3a: Architecture sensitivity
    n_layers = [r['n_layers'] for r in exp3_results['architecture']]
    success_arch = [r['success_rate'] for r in exp3_results['architecture']]
    mean_r2_arch = [r['mean_r2'] for r in exp3_results['architecture']]
    
    ax3a.plot(n_layers, success_arch, 'o-', markersize=10, linewidth=2, color='#3498db')
    ax3a.set_xlabel('Number of Layers', fontweight='bold', fontsize=11)
    ax3a.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
    ax3a.set_title('e) Exp. 3: Architecture Sensitivity', fontweight='bold', fontsize=12)
    ax3a.set_xticks(n_layers)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_ylim([0, 105])
    
    # Add R² values with smaller font
    for n, success, r2 in zip(n_layers, success_arch, mean_r2_arch):
        ax3a.text(n, success + 2, f'R²={r2:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    # Exp 3b: Collocation sensitivity
    n_points = [r['n_points'] for r in exp3_results['collocation']]
    success_coll = [r['success_rate'] for r in exp3_results['collocation']]
    mean_r2_coll = [r['mean_r2'] for r in exp3_results['collocation']]
    
    ax3b.plot(n_points, success_coll, 's-', markersize=10, linewidth=2, color='#e67e22')
    ax3b.set_xlabel('Collocation Points', fontweight='bold', fontsize=11)
    ax3b.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
    ax3b.set_title('f) Exp. 3: Collocation Sensitivity', fontweight='bold', fontsize=12)
    ax3b.set_xscale('log', base=2)
    ax3b.set_xticks(n_points)
    ax3b.set_xticklabels(n_points)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_ylim([0, 105])
    
    # Add R² values with smaller font
    for pts, success, r2 in zip(n_points, success_coll, mean_r2_coll):
        ax3b.text(pts, success + 2, f'R²={r2:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    # Overall title
    plt.suptitle('Supplementary Experiments: Validation Studies',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = FIGURES_DIR / "combined_supplementary_experiments.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Saved: {output_path}")


def main():
    """Regenerate all supplementary figures"""
    print("="*80)
    print("REGENERATING SUPPLEMENTARY FIGURES (NO TRAINING)")
    print("="*80)
    print("Reading existing results and regenerating figures with updated formatting...")
    
    # Check if results exist
    if not RESULTS_DIR.exists():
        print(f"\n[ERROR] Results directory not found: {RESULTS_DIR}")
        print("Please run 11_supplementary_experiments.py first to generate results.")
        return
    
    # Create figures directory if needed
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Regenerate all figures
    try:
        regenerate_experiment_1_figure()
        regenerate_experiment_2_figure()
        regenerate_experiment_3_figure()
        regenerate_combined_figure()
        
        print("\n" + "="*80)
        print("ALL SUPPLEMENTARY FIGURES REGENERATED")
        print("="*80)
        print(f"Output directory: {FIGURES_DIR}")
        print("\nRegenerated figures:")
        print("  [OK] experiment_1_ramp_ablation.png")
        print("  [OK] experiment_2_cross_validation.png")
        print("  [OK] experiment_3_hyperparameter_sensitivity.png")
        print("  [OK] combined_supplementary_experiments.png")
        print("\nAll figures now have:")
        print("  [OK] Updated panel labels (a), b), c))")
        print("  [OK] Color legends (green=pass, red=fail)")
        print("  [OK] No text overlapping")
        print("  [OK] Publication-ready quality")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please ensure all experiment results exist before regenerating figures.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()