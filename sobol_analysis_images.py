import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.gridspec import GridSpec

# Load the Sobol results from your comprehensive_validation.json
with open('results/comprehensive_validation.json', 'r') as f:
    data = json.load(f)

sobol_indices = data['validation_results']['sensitivity']['sobol_results']

# Create figure with GridSpec for better control
fig = plt.figure(figsize=(10, 23))
gs = GridSpec(3, 1, figure=fig, hspace=0.5, top=0.97, bottom=0.08)

axes = [fig.add_subplot(gs[i]) for i in range(3)]

# Panel a) Total-order indices bar chart
param_names = sobol_indices['param_names']
st_values = sobol_indices['ST']
sorted_idx = sobol_indices['sorted_indices']

# Color code: red for IC50/Hill, blue for others
colors = ['red' if name in ['log_IC50', 'log_hill'] else 'steelblue' 
          for name in [param_names[i] for i in sorted_idx]]

axes[0].barh(range(len(param_names)), 
             [st_values[i] for i in sorted_idx], 
             color=colors, alpha=0.7, edgecolor='black')
axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, 
                label='Expected dominance threshold (>0.5)')
axes[0].set_yticks(range(len(param_names)))
axes[0].set_yticklabels([param_names[i].replace('log_', '') for i in sorted_idx], fontsize=10)
axes[0].set_xlabel('Total-Order Sobol Index (ST)', fontsize=11, fontweight='bold')
axes[0].set_title('a) Parameter Sensitivities: Failed IC50/Hill Dominance', 
                  fontsize=12, fontweight='bold', loc='left', pad=10)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(axis='x', alpha=0.3)
axes[0].set_xlim(0, max(st_values) * 1.1)

# Add text annotation for IC50+Hill combined
ic50_hill_combined = st_values[param_names.index('log_IC50')] + st_values[param_names.index('log_hill')]
axes[0].text(0.35, 9.5, f'IC50 + Hill = {ic50_hill_combined:.3f} (3.2%)\nExpected: >0.5 (50%)', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, verticalalignment='top')

# Panel b) S1 vs ST for top 5
s1_values = sobol_indices['S1']
top5_idx = sorted_idx[:5]
x = np.arange(5)
width = 0.35

bars1 = axes[1].bar(x - width/2, [s1_values[i] for i in top5_idx], 
                     width, label='First-order (S1)', color='lightcoral', edgecolor='black')
bars2 = axes[1].bar(x + width/2, [st_values[i] for i in top5_idx], 
                     width, label='Total-order (ST)', color='steelblue', edgecolor='black')

axes[1].set_xticks(x)
axes[1].set_xticklabels([param_names[i].replace('log_', '') for i in top5_idx], 
                        rotation=45, ha='right', fontsize=10)
axes[1].set_ylabel('Sobol Index', fontsize=11, fontweight='bold')
axes[1].set_title('b) Top 5 Parameters: Direct vs Total Effects', 
                  fontsize=12, fontweight='bold', loc='left', pad=10)
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim(0, max(st_values[i] for i in top5_idx) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)

# Panel c) S2 interaction matrix
s2_matrix = np.array(sobol_indices['S2'])

# Create custom colormap where NaN is white
cmap = plt.cm.RdBu_r
cmap.set_bad(color='white')

im = axes[2].imshow(s2_matrix, cmap=cmap, aspect='auto', 
                     vmin=-0.05, vmax=0.05)
axes[2].set_xticks(range(len(param_names)))
axes[2].set_yticks(range(len(param_names)))
axes[2].set_xticklabels([name.replace('log_', '') for name in param_names], 
                        rotation=90, fontsize=8, ha='center')
axes[2].set_yticklabels([name.replace('log_', '') for name in param_names], 
                        fontsize=8)
axes[2].set_title('c) Second-Order Interactions: NaN Values Indicate Failed Estimation', 
                  fontsize=12, fontweight='bold', loc='left', pad=15)

# Make sure x-axis labels have enough room
axes[2].tick_params(axis='x', pad=5)

# Add colorbar
cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label('S2 Index', fontsize=10, fontweight='bold')

# Add text showing NaN percentage
mask = np.isnan(s2_matrix)
nan_percentage = np.sum(mask) / mask.size * 100
axes[2].text(0.98, 0.02, f'{nan_percentage:.1f}% NaN values', 
             transform=axes[2].transAxes,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=9, verticalalignment='bottom', horizontalalignment='right')

# Save with extra padding at bottom for rotated labels
plt.savefig('results/sobol_comprehensive_analysis.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.3)
print("Figure saved to: results/sobol_comprehensive_analysis.png")

# Also create the summary table
print("\n" + "="*80)
print("Table XV: Sobol Sensitivity Indices (Sorted by Total-Order Importance)")
print("="*80)
print(f"{'Parameter':<20} {'First-Order (S1)':<18} {'Total-Order (ST)':<18} {'Rank':<6}")
print("-"*80)

for rank, idx in enumerate(sorted_idx, 1):
    param = param_names[idx].replace('log_', '')
    s1 = s1_values[idx]
    st = st_values[idx]
    marker = " ***" if param in ['IC50', 'hill'] else ""
    print(f"{param:<20} {s1:>16.4f}   {st:>16.4f}   {rank:<6}{marker}")

print("-"*80)
ic50_idx = param_names.index('log_IC50')
hill_idx = param_names.index('log_hill')
print(f"\nIC50 + Hill combined:     {st_values[ic50_idx] + st_values[hill_idx]:.4f} (3.2%)")
print(f"Expected for dominance:   >0.5000 (>50%)")
print(f"Dominance test:           FAILED")
print("\n*** Parameters that were expected to dominate but showed weak sensitivity")
print("="*80)

plt.show()
