"""
Generate Figure 4: Comprehensive Residual Diagnostics
For IEEE Access manuscript on PINN Glucocorticoid-Renin modeling

Creates a 4-panel publication-quality figure showing:
- Panel A: Residuals vs. Fitted Values (homoscedasticity)
- Panel B: Q-Q Plot (normality assessment)
- Panel C: Residuals vs. Dose (systematic patterns)
- Panel D: Histogram with Normal Distribution overlay

Data from Table IV and Results section of manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# ============================================================================
# DATA FROM MANUSCRIPT
# ============================================================================

# Experimental data (Table IV, lines 657-710)
dex_doses = np.array([0, 0.3, 3.0, 30.0])  # mg/dl
observed_renin = np.array([1.000, 0.915, 0.847, 0.915])  # Normalized
experimental_std = np.array([0.074, 0.063, 0.043, 0.211])  # From IQR/1.35

# Predicted values (from Results section, lines 916-923)
predicted_renin = np.array([0.976, 0.938, 0.880, 0.907])

# Residuals (lines 953-954)
residuals = observed_renin - predicted_renin  # [+0.024, -0.024, -0.033, +0.008]

# Standardized residuals (lines 955-956)
std_errors = experimental_std / np.sqrt(9)  # Standard error (n=9 replicates)
standardized_residuals = residuals / std_errors
# Expected: [+1.05, -1.01, -1.44, +0.32] approximately

# Statistical test results (from Table VII, lines 959-1006)
shapiro_w = 0.931
shapiro_p = 0.601
jb_statistic = 0.478
jb_p = 0.787
dw_statistic = 1.752

# Summary statistics (Table IX, lines 1015-1059)
residual_mean = -0.0063
residual_std = 0.0233
residual_min = -0.0335
residual_max = +0.0243
residual_median = -0.0080
skewness = +0.121
kurtosis = -1.677

# ============================================================================
# CREATE FIGURE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comprehensive Residual Diagnostics for PINN Model Fit', 
             fontsize=14, fontweight='bold', y=0.98)

# ----------------------------------------------------------------------------
# PANEL A: Residuals vs. Fitted Values (Homoscedasticity Check)
# ----------------------------------------------------------------------------
ax1 = axes[0, 0]

# Scatter plot
ax1.scatter(predicted_renin, residuals, s=120, alpha=0.7, 
           edgecolors='black', linewidth=1.5, zorder=3)

# Zero line
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label='Zero residual', zorder=2)

# ±2 std error bands
ax1.axhline(y=2*residual_std, color='orange', linestyle=':', linewidth=1.5, 
           alpha=0.5, label='±2σ bands')
ax1.axhline(y=-2*residual_std, color='orange', linestyle=':', linewidth=1.5, 
           alpha=0.5)

# Trend line (LOWESS-like)
z = np.polyfit(predicted_renin, residuals, 1)
p = np.poly1d(z)
x_trend = np.linspace(predicted_renin.min(), predicted_renin.max(), 100)
ax1.plot(x_trend, p(x_trend), 'g-', alpha=0.6, linewidth=2, 
        label=f'Trend (slope={z[0]:.3f})')

# Labels and formatting
ax1.set_xlabel('Fitted Values (Predicted Renin)', fontsize=12)
ax1.set_ylabel('Residuals (Observed - Predicted)', fontsize=12)
ax1.set_title('a) Residuals vs. Fitted Values', fontsize=13, fontweight='bold',
             pad=10)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add text with mean residual
textstr = f'Mean residual: {residual_mean:.4f}\nStd: {residual_std:.4f}'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ----------------------------------------------------------------------------
# PANEL B: Q-Q Plot (Normality Assessment)
# ----------------------------------------------------------------------------
ax2 = axes[0, 1]

# Calculate theoretical quantiles
theoretical_quantiles = stats.norm.ppf(np.linspace(0.1, 0.9, len(residuals)))
sample_quantiles = np.sort(residuals)

# Q-Q plot
ax2.scatter(theoretical_quantiles, sample_quantiles, s=120, alpha=0.7,
           edgecolors='black', linewidth=1.5, zorder=3)

# 45-degree reference line
qq_min = min(theoretical_quantiles.min(), sample_quantiles.min())
qq_max = max(theoretical_quantiles.max(), sample_quantiles.max())
ax2.plot([qq_min, qq_max], [qq_min, qq_max], 'r--', linewidth=2, 
        alpha=0.7, label='Perfect normality', zorder=2)

# Labels and formatting
ax2.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
ax2.set_ylabel('Sample Quantiles (Residuals)', fontsize=12)
ax2.set_title('b) Q-Q Plot for Normality', fontsize=13, fontweight='bold',
             pad=10)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add statistical test results
textstr = f'Shapiro-Wilk:\nW = {shapiro_w:.3f}\np = {shapiro_p:.3f}\n\n'
textstr += f'Jarque-Bera:\nJB = {jb_statistic:.3f}\np = {jb_p:.3f}'
ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes,
        verticalalignment='bottom', horizontalalignment='right',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ----------------------------------------------------------------------------
# PANEL C: Residuals vs. Dose (Systematic Pattern Check)
# ----------------------------------------------------------------------------
ax3 = axes[1, 0]

# Create x-axis with proper spacing for dose levels
x_positions = [0, 1, 2, 3]
x_labels = ['0\n(Control)', '0.3\n(Low)', '3.0\n(Medium)', '30.0\n(High)']

# Bar plot with color coding
colors = ['green' if abs(r) < 2*residual_std else 'orange' for r in residuals]
bars = ax3.bar(x_positions, residuals, width=0.6, color=colors, 
              alpha=0.7, edgecolor='black', linewidth=1.5)

# Zero line
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)

# ±2 std bands
ax3.axhline(y=2*residual_std, color='orange', linestyle=':', linewidth=1.5, 
           alpha=0.5)
ax3.axhline(y=-2*residual_std, color='orange', linestyle=':', linewidth=1.5, 
           alpha=0.5)

# Labels and formatting
ax3.set_xlabel('Dexamethasone Dose (mg/dl)', fontsize=12)
ax3.set_ylabel('Residuals', fontsize=12)
ax3.set_title('c) Residuals vs. Dose Level', fontsize=13, fontweight='bold',
             pad=10)
ax3.set_xticks(x_positions)
ax3.set_xticklabels(x_labels)
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

# Add runs test result
textstr = f'Durbin-Watson: {dw_statistic:.3f}\n(No autocorrelation:\n1.5 < DW < 2.5)'
ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
        verticalalignment='top', horizontalalignment='right',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ----------------------------------------------------------------------------
# PANEL D: Histogram with Normal Distribution Overlay
# ----------------------------------------------------------------------------
ax4 = axes[1, 1]

# Histogram of standardized residuals
n, bins, patches = ax4.hist(standardized_residuals, bins=8, 
                            density=True, alpha=0.6, color='skyblue',
                            edgecolor='black', linewidth=1.5,
                            label='Standardized residuals')

# Overlay theoretical normal distribution
x_normal = np.linspace(-2, 2, 100)
y_normal = stats.norm.pdf(x_normal, 0, 1)
ax4.plot(x_normal, y_normal, 'r-', linewidth=2.5, 
        label='Standard Normal N(0,1)')

# Critical thresholds at ±2
ax4.axvline(x=-2, color='orange', linestyle='--', linewidth=1.5, 
           alpha=0.7, label='±2σ thresholds')
ax4.axvline(x=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

# Labels and formatting
ax4.set_xlabel('Standardized Residuals', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('d) Distribution of Standardized Residuals',
             fontsize=13, fontweight='bold', pad=10)
ax4.legend(loc='lower center', framealpha=0.9, fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

# Add distribution statistics
textstr = f'Skewness: {skewness:+.3f}\nKurtosis: {kurtosis:.3f}\n'
textstr += f'(Normal: 0, 0)'
ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes,
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Add count of outliers
n_outliers = np.sum(np.abs(standardized_residuals) > 2)
textstr_outliers = f'Outliers (|z|>2): {n_outliers}/4'
ax4.text(0.95, 0.95, textstr_outliers, transform=ax4.transAxes,
        verticalalignment='top', horizontalalignment='right',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# ============================================================================
# SAVE FIGURE
# ============================================================================

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/Figure4_residual_diagnostics.png', dpi=300, 
           bbox_inches='tight', facecolor='white')
print("\n" + "="*70)
print("Figure 4 generated successfully!")
print("Saved to: results/Figure4_residual_diagnostics.png")
print("="*70)

# Print summary statistics for verification
print("\nSummary Statistics:")
print(f"  Residuals: {residuals}")
print(f"  Mean: {residual_mean:.4f}")
print(f"  Std: {residual_std:.4f}")
print(f"  Standardized: {standardized_residuals}")
print(f"\nStatistical Tests:")
print(f"  Shapiro-Wilk: W={shapiro_w:.3f}, p={shapiro_p:.3f} {'✓ Normal' if shapiro_p > 0.05 else '✗ Not normal'}")
print(f"  Jarque-Bera: JB={jb_statistic:.3f}, p={jb_p:.3f} {'✓ Normal' if jb_p > 0.05 else '✗ Not normal'}")
print(f"  Durbin-Watson: DW={dw_statistic:.3f} {'✓ No autocorr' if 1.5 < dw_statistic < 2.5 else '✗ Autocorrelation'}")
print(f"  Outliers (|z|>2): {n_outliers}/4")
print("="*70 + "\n")

plt.show()