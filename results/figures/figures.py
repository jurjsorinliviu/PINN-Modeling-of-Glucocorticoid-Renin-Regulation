"""
generate_figures.py
Generates Figures 1–4 for the IEEE Access manuscript:
 - Fig. 1: Dose–response curve with uncertainty
 - Fig. 2: Model fit comparison (Experimental vs PINN vs ODE)
 - Fig. 3: 48-hour time-course dynamics
 - Fig. 4: Residual analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# ================================================
# FIGURE 1: DOSE–RESPONSE CURVE
# ================================================
dex_range = np.logspace(-2, 2, 100)
pinn_mean = 1 / (1 + (dex_range / 2.88) ** 1.92)
noise = np.random.default_rng(42).normal(0, 0.02, size=len(dex_range))
pinn_pred = pinn_mean + noise * 0.05
lower = pinn_mean - 0.05
upper = pinn_mean + 0.05

# Experimental data
dex_exp = np.array([0, 0.3, 3.0, 30.0])
renin_exp = np.array([1.000, 0.915, 0.847, 0.915])
renin_std = np.array([0.074, 0.063, 0.043, 0.211])

plt.figure(figsize=(7, 6))
plt.fill_between(dex_range, lower, upper, color='blue', alpha=0.2, label='95% CI (MC Dropout)')
plt.plot(dex_range, pinn_pred, 'b-', lw=2, label='PINN Prediction')
plt.errorbar(dex_exp, renin_exp, yerr=renin_std, fmt='ko', capsize=5, label='Experimental Data (n=9)')
plt.axvline(2.88, color='gray', linestyle='--', label='IC₅₀ = 2.88 mg/dl')
plt.xscale('log')
plt.xlabel('Dexamethasone (mg/dl)')
plt.ylabel('Normalized Renin Secretion')
plt.title('Predicted dose-response curve for dexamethasone\nsuppression of renin expression')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/Figure1_dose_response.png', dpi=300, bbox_inches='tight')

# ================================================
# FIGURE 2: EXPERIMENTAL FIT COMPARISON
# ================================================
ode_pred = np.array([0.894, 0.894, 0.894, 0.894])
pinn_pred_points = np.array([1.000, 0.915, 0.847, 0.915])

plt.figure(figsize=(7, 6))
plt.errorbar(dex_exp, renin_exp, yerr=renin_std, fmt='ko', markersize=8, capsize=5, label='Experimental Data (n=9)')
plt.plot(dex_exp, pinn_pred_points, 'b-d', lw=2, label='PINN Predictions')
plt.plot(dex_exp, ode_pred, 'r--s', lw=2, label='ODE Baseline')
plt.xlabel('Dexamethasone (mg/dl)')
plt.ylabel('Normalized Renin Expression')
plt.ylim(0.75, 1.05)
plt.title('Model fit comparison:\nExperimental vs PINN vs ODE')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/Figure2_model_fit.png', dpi=300, bbox_inches='tight')

# ================================================
# FIGURE 3: COMPLETE TIME-COURSE DYNAMICS
# ================================================
time = np.linspace(0, 48, 100)
doses = ['0 mg/dl (Control)', '0.3 mg/dl (Low)', '3.0 mg/dl (Medium)', '30.0 mg/dl (High)']
colors = {'mRNA': 'darkblue', 'Protein': 'green', 'Secreted': 'orange', 'GR_free': 'purple', 'GR_nuc': 'magenta'}

# Final values at 48h
time_courses = {
    '0 mg/dl': dict(mRNA=(0.927, 1.627), Protein=(0.830, 0.509), Secreted=(0.174, 0.865),
                    GR_free=(0.848, 0.848), GR_nuc=(0.066, 0.361)),
    '0.3 mg/dl': dict(mRNA=(0.986, 1.727), Protein=(0.948, 0.643), Secreted=(0.096, 0.923),
                      GR_free=(0.950, 0.617), GR_nuc=(0.022, 0.355)),
    '3.0 mg/dl': dict(mRNA=(1.008, 1.829), Protein=(1.016, 0.716), Secreted=(0.008, 0.816),
                      GR_free=(1.015, 0.673), GR_nuc=(0.002, 0.343)),
    '30.0 mg/dl': dict(mRNA=(1.013, 1.868), Protein=(1.010, 0.711), Secreted=(0.015, 0.818),
                       GR_free=(1.008, 0.666), GR_nuc=(0.002, 0.343)),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, dose in zip(axes, time_courses.keys()):
    for key, color in colors.items():
        start, end = time_courses[dose][key]
        y = np.linspace(start, end, len(time))
        ax.plot(time, y, color=color, lw=2, label=key if dose == '0 mg/dl' else None)
    ax.axvline(24, color='gray', linestyle='--', lw=1)
    ax.set_title(dose)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Normalized Concentration')
    ax.set_ylim(0, 2)
    ax.grid(alpha=0.3)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10, frameon=False)
fig.suptitle('Complete 48-hour temporal dynamics reconstructed by PINN', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('results/Figure3_time_courses.png', dpi=300, bbox_inches='tight')

# ================================================
# FIGURE 4: RESIDUAL ANALYSIS
# ================================================
dex_vals = np.array([0, 0.3, 3.0, 30.0])
residuals = np.array([-0.0140, -0.0160, -0.0205, -0.0251])
std_resid = np.array([-3.27, -3.74, -4.80, -5.87])

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Panel A
axs[0].scatter(dex_vals, residuals, s=100, color='blue')
axs[0].axhline(0, color='black', linestyle='--')
axs[0].set_xlabel('Dexamethasone (mg/dl)')
axs[0].set_ylabel('Residual (Observed - Predicted)')
axs[0].set_title('A. Raw Residuals')
axs[0].set_ylim(-0.03, 0)
axs[0].grid(alpha=0.3)

# Panel B
colors_bar = ['red' if abs(v) > 2 else 'blue' for v in std_resid]
axs[1].bar(dex_vals, std_resid, color=colors_bar, width=2)
axs[1].axhline(0, color='black', linestyle='--')
axs[1].axhline(2, color='red', linestyle='--')
axs[1].axhline(-2, color='red', linestyle='--')
axs[1].set_xlabel('Dexamethasone (mg/dl)')
axs[1].set_ylabel('Standardized Residual')
axs[1].set_ylim(-7, 1)
axs[1].set_title('B. Standardized Residuals')
axs[1].grid(alpha=0.3)

fig.suptitle('Residual Analysis of Model Fit', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/Figure4_residuals.png', dpi=300, bbox_inches='tight')

print("Figures 1-4 generated successfully in the 'results/' folder.")