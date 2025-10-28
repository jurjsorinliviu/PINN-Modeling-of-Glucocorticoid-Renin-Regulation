"""
Enhanced Visualization and Diagnostics for IEEE Access Submission

This module provides comprehensive plotting and diagnostic tools:
- Predicted vs. observed plots with diagonal lines
- Residual analysis (histograms, Q-Q plots, vs. dose)
- Parameter uncertainty distributions (histograms, joint plots)
- Ablation study comparison figures
- Time course simulations with confidence bands
- Dose-response extrapolation plots
- Multi-panel publication-ready figures

For IEEE Access submission requirements.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import os


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class EnhancedDiagnostics:
    """
    Comprehensive diagnostics and visualization suite
    """
    
    def __init__(self, output_dir: str = 'results/figures'):
        """
        Initialize diagnostics suite
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_predicted_vs_observed(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_std: Optional[np.ndarray] = None,
                                   labels: Optional[List[str]] = None,
                                   title: str = 'Predicted vs. Observed',
                                   filename: str = 'predicted_vs_observed.png') -> plt.Figure:
        """
        Create predicted vs. observed plot with diagonal line and error bars
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_std: Standard deviations (optional)
            labels: Point labels (optional)
            title: Plot title
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Diagonal line (perfect prediction)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.5, linewidth=1.5, label='Perfect prediction')
        
        # Scatter plot with error bars
        if y_std is not None:
            ax.errorbar(y_true, y_pred, yerr=y_std, fmt='o', 
                       capsize=5, capthick=1.5, markersize=8,
                       alpha=0.7, label='Predictions ± σ')
        else:
            ax.scatter(y_true, y_pred, s=100, alpha=0.7, 
                      edgecolors='black', linewidths=1.5)
        
        # Add labels if provided
        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (y_true[i], y_pred[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Calculate R²
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Add metrics text
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Observed Renin (normalized)')
        ax.set_ylabel('Predicted Renin (normalized)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residual_diagnostics(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  doses: np.ndarray,
                                  filename: str = 'residual_diagnostics.png') -> plt.Figure:
        """
        Create comprehensive residual diagnostic plots
        
        Args:
            y_true: True values
            y_pred: Predicted values
            doses: Dexamethasone doses
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure with 4 subplots
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs. Dose
        ax = axes[0, 0]
        ax.scatter(doses, residuals, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero line')
        
        # Add trend line
        z = np.polyfit(doses, residuals, 1)
        p = np.poly1d(z)
        dose_range = np.linspace(doses.min(), doses.max(), 100)
        ax.plot(dose_range, p(dose_range), "b-", alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_xlabel('Dexamethasone Dose (mg/dl)')
        ax.set_ylabel('Residuals (Observed - Predicted)')
        ax.set_title('Residuals vs. Dose')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Histogram of Residuals
        ax = axes[0, 1]
        ax.hist(residuals, bins=15, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'N({mu:.3f}, {sigma:.3f})')
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        ax = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # 4. Residuals vs. Fitted Values
        ax = axes[1, 1]
        ax.scatter(y_pred, residuals, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs. Fitted')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_parameter_uncertainty(self,
                                   param_samples: Dict[str, np.ndarray],
                                   filename: str = 'parameter_uncertainty.png') -> plt.Figure:
        """
        Plot parameter uncertainty distributions and joint plots
        
        Args:
            param_samples: Dictionary of parameter samples {name: array}
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure
        """
        # Focus on IC50 and Hill coefficient
        ic50_samples = param_samples.get('IC50', param_samples.get('log_IC50', None))
        hill_samples = param_samples.get('hill', param_samples.get('log_hill', None))
        
        if ic50_samples is None or hill_samples is None:
            raise ValueError("IC50 and hill parameters must be in param_samples")
        
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # 1. IC50 Histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(ic50_samples, bins=30, density=True, alpha=0.7, 
                edgecolor='black', color='steelblue')
        
        mean_ic50 = np.mean(ic50_samples)
        std_ic50 = np.std(ic50_samples)
        ci_lower = np.percentile(ic50_samples, 2.5)
        ci_upper = np.percentile(ic50_samples, 97.5)
        
        ax1.axvline(mean_ic50, color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, label='95% CI')
        ax1.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
        
        ax1.set_xlabel('IC₅₀ (mg/dl)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'IC₅₀ Distribution\n{mean_ic50:.2f} ± {std_ic50:.2f} mg/dl')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Hill Coefficient Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(hill_samples, bins=30, density=True, alpha=0.7,
                edgecolor='black', color='coral')
        
        mean_hill = np.mean(hill_samples)
        std_hill = np.std(hill_samples)
        ci_lower_h = np.percentile(hill_samples, 2.5)
        ci_upper_h = np.percentile(hill_samples, 97.5)
        
        ax2.axvline(mean_hill, color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(ci_lower_h, color='orange', linestyle=':', linewidth=2, label='95% CI')
        ax2.axvline(ci_upper_h, color='orange', linestyle=':', linewidth=2)
        
        ax2.set_xlabel('Hill Coefficient')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Hill Distribution\n{mean_hill:.2f} ± {std_hill:.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Joint Distribution (2D histogram)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # 2D histogram
        h = ax3.hist2d(ic50_samples, hill_samples, bins=30, cmap='Blues', alpha=0.8)
        plt.colorbar(h[3], ax=ax3, label='Count')
        
        # Add mean point
        ax3.scatter([mean_ic50], [mean_hill], color='red', s=200, 
                   marker='x', linewidths=3, label='Mean', zorder=10)
        
        # Add confidence ellipse
        from matplotlib.patches import Ellipse
        cov = np.cov(ic50_samples, hill_samples)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                         facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        
        scale_x = np.sqrt(cov[0, 0]) * 2.448  # 95% confidence
        scale_y = np.sqrt(cov[1, 1]) * 2.448
        
        transf = plt.matplotlib.transforms.Affine2D() \
            .scale(scale_x, scale_y) \
            .translate(mean_ic50, mean_hill)
        
        ellipse.set_transform(transf + ax3.transData)
        ax3.add_patch(ellipse)
        
        ax3.set_xlabel('IC₅₀ (mg/dl)')
        ax3.set_ylabel('Hill Coefficient')
        ax3.set_title(f'Joint Distribution\n(ρ = {pearson:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ablation_comparison(self,
                                 ablation_results: Dict,
                                 filename: str = 'ablation_comparison.png') -> plt.Figure:
        """
        Create ablation study comparison figure
        
        Args:
            ablation_results: Results from ablation study
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure
        """
        # Extract data
        config_names = []
        r2_means = []
        r2_stds = []
        rmse_means = []
        rmse_stds = []
        
        for config_name, result in ablation_results.items():
            config_names.append(config_name.replace('_', ' ').title())
            r2_means.append(result['r2']['mean'])
            r2_stds.append(result['r2']['std'])
            rmse_means.append(result['rmse']['mean'])
            rmse_stds.append(result['rmse']['std'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(config_names))
        width = 0.6
        
        # R² comparison
        bars1 = ax1.bar(x, r2_means, width, yerr=r2_stds, 
                       capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Color baseline differently
        bars1[0].set_color('green')
        bars1[0].set_alpha(0.9)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Performance Across Configurations')
        ax1.set_xticks(x)
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([min(r2_means) - 0.05, 1.0])
        
        # Add horizontal line for baseline
        ax1.axhline(y=r2_means[0], color='green', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Baseline')
        ax1.legend()
        
        # RMSE comparison
        bars2 = ax2.bar(x, rmse_means, width, yerr=rmse_stds,
                       capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        bars2[0].set_color('green')
        bars2[0].set_alpha(0.9)
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Across Configurations')
        ax2.set_xticks(x)
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax2.axhline(y=rmse_means[0], color='green', linestyle='--',
                   linewidth=2, alpha=0.5, label='Baseline')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_course_with_uncertainty(self,
                                         time_course_results: Dict,
                                         filename: str = 'time_courses_uncertainty.png') -> plt.Figure:
        """
        Plot time courses with confidence bands for multiple doses
        
        Args:
            time_course_results: Results from temporal validation
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure
        """
        time_points = time_course_results['time_points']
        doses = time_course_results['dex_doses']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for idx, dose in enumerate(doses[:4]):  # Plot first 4 doses
            ax = axes[idx]
            
            predictions = time_course_results['predictions'][dose]
            uncertainties = time_course_results['uncertainties'][dose]
            
            # Extract renin (state variable 2)
            if predictions.ndim > 1:
                renin_mean = predictions[:, 2]
                renin_std = uncertainties[:, 2]
            else:
                renin_mean = predictions
                renin_std = uncertainties
            
            # Plot mean trajectory
            ax.plot(time_points, renin_mean, color=colors[idx], 
                   linewidth=2.5, label='Mean prediction')
            
            # Plot confidence band (2σ)
            ax.fill_between(time_points,
                           renin_mean - 2*renin_std,
                           renin_mean + 2*renin_std,
                           alpha=0.3, color=colors[idx], label='95% CI')
            
            # Mark experimental time point
            ax.axvline(x=24, color='gray', linestyle='--', 
                      linewidth=1.5, alpha=0.5, label='Measurement (24h)')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Normalized Renin')
            ax.set_title(f'Dexamethasone: {dose} mg/dl')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 48])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dose_response_extrapolation(self,
                                        dose_response: Dict,
                                        training_doses: np.ndarray,
                                        filename: str = 'dose_response_extrapolation.png') -> plt.Figure:
        """
        Plot dose-response curve with extrapolation regions highlighted
        
        Args:
            dose_response: Results from dose-response analysis
            training_doses: Training dose range
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure
        """
        doses = dose_response['doses']
        renin_mean = dose_response['renin_mean']
        ci_lower = dose_response['ci_lower']
        ci_upper = dose_response['ci_upper']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Identify interpolation and extrapolation regions
        min_train = training_doses.min()
        max_train = training_doses.max()
        
        interp_mask = (doses >= min_train) & (doses <= max_train)
        extrap_low = doses < min_train
        extrap_high = doses > max_train
        
        # Plot interpolation region
        if np.any(interp_mask):
            ax.plot(doses[interp_mask], renin_mean[interp_mask], 
                   'b-', linewidth=2.5, label='Interpolation')
            ax.fill_between(doses[interp_mask],
                           ci_lower[interp_mask],
                           ci_upper[interp_mask],
                           alpha=0.3, color='blue')
        
        # Plot extrapolation regions
        if np.any(extrap_low):
            ax.plot(doses[extrap_low], renin_mean[extrap_low],
                   'r--', linewidth=2.5, label='Extrapolation (low)')
            ax.fill_between(doses[extrap_low],
                           ci_lower[extrap_low],
                           ci_upper[extrap_low],
                           alpha=0.2, color='red')
        
        if np.any(extrap_high):
            ax.plot(doses[extrap_high], renin_mean[extrap_high],
                   'r--', linewidth=2.5, label='Extrapolation (high)')
            ax.fill_between(doses[extrap_high],
                           ci_lower[extrap_high],
                           ci_upper[extrap_high],
                           alpha=0.2, color='red')
        
        # Mark training doses
        ax.scatter(training_doses, 
                  [renin_mean[np.argmin(np.abs(doses - d))] for d in training_doses],
                  s=150, c='black', marker='o', edgecolors='white', 
                  linewidths=2, zorder=10, label='Training data')
        
        # Vertical lines for training range
        ax.axvline(min_train, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(max_train, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Dexamethasone Dose (mg/dl)')
        ax.set_ylabel('Normalized Renin')
        ax.set_title('Dose-Response Curve with Extrapolation')
        ax.set_xscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_figure(self,
                                   data_dict: Dict,
                                   model_predictions: Dict,
                                   temporal_results: Dict,
                                   dose_response: Dict,
                                   residual_data: Dict,
                                   filename: str = 'comprehensive_analysis.png') -> plt.Figure:
        """
        Create multi-panel comprehensive figure for manuscript
        
        Args:
            data_dict: Experimental data
            model_predictions: Model prediction results
            temporal_results: Temporal validation results
            dose_response: Dose-response analysis
            residual_data: Residual analysis data
            filename: Output filename
            
        Returns:
            fig: Matplotlib figure with 6 panels
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # Panel A: Predicted vs. Observed
        ax_a = fig.add_subplot(gs[0, 0])
        y_true = data_dict['renin_normalized']
        y_pred = model_predictions['renin_pred']
        
        ax_a.scatter(y_true, y_pred, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax_a.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5)
        ax_a.set_xlabel('Observed')
        ax_a.set_ylabel('Predicted')
        ax_a.set_title('(A) Model Fit')
        ax_a.grid(True, alpha=0.3)
        ax_a.text(0.05, 0.95, f"R² = {model_predictions['r2']:.3f}",
                 transform=ax_a.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel B: Residuals vs. Dose
        ax_b = fig.add_subplot(gs[0, 1])
        residuals = residual_data['residuals']
        doses = residual_data['doses']
        
        ax_b.scatter(doses, residuals, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax_b.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_b.set_xlabel('Dose (mg/dl)')
        ax_b.set_ylabel('Residuals')
        ax_b.set_title('(B) Residual Analysis')
        ax_b.grid(True, alpha=0.3)
        
        # Panel C: Q-Q Plot
        ax_c = fig.add_subplot(gs[0, 2])
        stats.probplot(residuals, dist="norm", plot=ax_c)
        ax_c.set_title('(C) Q-Q Plot')
        ax_c.grid(True, alpha=0.3)
        
        # Panel D: Time Courses
        ax_d = fig.add_subplot(gs[1, :2])
        time_points = temporal_results['time_points']
        for dose in temporal_results['dex_doses'][:3]:
            predictions = temporal_results['predictions'][dose]
            renin = predictions[:, 2] if predictions.ndim > 1 else predictions
            ax_d.plot(time_points, renin, linewidth=2.5, label=f'{dose} mg/dl')
        
        ax_d.axvline(x=24, color='gray', linestyle='--', alpha=0.5)
        ax_d.set_xlabel('Time (hours)')
        ax_d.set_ylabel('Normalized Renin')
        ax_d.set_title('(D) Temporal Dynamics')
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
        
        # Panel E: Dose-Response
        ax_e = fig.add_subplot(gs[1, 2])
        doses_dr = dose_response['doses']
        renin_mean = dose_response['renin_mean']
        ci_lower = dose_response['ci_lower']
        ci_upper = dose_response['ci_upper']
        
        ax_e.plot(doses_dr, renin_mean, 'b-', linewidth=2.5)
        ax_e.fill_between(doses_dr, ci_lower, ci_upper, alpha=0.3)
        ax_e.set_xlabel('Dose (mg/dl)')
        ax_e.set_ylabel('Normalized Renin')
        ax_e.set_title('(E) Dose-Response Curve')
        ax_e.set_xscale('log')
        ax_e.grid(True, alpha=0.3)
        
        # Panel F: Parameter Distributions
        ax_f = fig.add_subplot(gs[2, :])
        param_names = list(model_predictions.get('param_uncertainty', {}).keys())[:6]
        param_values = [model_predictions['param_uncertainty'][p]['mean'] 
                       for p in param_names]
        param_stds = [model_predictions['param_uncertainty'][p]['std'] 
                     for p in param_names]
        
        x = np.arange(len(param_names))
        ax_f.bar(x, param_values, yerr=param_stds, capsize=5, alpha=0.8,
                edgecolor='black', linewidth=1.5)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(param_names, rotation=45, ha='right')
        ax_f.set_ylabel('Parameter Value')
        ax_f.set_title('(F) Learned Parameters with Uncertainty')
        ax_f.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    print("Enhanced Visualization Module")
    print("=" * 60)
    print("This module provides publication-quality plots:")
    print("  - Predicted vs. observed with error bars")
    print("  - Comprehensive residual diagnostics")
    print("  - Parameter uncertainty visualization")
    print("  - Ablation study comparisons")
    print("  - Time course simulations with CI")
    print("  - Dose-response extrapolation")
    print("\nUsage:")
    print("  from src.enhanced_visualization import EnhancedDiagnostics")
    print("  diag = EnhancedDiagnostics()")
    print("  fig = diag.plot_predicted_vs_observed(y_true, y_pred)")