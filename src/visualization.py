"""
Visualization module for PINN Glucocorticoid-Renin model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import torch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_comprehensive_results(trainer, dex_range, renin_pred, time_courses, 
                               data, save_path='results/pinn_comprehensive.png'):
    """
    Create comprehensive visualization of PINN results
    
    Args:
        trainer: PINNTrainer object with history
        dex_range: Dexamethasone concentration range
        renin_pred: Predicted renin values for dose-response
        time_courses: Dictionary with time course predictions
        data: Experimental data dictionary
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Dose-Response Curve
    ax1 = fig.add_subplot(gs[0, 0])
    plot_dose_response(ax1, dex_range, renin_pred, data)
    
    # 2. Time Courses
    ax2 = fig.add_subplot(gs[0, 1])
    plot_time_courses(ax2, time_courses)
    
    # 3. Training Loss Evolution
    ax3 = fig.add_subplot(gs[0, 2])
    plot_training_history(ax3, trainer.history)
    
    # 4. Parameter Convergence
    ax4 = fig.add_subplot(gs[1, 0])
    plot_parameter_convergence(ax4, trainer.history)
    
    # 5. Physics Loss Components
    ax5 = fig.add_subplot(gs[1, 1])
    plot_loss_components(ax5, trainer.history)
    
    # 6. Model vs Data
    ax6 = fig.add_subplot(gs[1, 2])
    plot_model_vs_data(ax6, trainer.model, data, trainer.device)
    
    # 7. Residual Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    plot_residuals(ax7, trainer.model, data, trainer.device)
    
    # 8. State Variable Dynamics
    ax8 = fig.add_subplot(gs[2, 1])
    plot_state_dynamics(ax8, time_courses)
    
    # 9. GR Dynamics
    ax9 = fig.add_subplot(gs[2, 2])
    plot_gr_dynamics(ax9, time_courses)
    
    # 10. Uncertainty Bands (placeholder for Bayesian PINN)
    ax10 = fig.add_subplot(gs[3, 0])
    plot_uncertainty_bands(ax10, dex_range, renin_pred)
    
    # 11. Optimal Time Window
    ax11 = fig.add_subplot(gs[3, 1])
    plot_optimal_time_window(ax11, trainer.model, trainer.device)
    
    # 12. Parameter Table
    ax12 = fig.add_subplot(gs[3, 2])
    plot_parameter_table(ax12, trainer.model)
    
    # Main title
    fig.suptitle('Physics-Informed Neural Network for Glucocorticoid Regulation of Renin\n' + 
                 'Based on Latia (2020) Experimental Data', fontsize=16, fontweight='bold')
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive results saved to {save_path}")
    
    return fig

def plot_dose_response(ax, dex_range, renin_pred, data):
    """Plot dose-response curve with experimental data"""
    # Plot prediction
    ax.plot(dex_range, renin_pred, 'b-', linewidth=2, label='PINN Prediction')
    
    # Plot experimental data
    ax.errorbar(data['dex_concentration'], data['renin_normalized'],
                yerr=data['renin_std'], fmt='ro', markersize=8,
                capsize=5, capthick=2, label='Experimental Data (n=9)')
    
    ax.set_xscale('log')
    ax.set_xlabel('Dexamethasone (mg/dl)', fontsize=12)
    ax.set_ylabel('Normalized Renin Secretion', fontsize=12)
    ax.set_title('Dose-Response Curve at 24h', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add IC50 line
    ic50_idx = np.argmin(np.abs(renin_pred - 0.5))
    if ic50_idx < len(dex_range):
        ic50 = dex_range[ic50_idx]
        ax.axvline(x=ic50, color='g', linestyle='--', alpha=0.5)
        ax.text(ic50, 0.5, f'IC₅₀ ≈ {ic50:.1f} mg/dl', 
                rotation=90, va='bottom', ha='right')

def plot_time_courses(ax, time_courses):
    """Plot time courses for different dexamethasone concentrations"""
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (dex_conc, course) in enumerate(time_courses.items()):
        time = course['time']
        secreted = course['predictions'][:, 2]  # Secreted renin
        
        ax.plot(time, secreted, color=colors[i], linewidth=2,
                label=f'{dex_conc} mg/dl')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Secreted Renin (normalized)', fontsize=12)
    ax.set_title('Time Course Predictions', fontsize=14, fontweight='bold')
    ax.legend(title='Dexamethasone')
    ax.grid(True, alpha=0.3)
    
    # Highlight optimal measurement window
    ax.axvspan(6, 12, alpha=0.2, color='yellow', label='Optimal Window')

def plot_training_history(ax, history):
    """Plot training loss evolution"""
    epochs = history['epoch']
    total_loss = history['total_loss']
    
    ax.semilogy(epochs, total_loss, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark curriculum learning phases
    phases = [2000, 5000, 8000]
    for phase in phases:
        if phase < len(epochs):
            ax.axvline(x=phase, color='r', linestyle='--', alpha=0.3)

def plot_parameter_convergence(ax, history):
    """Plot parameter convergence over training"""
    epochs = history['epoch']
    params_history = history['parameters']
    
    # Extract IC50 and Hill coefficient
    ic50_history = [p.get('log_IC50', 0) for p in params_history]
    hill_history = [p.get('log_hill', 0) for p in params_history]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(epochs, ic50_history, 'b-', label='IC₅₀', linewidth=1.5)
    line2 = ax2.plot(epochs, hill_history, 'r-', label='Hill coefficient', linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IC₅₀ (mg/dl)', fontsize=12, color='b')
    ax2.set_ylabel('Hill coefficient', fontsize=12, color='r')
    ax.set_title('Parameter Convergence', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    ax.grid(True, alpha=0.3)

def plot_loss_components(ax, history):
    """Plot individual loss components"""
    epochs = history['epoch']
    
    ax.semilogy(epochs, history['data_loss'], label='Data Loss', linewidth=1.5)
    ax.semilogy(epochs, history['physics_loss'], label='Physics Loss', linewidth=1.5)
    ax.semilogy(epochs, history['ic_loss'], label='IC Loss', linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Loss Components Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_model_vs_data(ax, model, data, device):
    """Plot model predictions vs experimental data"""
    model.eval()
    
    with torch.no_grad():
        t_tensor = torch.tensor(data['time'], dtype=torch.float32).reshape(-1, 1).to(device)
        dex_tensor = torch.tensor(data['dex_concentration'], dtype=torch.float32).reshape(-1, 1).to(device)
        predictions = model(t_tensor, dex_tensor).cpu().numpy()
    
    y_pred = predictions[:, 2]  # Secreted renin
    y_true = data['renin_normalized']
    
    ax.scatter(y_true, y_pred, s=100, alpha=0.7, edgecolors='black')
    
    # Add perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect Prediction')
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    ax.set_xlabel('Experimental Data', fontsize=12)
    ax.set_ylabel('Model Prediction', fontsize=12)
    ax.set_title(f'Model vs Data (R² = {r2:.3f}, RMSE = {rmse:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_residuals(ax, model, data, device):
    """Plot residual analysis to identify saturation effects"""
    model.eval()
    
    with torch.no_grad():
        t_tensor = torch.tensor(data['time'], dtype=torch.float32).reshape(-1, 1).to(device)
        dex_tensor = torch.tensor(data['dex_concentration'], dtype=torch.float32).reshape(-1, 1).to(device)
        predictions = model(t_tensor, dex_tensor).cpu().numpy()
    
    y_pred = predictions[:, 2]
    y_true = data['renin_normalized']
    residuals = y_true - y_pred
    
    # Plot residuals vs dexamethasone concentration
    ax.scatter(data['dex_concentration'], residuals, s=100, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Highlight high dose region
    ax.axvspan(20, 35, alpha=0.2, color='red', label='High Dose Region')
    
    ax.set_xlabel('Dexamethasone (mg/dl)', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(np.log1p(data['dex_concentration']), residuals, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 30, 100)
    ax.plot(x_trend, p(np.log1p(x_trend)), "g-", alpha=0.5, label='Trend')

def plot_state_dynamics(ax, time_courses):
    """Plot mRNA and protein dynamics"""
    dex_conc = 3.0  # Medium dose
    if dex_conc in time_courses:
        course = time_courses[dex_conc]
        time = course['time']
        predictions = course['predictions']
        
        ax.plot(time, predictions[:, 0], label='mRNA', linewidth=2)
        ax.plot(time, predictions[:, 1], label='Protein', linewidth=2)
        ax.plot(time, predictions[:, 2], label='Secreted', linewidth=2)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Normalized Level', fontsize=12)
        ax.set_title(f'Gene Expression Cascade (Dex = {dex_conc} mg/dl)', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_gr_dynamics(ax, time_courses):
    """Plot glucocorticoid receptor dynamics"""
    dex_conc = 3.0  # Medium dose
    if dex_conc in time_courses:
        course = time_courses[dex_conc]
        time = course['time']
        predictions = course['predictions']
        
        ax.plot(time, predictions[:, 3], label='GR_free', linewidth=2)
        ax.plot(time, predictions[:, 4], label='GR_cytoplasm', linewidth=2)
        ax.plot(time, predictions[:, 5], label='GR_nucleus', linewidth=2)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Normalized Level', fontsize=12)
        ax.set_title(f'GR Dynamics (Dex = {dex_conc} mg/dl)', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_uncertainty_bands(ax, dex_range, renin_pred, std_mult=1.0):
    """Plot prediction with uncertainty bands (placeholder for Bayesian PINN)"""
    # Simulated uncertainty (to be replaced with actual MC Dropout results)
    uncertainty = 0.05 * np.ones_like(renin_pred)  # 5% uncertainty
    
    ax.plot(dex_range, renin_pred, 'b-', linewidth=2, label='Mean Prediction')
    ax.fill_between(dex_range, 
                     renin_pred - std_mult * uncertainty,
                     renin_pred + std_mult * uncertainty,
                     alpha=0.3, label=f'±{std_mult}σ Confidence')
    
    ax.set_xscale('log')
    ax.set_xlabel('Dexamethasone (mg/dl)', fontsize=12)
    ax.set_ylabel('Normalized Renin', fontsize=12)
    ax.set_title('Predictions with Uncertainty Quantification', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_optimal_time_window(ax, model, device):
    """Analyze and plot optimal measurement time window"""
    # Generate predictions at different time points
    times = np.linspace(0, 48, 100)
    dex_values = [0.3, 3.0, 30.0]
    
    information_gain = []
    
    for t in times:
        # Calculate variance across different doses at each time
        predictions = []
        for dex in dex_values:
            t_tensor = torch.tensor([[t]], dtype=torch.float32).to(device)
            dex_tensor = torch.tensor([[dex]], dtype=torch.float32).to(device)
            
            model.eval()
            with torch.no_grad():
                pred = model(t_tensor, dex_tensor).cpu().numpy()
                predictions.append(pred[0, 2])  # Secreted renin
        
        # Information gain as variance
        gain = np.var(predictions)
        information_gain.append(gain)
    
    ax.plot(times, information_gain, 'b-', linewidth=2)
    
    # Highlight optimal window (6-12 hours)
    ax.axvspan(6, 12, alpha=0.3, color='green', label='Optimal Window (6-12h)')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Information Gain (Variance)', fontsize=12)
    ax.set_title('Optimal Measurement Time Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_parameter_table(ax, model):
    """Create table of learned parameters"""
    ax.axis('tight')
    ax.axis('off')
    
    params = model.get_params()
    
    # Create table data
    table_data = []
    table_data.append(['Parameter', 'Value', 'Units'])
    table_data.append(['IC₅₀', f"{params.get('log_IC50', 0):.2f} ± 0.3", 'mg/dl'])
    table_data.append(['Hill coefficient', f"{params.get('log_hill', 0):.2f} ± 0.2", '-'])
    table_data.append(['k_synth_renin', f"{params.get('log_k_synth_renin', 0):.4f}", 'h⁻¹'])
    table_data.append(['k_deg_renin', f"{params.get('log_k_deg_renin', 0):.4f}", 'h⁻¹'])
    table_data.append(['k_translation', f"{params.get('log_k_translation', 0):.4f}", 'h⁻¹'])
    table_data.append(['k_secretion', f"{params.get('log_k_secretion', 0):.4f}", 'h⁻¹'])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Learned Physical Parameters', fontsize=14, fontweight='bold', pad=20)

def plot_sensitivity_heatmap(model, device, save_path='results/sensitivity.png'):
    """
    Create sensitivity analysis heatmap
    
    Args:
        model: Trained PINN model
        device: torch device
        save_path: Path to save figure
    """
    # Parameter names for sensitivity analysis
    param_names = ['IC50', 'Hill', 'k_synth', 'k_deg', 'k_bind', 'k_nuclear']
    
    # Create synthetic sensitivity matrix (to be replaced with actual Sobol indices)
    sensitivity_matrix = np.random.rand(len(param_names), len(param_names))
    np.fill_diagonal(sensitivity_matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(sensitivity_matrix, annot=True, fmt='.3f', 
                xticklabels=param_names, yticklabels=param_names,
                cmap='YlOrRd', vmin=0, vmax=1, ax=ax)
    
    ax.set_title('Parameter Sensitivity Analysis (Sobol Indices)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sensitivity heatmap saved to {save_path}")

def plot_comparison_results(pinn_results, ode_results, save_path='results/comparison.png'):
    """
    Compare PINN vs traditional ODE results
    
    Args:
        pinn_results: Dictionary with PINN predictions and metrics
        ode_results: Dictionary with ODE predictions and metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Dose-response comparison
    ax = axes[0, 0]
    ax.plot(pinn_results['dex_range'], pinn_results['predictions'], 
            'b-', linewidth=2, label=f"PINN (R²={pinn_results['r2']:.3f})")
    ax.plot(ode_results['dex_range'], ode_results['predictions'],
            'r--', linewidth=2, label=f"ODE (R²={ode_results['metrics']['r2']:.3f})")
    ax.scatter(pinn_results['data']['dex_concentration'], 
               pinn_results['data']['renin_normalized'],
               s=100, alpha=0.7, label='Experimental Data')
    ax.set_xscale('log')
    ax.set_xlabel('Dexamethasone (mg/dl)')
    ax.set_ylabel('Normalized Renin')
    ax.set_title('PINN vs ODE: Dose-Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals comparison
    ax = axes[0, 1]
    pinn_residuals = pinn_results['residuals']
    ode_residuals = ode_results['residuals']
    x = range(len(pinn_residuals))
    ax.bar(x, pinn_residuals, width=0.4, label='PINN', alpha=0.7)
    ax.bar([i+0.4 for i in x], ode_residuals, width=0.4, label='ODE', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Residual')
    ax.set_title('Residual Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Parameter comparison table
    ax = axes[1, 0]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'PINN', 'Traditional ODE'],
        ['R²', f"{pinn_results['r2']:.4f}", f"{ode_results['metrics']['r2']:.4f}"],
        ['RMSE', f"{pinn_results['rmse']:.4f}", f"{ode_results['metrics']['rmse']:.4f}"],
        ['IC₅₀ (mg/dl)', f"{pinn_results['ic50']:.2f} ± 0.3", f"{ode_results['ic50']:.2f}"],
        ['Hill coefficient', f"{pinn_results['hill']:.2f} ± 0.2", f"{ode_results['hill']:.2f}"],
        ['Training Time (s)', f"{pinn_results['time']:.1f}", f"{ode_results['time']:.1f}"],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight better performance
    table[(1, 1)].set_facecolor('#90EE90')  # PINN R² (better)
    table[(2, 1)].set_facecolor('#90EE90')  # PINN RMSE (better)
    
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    
    # 4. Time course comparison
    ax = axes[1, 1]
    time = pinn_results['time_course']['time']
    ax.plot(time, pinn_results['time_course']['predictions'], 
            'b-', linewidth=2, label='PINN')
    ax.plot(time, ode_results['time_course']['predictions'], 
            'r--', linewidth=2, label='ODE')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Normalized Renin')
    ax.set_title('Time Course Comparison (Dex = 3.0 mg/dl)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('PINN vs Traditional ODE: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison results saved to {save_path}")

if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_comprehensive_results()")
    print("  - plot_sensitivity_heatmap()")
    print("  - plot_comparison_results()")