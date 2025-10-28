"""
Comprehensive IEEE Access Submission Analysis (Deep Ensemble Version)

This script runs ALL enhanced analyses for the IEEE Access submission using a
Deep Ensemble of PINN models for robust uncertainty quantification.
1. Ablation studies (10 configurations)
2. Multi-timepoint temporal validation
3. Dose-response extrapolation with cross-validation
4. Enhanced diagnostics and visualizations
5. Parameter uncertainty quantification from the ensemble
6. Comprehensive reporting

Run this after training the Deep Ensemble model (3_train_pinn_model.py)

Estimated runtime: 2-4 hours depending on hardware
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import prepare_training_data
from src.model import ReninPINN # CHANGED: Import standard PINN
from src.ablation_study import AblationStudy
from src.temporal_validation import TemporalValidator, generate_temporal_validation_report
from src.dose_response_analysis import DoseResponseAnalyzer, generate_dose_response_report
from src.enhanced_visualization import EnhancedDiagnostics
from src.comprehensive_reporting import ComprehensiveReporter
from src.statistical_utils import calculate_metrics, residual_analysis


class EnsembleModel:
    """
    A wrapper class for a list of PyTorch models to make them behave like a single model.
    Predictions are the mean of the ensemble members.
    """
    def __init__(self, models):
        if not models:
            raise ValueError("Model list for ensemble cannot be empty.")
        self.models = models
        # Get architecture from the first model
        self.layers = self.models[0].layers

    def __call__(self, t, dex):
        """Forward pass: returns the mean prediction of the ensemble."""
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(t, dex).cpu().numpy()
                predictions.append(pred)
        
        # Stack predictions and calculate mean
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        return torch.tensor(mean_pred, dtype=torch.float32)

    def predict_with_uncertainty(self, t, dex, n_samples=None, device=None):
        """Returns mean and std deviation of the ensemble predictions.
        
        Args:
            t: Time tensor or array
            dex: Dose tensor or array
            n_samples: Number of samples (ignored for ensemble, kept for compatibility)
            device: Device to use (ignored, kept for compatibility)
        
        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation of predictions
            samples: All predictions (for compatibility)
        """
        predictions = []
        with torch.no_grad():
            # Ensure input tensors are on the same device as the model
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, dtype=torch.float32)
            if isinstance(dex, np.ndarray):
                dex = torch.tensor(dex, dtype=torch.float32)
            
            for model in self.models:
                model.eval()
                # Move tensors to the model's device
                model_device = next(model.parameters()).device
                t_device = t.to(model_device)
                dex_device = dex.to(model_device)
                
                # Ensure tensors have the correct shape (batch_size, 1)
                if t_device.dim() == 1:
                    t_device = t_device.unsqueeze(1)
                if dex_device.dim() == 1:
                    dex_device = dex_device.unsqueeze(1)
                
                pred = model(t_device, dex_device).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Return samples for compatibility
        return mean_pred, std_pred, predictions

    def get_params(self):
        """Returns the mean and std of parameters across the ensemble."""
        all_params = [model.get_params() for model in self.models]
        param_means = {key: np.mean([p[key] for p in all_params]) for key in all_params[0]}
        param_stds = {key: np.std([p[key] for p in all_params]) for key in all_params[0]}
        return {'mean': param_means, 'std': param_stds}
    
    def to(self, device):
        """
        Move all models in the ensemble to the specified device.
        
        Args:
            device: Device to move models to ('cpu' or 'cuda')
        
        Returns:
            Self for method chaining
        """
        for model in self.models:
            model.to(device)
        return self
    
    def eval(self):
        """
        Set all models in the ensemble to evaluation mode.
        
        Returns:
            Self for method chaining
        """
        for model in self.models:
            model.eval()
        return self


def load_trained_ensemble(ensemble_dir='results/models/ensemble/', device='cpu'):
    """Load the trained Deep Ensemble of PINN models"""
    print("\n" + "="*80)
    print("LOADING TRAINED DEEP ENSEMBLE")
    print("="*80)
    
    if not os.path.exists(ensemble_dir):
        print(f"ERROR: Ensemble directory not found at {ensemble_dir}")
        print("Please run 3_train_pinn_model.py first to train the ensemble.")
        return None

    # Find all model files in the directory
    model_files = [f for f in os.listdir(ensemble_dir) if f.startswith('pinn_ensemble_member_') and f.endswith('.pth')]
    model_files.sort() # Ensure consistent ordering
    
    if not model_files:
        print(f"ERROR: No ensemble model files found in {ensemble_dir}")
        return None

    print(f"  Found {len(model_files)} trained models.")
    
    ensemble_models = []
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(ensemble_dir, model_file)
        model = ReninPINN(hidden_layers=[128, 128, 128, 128])
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        ensemble_models.append(model)
        print(f"  [OK] Loaded {model_file}")

    print(f"\n[OK] All {len(ensemble_models)} ensemble models loaded successfully.")
    
    # Wrap the list of models in our helper class
    ensemble_wrapper = EnsembleModel(ensemble_models)
    print(f"  Device: {device}")
    print(f"  Parameters per model: {sum(p.numel() for p in ensemble_models[0].parameters()):,} total")
    
    return ensemble_wrapper


def run_ablation_study(data_dict, device='cpu', n_epochs=3000, n_runs=3):
    """Run comprehensive ablation study"""
    print("\n" + "="*80)
    print("STEP 1: ABLATION STUDY")
    print("="*80)
    print("Testing 10 different configurations...")
    print(f"- Epochs per configuration: {n_epochs}")
    print(f"- Runs per configuration: {n_runs}")
    print("="*80)
    
    study = AblationStudy(data_dict, device=device, verbose=True)
    
    results = study.run_full_ablation(n_epochs=n_epochs, n_runs=n_runs)
    
    # Save results
    study.save_results('results/comprehensive/ablation_study.json')
    
    # Generate comparison table
    df = study.generate_comparison_table()
    df.to_csv('results/comprehensive/ablation_table.csv', index=False)
    print(f"\n[OK] Ablation table saved to: results/comprehensive/ablation_table.csv")
    
    return results


def run_temporal_validation(ensemble_model, device='cpu'):
    """Run temporal validation with multi-timepoint simulations"""
    print("\n" + "="*80)
    print("STEP 2: TEMPORAL VALIDATION")
    print("="*80)
    print("Simulating dynamics at multiple timepoints using the ensemble...")
    print("="*80)
    
    # The validator can now accept the ensemble wrapper directly
    validator = TemporalValidator(ensemble_model, device=device)
    
    # Generate comprehensive temporal validation
    report = generate_temporal_validation_report(
        validator,
        dex_doses=[0.0, 0.3, 3.0, 30.0],
        output_dir='results/comprehensive/temporal'
    )
    
    print("\n[OK] Temporal validation complete")
    
    return report


def run_dose_response_analysis(ensemble_model, data_dict, device='cpu', n_epochs_cv=3000):
    """Run dose-response extrapolation and cross-validation"""
    print("\n" + "="*80)
    print("STEP 3: DOSE-RESPONSE EXTRAPOLATION")
    print("="*80)
    print("Analyzing dose-response curve and extrapolation using the ensemble...")
    print("="*80)
    
    report = generate_dose_response_report(
        ensemble_model, # Pass the ensemble wrapper
        data_dict,
        output_dir='results/comprehensive/dose_response',
        n_epochs_cv=n_epochs_cv,
        device=device
    )
    
    print("\n[OK] Dose-response analysis complete")
    
    return report


def generate_enhanced_visualizations(ensemble_model, data_dict, ablation_results, 
                                     temporal_results, dose_response_results, 
                                     device='cpu'):
    """Generate all enhanced diagnostic plots using the ensemble."""
    print("\n" + "="*80)
    print("STEP 4: ENHANCED VISUALIZATIONS")
    print("="*80)
    print("Generating publication-quality figures from ensemble predictions...")
    print("="*80)
    
    diagnostics = EnhancedDiagnostics(output_dir='results/comprehensive/figures')
    
    # Get ensemble predictions (mean and std)
    t_tensor = torch.tensor(
        data_dict['time'], dtype=torch.float32
    ).reshape(-1, 1).to(device)
    dex_tensor = torch.tensor(
        data_dict['dex_concentration'], dtype=torch.float32
    ).reshape(-1, 1).to(device)
    
    mean_pred, std_pred, _ = ensemble_model.predict_with_uncertainty(t_tensor, dex_tensor)
    y_pred = mean_pred[:, 2]
    y_std = std_pred[:, 2]
    y_true = data_dict['renin_normalized']
    
    # 1. Predicted vs. Observed
    print("  1. Predicted vs. Observed plot...")
    diagnostics.plot_predicted_vs_observed(
        y_true, y_pred,
        labels=[f"{d:.1f} mg/dl" for d in data_dict['dex_concentration']],
        filename='enhanced_predicted_vs_observed.png'
    )
    
    # 2. Residual Diagnostics
    print("  2. Residual diagnostic plots...")
    diagnostics.plot_residual_diagnostics(
        y_true, y_pred, data_dict['dex_concentration'],
        filename='enhanced_residual_diagnostics.png'
    )
    
    # 3. Parameter Uncertainty from Ensemble
    print("  3. Parameter uncertainty distributions from ensemble...")
    param_stats = ensemble_model.get_params()
    
    param_samples = {
        'IC50': np.random.normal(param_stats['mean']['log_IC50'], param_stats['std']['log_IC50'], 100),
        'hill': np.random.normal(param_stats['mean']['log_hill'], param_stats['std']['log_hill'], 100)
    }
    
    diagnostics.plot_parameter_uncertainty(
        param_samples,
        filename='enhanced_parameter_uncertainty.png'
    )
    
    # 4. Ablation Comparison
    print("  4. Ablation study comparison...")
    diagnostics.plot_ablation_comparison(
        ablation_results,
        filename='enhanced_ablation_comparison.png'
    )
    
    # 5. Time Courses
    print("  5. Time course simulations...")
    diagnostics.plot_time_course_with_uncertainty(
        temporal_results['time_course'],
        filename='enhanced_time_courses.png'
    )
    
    # 6. Dose-Response Extrapolation
    print("  6. Dose-response extrapolation...")
    training_doses = np.unique(data_dict['dex_concentration'])
    diagnostics.plot_dose_response_extrapolation(
        dose_response_results['dose_response_curve'],
        training_doses,
        filename='enhanced_dose_response_extrapolation.png'
    )
    
    print("\n[OK] All enhanced visualizations generated")
    print(f"  Saved to: results/comprehensive/figures/")


def generate_comprehensive_report(ablation_results, temporal_results, 
                                  dose_response_results, model_performance, ensemble_params):
    """Generate final comprehensive report"""
    print("\n" + "="*80)
    print("STEP 5: COMPREHENSIVE REPORTING")
    print("="*80)
    print("Generating final reports and tables...")
    print("="*80)
    
    reporter = ComprehensiveReporter(output_dir='results/comprehensive')
    
    # Add all results
    reporter.add_ablation_results(ablation_results)
    reporter.add_temporal_validation(temporal_results)
    reporter.add_dose_response(dose_response_results)
    reporter.add_performance_metrics(model_performance)
    
    # Add parameter uncertainty from the ensemble
    param_uncertainty = {}
    for key, value in ensemble_params['mean'].items():
        param_uncertainty[key] = {
            'mean': value,
            'std': ensemble_params['std'][key],
            'ci_95': (value - 1.96 * ensemble_params['std'][key], value + 1.96 * ensemble_params['std'][key])
        }
    reporter.add_parameter_uncertainty(param_uncertainty)
    
    # Generate comprehensive report
    report_path = reporter.generate_comprehensive_report(
        include_latex=True,
        include_json=True
    )
    
    print(f"\n[OK] Comprehensive report generated")
    print(f"  Main report: {report_path}")
    print(f"  LaTeX tables: results/comprehensive/latex_tables/")
    print(f"  JSON data: results/comprehensive/")


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE IEEE ACCESS SUBMISSION ANALYSIS (DEEP ENSEMBLE)")
    print("Physics-Informed Neural Networks for Glucocorticoid-Renin Modeling")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis comprehensive analysis includes:")
    print("  1. Ablation Study (10 configurations)")
    print("  2. Temporal Validation (multi-timepoint simulations)")
    print("  3. Dose-Response Extrapolation (with cross-validation)")
    print("  4. Enhanced Diagnostics (publication-quality figures)")
    print("  5. Comprehensive Reporting (tables, LaTeX, JSON)")
    print("\nEstimated time: 2-4 hours")
    print("="*80)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directories
    os.makedirs('results/comprehensive', exist_ok=True)
    os.makedirs('results/comprehensive/figures', exist_ok=True)
    os.makedirs('results/comprehensive/temporal', exist_ok=True)
    os.makedirs('results/comprehensive/dose_response', exist_ok=True)
    os.makedirs('results/comprehensive/latex_tables', exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data_dict = prepare_training_data(dataset='elisa', use_log_scale=False)
    print(f"[OK] Data loaded: {data_dict['n_samples']} samples")
    
    # Load trained ensemble
    ensemble_model = load_trained_ensemble(device=str(device))
    if ensemble_model is None:
        print("\nERROR: Cannot proceed without a trained ensemble.")
        print("Please run 3_train_pinn_model.py first.")
        return
    
    # Get baseline performance from the ensemble
    t_tensor = torch.tensor(
        data_dict['time'], dtype=torch.float32
    ).reshape(-1, 1).to(device)
    dex_tensor = torch.tensor(
        data_dict['dex_concentration'], dtype=torch.float32
    ).reshape(-1, 1).to(device)
    
    mean_pred, std_pred, _ = ensemble_model.predict_with_uncertainty(t_tensor, dex_tensor)
    y_pred = mean_pred[:, 2]
    y_true = data_dict['renin_normalized']
    model_performance = calculate_metrics(y_true, y_pred)
    
    # Get ensemble parameters for reporting
    ensemble_params = ensemble_model.get_params()
    
    print(f"\nEnsemble Model Performance:")
    print(f"  R² = {model_performance['r2']:.4f}")
    print(f"  RMSE = {model_performance['rmse']:.4f}")
    print(f"  IC50 = {ensemble_params['mean']['log_IC50']:.2f} ± {ensemble_params['std']['log_IC50']:.2f} mg/dl")
    print(f"  Hill = {ensemble_params['mean']['log_hill']:.2f} ± {ensemble_params['std']['log_hill']:.2f}")
    
    # Configuration - adjust these for faster testing
    ABLATION_EPOCHS = int(os.environ.get('ABLATION_EPOCHS', '3000'))
    ABLATION_RUNS = int(os.environ.get('ABLATION_RUNS', '3'))
    CV_EPOCHS = int(os.environ.get('CV_EPOCHS', '3000'))
    
    print(f"\nConfiguration:")
    print(f"  Ablation epochs: {ABLATION_EPOCHS}")
    print(f"  Ablation runs per config: {ABLATION_RUNS}")
    print(f"  Cross-validation epochs: {CV_EPOCHS}")
    print("\nTo reduce runtime, set environment variables:")
    print("  ABLATION_EPOCHS=1000 ABLATION_RUNS=1 CV_EPOCHS=1000")
    print("\nStarting analysis...")
    
    # Run all analyses
    try:
        # 1. Ablation Study
        ablation_results = run_ablation_study(
            data_dict, 
            device=str(device),
            n_epochs=ABLATION_EPOCHS,
            n_runs=ABLATION_RUNS
        )
        
        # 2. Temporal Validation
        temporal_results = run_temporal_validation(ensemble_model, device=str(device))
        
        # 3. Dose-Response Analysis
        dose_response_results = run_dose_response_analysis(
            ensemble_model, 
            data_dict,
            device=str(device),
            n_epochs_cv=CV_EPOCHS
        )
        
        # 4. Enhanced Visualizations
        generate_enhanced_visualizations(
            ensemble_model, data_dict, ablation_results,
            temporal_results, dose_response_results,
            device=device
        )
        
        # 5. Comprehensive Report
        generate_comprehensive_report(
            ablation_results, temporal_results,
            dose_response_results, model_performance, ensemble_params
        )
        
        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nResults saved to:")
        print("  - results/comprehensive/")
        print("    +-- ablation_study.json")
        print("    +-- ablation_table.csv")
        print("    +-- comprehensive_report_*.txt")
        print("    +-- report_data_*.json")
        print("    +-- figures/")
        print("    |   +-- enhanced_predicted_vs_observed.png")
        print("    |   +-- enhanced_residual_diagnostics.png")
        print("    |   +-- enhanced_parameter_uncertainty.png")
        print("    |   +-- enhanced_ablation_comparison.png")
        print("    |   +-- enhanced_time_courses.png")
        print("    |   +-- enhanced_dose_response_extrapolation.png")
        print("    +-- latex_tables/")
        print("    |   +-- ablation_table.tex")
        print("    |   +-- parameter_table.tex")
        print("    |   +-- validation_summary.tex")
        print("    +-- temporal/")
        print("    +-- dose_response/")
        print("\nAll enhancements for IEEE Access submission are complete!")
        print("Use the generated figures and tables in your manuscript.")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        print("Partial results may be available in results/comprehensive/")
    
    except Exception as e:
        print(f"\n\nERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPartial results may be available in results/comprehensive/")


if __name__ == "__main__":
    main()