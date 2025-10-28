"""
Dose-Response Extrapolation and Cross-Validation Analysis (Deep Ensemble Version)

This module tests model generalization by:
- Predicting renin at doses outside the training range
- Cross-validation by leaving out each dose in turn using a pre-trained ensemble
- Generating continuous dose-response curves with uncertainty bands
- Evaluating extrapolation reliability

For IEEE Access submission requirements.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

from .statistical_utils import calculate_metrics


class DoseResponseAnalyzer:
    """
    Analyzes dose-response relationships and extrapolation capability.
    UPDATED: Now designed to work with any model that has a `predict_with_uncertainty` method,
    such as the Deep Ensemble wrapper.
    """
    
    def __init__(self, device: str = 'cpu', verbose: bool = True):
        """
        Initialize dose-response analyzer
        
        Args:
            device: 'cpu' or 'cuda'
            verbose: Print progress messages
        """
        self.device = torch.device(device)
        self.verbose = verbose
        self.models = {}
        self.predictions = {}
    
    def generate_continuous_dose_response(self,
                                         model,
                                         dose_range: Tuple[float, float] = (0.0, 100.0),
                                         n_points: int = 200,
                                         time_point: float = 24.0,
                                         n_mc_samples: int = 100) -> Dict:
        """
        Generate continuous dose-response curve with uncertainty.
        UPDATED: Now model-agnostic, relies on `predict_with_uncertainty` method.
        
        Args:
            model: Trained model (e.g., EnsembleModel wrapper)
            dose_range: (min_dose, max_dose) in mg/dl
            n_points: Number of points to evaluate
            time_point: Time point for evaluation (hours)
            n_mc_samples: Samples for uncertainty (used by ensemble)
            
        Returns:
            results: Dose-response curve with uncertainties
        """
        # Generate dose points (log-spaced for better coverage)
        doses = np.logspace(
            np.log10(max(dose_range[0], 0.01)),
            np.log10(dose_range[1]),
            n_points
        )
        
        # Add zero dose at the beginning if needed
        if dose_range[0] == 0.0:
            doses = np.concatenate([[0.0], doses])
        
        time_points = np.full_like(doses, time_point)
        
        # CHANGED: Removed isinstance check. Assume model has predict_with_uncertainty.
        # This works for our EnsembleModel and other models with this interface.
        try:
            mean, std, samples = model.predict_with_uncertainty(
                time_points,
                doses,
                n_samples=n_mc_samples,
                device=str(self.device)
            )
            renin_mean = mean[:, 2]
            renin_std = std[:, 2]
            
            # Calculate 95% confidence intervals
            renin_samples = samples[:, :, 2]
            ci_lower = np.percentile(renin_samples, 2.5, axis=0)
            ci_upper = np.percentile(renin_samples, 97.5, axis=0)
        except AttributeError:
            # Fallback for models without predict_with_uncertainty (deterministic models)
            if self.verbose:
                print("Warning: Model does not have predict_with_uncertainty. Using deterministic prediction.")
            model.eval()
            with torch.no_grad():
                t_tensor = torch.tensor(
                    time_points, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                dex_tensor = torch.tensor(
                    doses, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                pred = model(t_tensor, dex_tensor).cpu().numpy()
            
            renin_mean = pred[:, 2]
            renin_std = np.zeros_like(renin_mean)
            ci_lower = renin_mean
            ci_upper = renin_mean
        
        results = {
            'doses': doses,
            'renin_mean': renin_mean,
            'renin_std': renin_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time_point': time_point,
            'dose_range': dose_range
        }
        
        return results
    
    def cross_validation_leave_one_out(self,
                                      data_dict: Dict,
                                      model) -> Dict:
        """
        Leave-one-dose-out cross-validation using a pre-trained ensemble.
        REFACTORED: No longer trains models. Uses the provided pre-trained ensemble for evaluation.
        
        Args:
            data_dict: Full dataset
            model: Pre-trained model (e.g., EnsembleModel wrapper)
            
        Returns:
            cv_results: Cross-validation results
        """
        if self.verbose:
            print("\n" + "="*70)
            print("LEAVE-ONE-OUT CROSS-VALIDATION (DEEP ENSEMBLE)")
            print("="*70)
        
        # Get unique doses
        unique_doses = np.unique(data_dict['dex_concentration'])
        n_folds = len(unique_doses)
        
        cv_results = {
            'folds': {},
            'summary': {}
        }
        
        all_predictions = []
        all_true_values = []
        all_uncertainties = []
        
        for fold_idx, held_out_dose in enumerate(unique_doses):
            if self.verbose:
                print(f"\nFold {fold_idx + 1}/{n_folds}: Holding out dose {held_out_dose} mg/dl")
            
            # Split data
            test_mask = data_dict['dex_concentration'] == held_out_dose
            
            test_data = {
                'time': data_dict['time'][test_mask],
                'dex_concentration': data_dict['dex_concentration'][test_mask],
                'renin_normalized': data_dict['renin_normalized'][test_mask],
                'renin_std': data_dict['renin_std'][test_mask]
            }
            
            # CHANGED: Use the pre-trained ensemble to predict on held-out data
            # No training is performed in this fold.
            mean, std, _ = model.predict_with_uncertainty(
                test_data['time'],
                test_data['dex_concentration'],
                n_samples=100,
                device=str(self.device)
            )
            predictions = mean[:, 2]
            uncertainties = std[:, 2]
            
            # Calculate metrics
            metrics = calculate_metrics(test_data['renin_normalized'], predictions)
            
            # Store results
            cv_results['folds'][held_out_dose] = {
                'predictions': predictions,
                'true_values': test_data['renin_normalized'],
                'uncertainties': uncertainties,
                'metrics': metrics
            }
            
            all_predictions.extend(predictions)
            all_true_values.extend(test_data['renin_normalized'])
            all_uncertainties.extend(uncertainties)
            
            if self.verbose:
                print(f"  R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        
        # Calculate overall metrics
        overall_metrics = calculate_metrics(
            np.array(all_true_values),
            np.array(all_predictions)
        )
        
        cv_results['summary'] = {
            'overall_r2': overall_metrics['r2'],
            'overall_rmse': overall_metrics['rmse'],
            'overall_mae': overall_metrics['mae'],
            'mean_uncertainty': np.mean(all_uncertainties)
        }
        
        if self.verbose:
            print("\n" + "="*70)
            print("CROSS-VALIDATION SUMMARY")
            print("-"*70)
            print(f"Overall R² = {overall_metrics['r2']:.4f}")
            print(f"Overall RMSE = {overall_metrics['rmse']:.4f}")
            print(f"Overall MAE = {overall_metrics['mae']:.4f}")
            print("="*70)
        
        return cv_results
    
    def test_extrapolation_robustness(self,
                                     model,
                                     training_doses: np.ndarray,
                                     test_doses: np.ndarray,
                                     time_point: float = 24.0,
                                     n_mc_samples: int = 100) -> Dict:
        """
        Test model's extrapolation beyond training dose range.
        UPDATED: Now model-agnostic.
        
        Args:
            model: Trained model (e.g., EnsembleModel wrapper)
            training_doses: Doses used for training
            test_doses: Doses to test (should include extrapolation)
            time_point: Time for evaluation
            n_mc_samples: Samples for uncertainty (used by ensemble)
            
        Returns:
            results: Extrapolation analysis
        """
        time_points = np.full_like(test_doses, time_point)
        
        # CHANGED: Removed isinstance check.
        try:
            mean, std, _ = model.predict_with_uncertainty(
                time_points,
                test_doses,
                n_samples=n_mc_samples,
                device=str(self.device)
            )
            renin_mean = mean[:, 2]
            renin_std = std[:, 2]
        except AttributeError:
            if self.verbose:
                print("Warning: Model does not have predict_with_uncertainty. Using deterministic prediction.")
            model.eval()
            with torch.no_grad():
                t_tensor = torch.tensor(
                    time_points, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                dex_tensor = torch.tensor(
                    test_doses, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                pred = model(t_tensor, dex_tensor).cpu().numpy()
            
            renin_mean = pred[:, 2]
            renin_std = np.zeros_like(renin_mean)
        
        # Classify doses
        min_train = training_doses.min()
        max_train = training_doses.max()
        
        interpolation_mask = (test_doses >= min_train) & (test_doses <= max_train)
        extrapolation_low_mask = test_doses < min_train
        extrapolation_high_mask = test_doses > max_train
        
        results = {
            'test_doses': test_doses,
            'predictions': renin_mean,
            'uncertainties': renin_std,
            'training_range': (min_train, max_train),
            'interpolation': {
                'doses': test_doses[interpolation_mask],
                'predictions': renin_mean[interpolation_mask],
                'uncertainties': renin_std[interpolation_mask]
            },
            'extrapolation_low': {
                'doses': test_doses[extrapolation_low_mask],
                'predictions': renin_mean[extrapolation_low_mask],
                'uncertainties': renin_std[extrapolation_low_mask]
            },
            'extrapolation_high': {
                'doses': test_doses[extrapolation_high_mask],
                'predictions': renin_mean[extrapolation_high_mask],
                'uncertainties': renin_std[extrapolation_high_mask]
            }
        }
        
        # Calculate uncertainty ratios
        if np.any(interpolation_mask):
            interp_unc = np.mean(renin_std[interpolation_mask])
        else:
            interp_unc = 0
        
        if np.any(extrapolation_high_mask):
            extrap_high_unc = np.mean(renin_std[extrapolation_high_mask])
            results['uncertainty_ratio_high'] = extrap_high_unc / (interp_unc + 1e-6)
        else:
            results['uncertainty_ratio_high'] = 0
        
        if np.any(extrapolation_low_mask):
            extrap_low_unc = np.mean(renin_std[extrapolation_low_mask])
            results['uncertainty_ratio_low'] = extrap_low_unc / (interp_unc + 1e-6)
        else:
            results['uncertainty_ratio_low'] = 0
        
        return results
    
    def check_saturation_behavior(self,
                                  dose_response: Dict,
                                  saturation_threshold: float = 0.01) -> Dict:
        """
        Check if model shows appropriate saturation at high doses.
        (No changes needed)
        
        Args:
            dose_response: Output from generate_continuous_dose_response
            saturation_threshold: Max slope for saturation (normalized units/dose)
            
        Returns:
            saturation_analysis: Saturation characteristics
        """
        doses = dose_response['doses']
        renin = dose_response['renin_mean']
        
        # Calculate derivative (slope)
        slopes = np.gradient(renin, doses)
        
        # Find where slope becomes very small (saturated)
        saturated_mask = np.abs(slopes) < saturation_threshold
        
        if np.any(saturated_mask):
            saturation_dose = doses[saturated_mask][0]
            saturation_detected = True
        else:
            saturation_dose = None
            saturation_detected = False
        
        # Check for plateau at high doses
        high_dose_mask = doses > doses[-1] * 0.8
        if np.any(high_dose_mask):
            high_dose_variation = np.std(renin[high_dose_mask])
            plateau_detected = high_dose_variation < 0.05
        else:
            plateau_detected = False
        
        analysis = {
            'saturation_detected': saturation_detected,
            'saturation_dose': saturation_dose,
            'plateau_detected': plateau_detected,
            'slopes': slopes,
            'min_slope': np.min(np.abs(slopes)),
            'max_slope': np.max(np.abs(slopes)),
            'high_dose_variation': high_dose_variation if np.any(high_dose_mask) else None
        }
        
        return analysis
    
    def compare_interpolation_vs_extrapolation(self,
                                              cv_results: Dict,
                                              extrapolation_results: Dict) -> Dict:
        """
        Compare model performance in interpolation vs extrapolation.
        (No changes needed)
        
        Args:
            cv_results: Cross-validation results
            extrapolation_results: Extrapolation test results
            
        Returns:
            comparison: Performance comparison
        """
        # Get interpolation performance from CV
        interp_r2 = cv_results['summary']['overall_r2']
        interp_rmse = cv_results['summary']['overall_rmse']
        interp_uncertainty = cv_results['summary']['mean_uncertainty']
        
        # Extrapolation uncertainties
        extrap_high_unc = np.mean(extrapolation_results['extrapolation_high']['uncertainties'])
        extrap_low_unc = np.mean(extrapolation_results['extrapolation_low']['uncertainties'])
        
        comparison = {
            'interpolation': {
                'r2': interp_r2,
                'rmse': interp_rmse,
                'mean_uncertainty': interp_uncertainty
            },
            'extrapolation_high': {
                'mean_uncertainty': extrap_high_unc,
                'uncertainty_increase': extrap_high_unc / (interp_uncertainty + 1e-6)
            },
            'extrapolation_low': {
                'mean_uncertainty': extrap_low_unc,
                'uncertainty_increase': extrap_low_unc / (interp_uncertainty + 1e-6)
            },
            'interpretation': self._interpret_comparison(
                interp_uncertainty,
                extrap_high_unc,
                extrap_low_unc
            )
        }
        
        return comparison
    
    def _interpret_comparison(self, interp_unc, extrap_high_unc, extrap_low_unc):
        """Interpret interpolation vs extrapolation comparison"""
        
        interpretation = []
        
        if extrap_high_unc > 2 * interp_unc:
            interpretation.append(
                "High uncertainty at high doses suggests caution in extrapolation beyond training range"
            )
        elif extrap_high_unc > 1.5 * interp_unc:
            interpretation.append(
                "Moderately increased uncertainty at high doses, typical for extrapolation"
            )
        else:
            interpretation.append(
                "Uncertainty remains stable at high doses, suggesting good model generalization"
            )
        
        if extrap_low_unc > 1.5 * interp_unc:
            interpretation.append(
                "Increased uncertainty at low doses indicates limited data coverage in this region"
            )
        else:
            interpretation.append(
                "Low-dose predictions appear reliable"
            )
        
        return interpretation


def generate_dose_response_report(model,
                                  data_dict: Dict,
                                  output_dir: str = 'results/dose_response',
                                  n_epochs_cv: int = 5000,
                                  device: str = 'cpu') -> Dict:
    """
    Generate comprehensive dose-response analysis report.
    UPDATED: Now uses a pre-trained model and removes unnecessary training parameters.
    
    Args:
        model: Pre-trained model (e.g., EnsembleModel wrapper)
        data_dict: Full dataset
        output_dir: Output directory
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        report: Complete dose-response analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DoseResponseAnalyzer(device=device, verbose=True)
    
    report = {}
    
    # 1. Continuous dose-response curve
    print("\nGenerating continuous dose-response curve...")
    dose_response = analyzer.generate_continuous_dose_response(
        model,
        dose_range=(0.0, 100.0),
        n_points=200,
        n_mc_samples=100
    )
    report['dose_response_curve'] = dose_response
    
    # 2. Cross-validation
    print("\nPerforming leave-one-out cross-validation...")
    # CHANGED: Removed n_epochs_cv and use_bayesian flag. Pass the model directly.
    cv_results = analyzer.cross_validation_leave_one_out(
        data_dict,
        model
    )
    report['cross_validation'] = cv_results
    
    # 3. Extrapolation testing
    print("\nTesting extrapolation robustness...")
    training_doses = np.unique(data_dict['dex_concentration'])
    test_doses = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    
    extrapolation = analyzer.test_extrapolation_robustness(
        model,
        training_doses,
        test_doses,
        n_mc_samples=100
    )
    report['extrapolation'] = extrapolation
    
    # 4. Saturation analysis
    print("\nAnalyzing saturation behavior...")
    saturation = analyzer.check_saturation_behavior(dose_response)
    report['saturation'] = saturation
    
    # 5. Comparison
    print("\nComparing interpolation vs extrapolation...")
    comparison = analyzer.compare_interpolation_vs_extrapolation(
        cv_results,
        extrapolation
    )
    report['comparison'] = comparison
    
    print("\nDose-response analysis complete!")
    
    # Save results to JSON
    import json
    json_path = os.path.join(output_dir, 'dose_response_results.json')
    
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            # Skip model objects in dictionaries
            return {k: convert_to_serializable(v) for k, v in obj.items() if k != 'model'}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None:
            return None
        # Skip PyTorch models and other non-serializable objects
        elif hasattr(obj, '__class__') and 'torch' in str(type(obj)):
            return None
        else:
            return obj
    
    # Convert entire report to JSON-serializable format
    json_report = convert_to_serializable(report)
    
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    
    return report


if __name__ == "__main__":
    print("Dose-Response Analysis Module (Deep Ensemble Version)")
    print("=" * 60)
    print("This module provides:")
    print("  - Continuous dose-response curves with uncertainty")
    print("  - Leave-one-out cross-validation using a pre-trained ensemble")
    print("  - Extrapolation robustness testing")
    print("  - Saturation behavior analysis")
    print("\nUsage:")
    print("  from src.dose_response_analysis import DoseResponseAnalyzer")
    print("  analyzer = DoseResponseAnalyzer()")
    print("  results = analyzer.generate_continuous_dose_response(model)")