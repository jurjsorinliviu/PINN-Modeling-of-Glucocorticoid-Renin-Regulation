"""
Multi-Timepoint Simulation and Temporal Validation (Deep Ensemble Version)

This module generates virtual time-series predictions to validate
model dynamics in lieu of additional experimental data at intermediate
timepoints. Uses the Deep Ensemble for robust uncertainty quantification.

For IEEE Access submission requirements.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Removed BayesianReninPINN import as it's no longer the primary method
from .model import ReninPINN


class TemporalValidator:
    """
    Validates temporal dynamics through multi-timepoint simulations.
    UPDATED: Now designed to work with any model that has a `predict_with_uncertainty` method,
    such as the Deep Ensemble wrapper.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Initialize temporal validator with a pre-trained model.
        
        Args:
            model: A pre-trained model (e.g., EnsembleModel wrapper).
            device: 'cpu' or 'cuda'.
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # CHANGED: No longer check for BayesianReninPINN. Relies on duck-typing.
        # The model is assumed to have a predict_with_uncertainty method.
        
    def simulate_time_course(self,
                            dex_doses: np.ndarray,
                            time_points: Optional[np.ndarray] = None,
                            n_mc_samples: int = 100) -> Dict:
        """
        Simulate complete time courses for multiple doses.
        UPDATED: Now uses the model's `predict_with_uncertainty` method.
        
        Args:
            dex_doses: Dexamethasone concentrations to simulate (mg/dl).
            time_points: Time points to evaluate (hours). Default: 0-48h.
            n_mc_samples: Number of samples for uncertainty (used by ensemble).
            
        Returns:
            results: Dictionary with predictions and uncertainties.
        """
        if time_points is None:
            time_points = np.linspace(0, 48, 200)
        
        results = {
            'time_points': time_points,
            'dex_doses': dex_doses,
            'predictions': {},
            'uncertainties': {},
            'trajectories': {}
        }
        
        for dose in dex_doses:
            # Create input tensors
            t_tensor = torch.tensor(
                time_points, dtype=torch.float32
            ).reshape(-1, 1).to(self.device)
            
            dex_tensor = torch.full_like(t_tensor, dose)
            
            # CHANGED: Use the model's predict_with_uncertainty method.
            # This works for both EnsembleModel and a single model with the method.
            try:
                mean, std, samples = self.model.predict_with_uncertainty(
                    time_points,
                    np.full_like(time_points, dose),
                    n_samples=n_mc_samples,
                    device=str(self.device)
                )
                
                results['predictions'][dose] = mean
                results['uncertainties'][dose] = std
                results['trajectories'][dose] = samples
            except AttributeError:
                # Fallback for models without predict_with_uncertainty
                print(f"Warning: Model does not have predict_with_uncertainty. Using deterministic prediction for dose {dose}.")
                with torch.no_grad():
                    pred = self.model(t_tensor, dex_tensor).cpu().numpy()
                
                results['predictions'][dose] = pred
                results['uncertainties'][dose] = np.zeros_like(pred)
                results['trajectories'][dose] = pred[np.newaxis, ...]
        
        return results
    
    def validate_physiological_plausibility(self, 
                                           simulation_results: Dict) -> Dict:
        """
        Check if simulated dynamics are physiologically plausible.
        (No changes needed).
        
        Args:
            simulation_results: Output from simulate_time_course.
            
        Returns:
            validation: Dictionary of validation checks.
        """
        validation = {
            'checks': {},
            'all_passed': True
        }
        
        time_points = simulation_results['time_points']
        
        for dose, predictions in simulation_results['predictions'].items():
            renin = predictions[:, 2] if predictions.ndim > 1 else predictions
            
            checks = {}
            
            # 1. Monotonicity: Renin should decrease with dose
            if dose > 0:
                initial_renin = renin[0]
                final_renin = renin[-1]
                checks['suppression'] = final_renin < initial_renin
            
            # 2. No negative values
            checks['non_negative'] = np.all(renin >= 0)
            
            # 3. Bounded response (shouldn't exceed baseline by much)
            checks['bounded'] = np.all(renin <= 1.5)
            
            # 4. Smooth dynamics (no abrupt changes)
            if len(time_points) > 1:
                derivatives = np.diff(renin) / np.diff(time_points)
                max_derivative = np.abs(derivatives).max()
                checks['smooth'] = max_derivative < 0.1  # Reasonable threshold
            
            # 5. Steady-state behavior (should stabilize)
            if len(time_points) > 10:
                late_variation = np.std(renin[-10:])
                checks['steady_state'] = late_variation < 0.05
            
            validation['checks'][f'dose_{dose}'] = checks
            
            # Update overall pass status
            if not all(checks.values()):
                validation['all_passed'] = False
        
        return validation
    
    def compare_with_ode_integration(self,
                                     dex_dose: float,
                                     time_points: Optional[np.ndarray] = None) -> Dict:
        """
        Compare PINN predictions with direct ODE integration.
        UPDATED: Now uses the mean parameters from the ensemble.
        
        Args:
            dex_dose: Dexamethasone concentration.
            time_points: Time points for comparison.
            
        Returns:
            comparison: PINN vs ODE predictions.
        """
        if time_points is None:
            time_points = np.linspace(0, 48, 200)
        
        # Get PINN prediction (mean from ensemble)
        t_tensor = torch.tensor(
            time_points, dtype=torch.float32
        ).reshape(-1, 1).to(self.device)
        
        dex_tensor = torch.full_like(t_tensor, dex_dose)
        
        # CHANGED: Use predict_with_uncertainty to get the mean prediction
        mean_pred, _, _ = self.model.predict_with_uncertainty(
            t_tensor, dex_tensor, n_samples=1
        )
        # mean_pred is already a numpy array from the ensemble
        pinn_pred = mean_pred
        
        # Get learned parameters (mean from ensemble)
        # CHANGED: Assumes model has a get_params method that returns mean/std
        if hasattr(self.model, 'get_params') and 'mean' in self.model.get_params():
            params = self.model.get_params()['mean']
        else:
            # Fallback for single models
            params = self.model.get_params()

        # Integrate ODE with learned parameters
        ode_pred = self._integrate_ode(
            time_points,
            dex_dose,
            params
        )
        
        # Calculate discrepancy
        discrepancy = np.abs(pinn_pred - ode_pred)
        
        comparison = {
            'time_points': time_points,
            'pinn_prediction': pinn_pred,
            'ode_prediction': ode_pred,
            'discrepancy': discrepancy,
            'mean_discrepancy': np.mean(discrepancy, axis=0),
            'max_discrepancy': np.max(discrepancy, axis=0),
            'parameters': params
        }
        
        return comparison
    
    def _integrate_ode(self,
                      time_points: np.ndarray,
                      dex: float,
                      params: Dict) -> np.ndarray:
        """
        Integrate the ODE system using scipy.
        (No changes needed).
        
        Args:
            time_points: Time points for integration.
            dex: Dexamethasone concentration (constant).
            params: Model parameters.
            
        Returns:
            solution: ODE solution at time_points.
        """
        # Extract parameters
        IC50 = params['log_IC50']
        hill = params['log_hill']
        k_synth_renin = params['log_k_synth_renin']
        k_deg_renin = params['log_k_deg_renin']
        k_synth_GR = params['log_k_synth_GR']
        k_deg_GR = params['log_k_deg_GR']
        k_bind = params['log_k_bind']
        k_unbind = params['log_k_unbind']
        k_nuclear = params['log_k_nuclear']
        k_translation = params['log_k_translation']
        k_secretion = params['log_k_secretion']
        
        def ode_system(y, t):
            """ODE system"""
            mRNA, protein, secreted, GR_free, GR_cyto, GR_nuc = y
            
            # Hill equation for inhibition
            inhibition = 1.0 / (1.0 + (GR_nuc / IC50) ** hill)
            
            # ODEs
            dmRNA_dt = k_synth_renin * inhibition - k_deg_renin * mRNA
            dprotein_dt = k_translation * mRNA - k_secretion * protein - k_deg_renin * protein
            dsecreted_dt = k_secretion * protein
            
            dGR_free_dt = k_synth_GR - k_bind * dex * GR_free + k_unbind * GR_cyto - k_deg_GR * GR_free
            dGR_cyto_dt = k_bind * dex * GR_free - k_unbind * GR_cyto - k_nuclear * GR_cyto
            dGR_nuc_dt = k_nuclear * GR_cyto - k_deg_GR * GR_nuc
            
            return [dmRNA_dt, dprotein_dt, dsecreted_dt, 
                   dGR_free_dt, dGR_cyto_dt, dGR_nuc_dt]
        
        # Initial conditions
        y0 = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        
        # Integrate
        solution = odeint(ode_system, y0, time_points)
        
        return solution
    
    def simulate_perturbations(self,
                              perturbation_type: str = 'pulse',
                              n_mc_samples: int = 100) -> Dict:
        """
        Simulate model response to different perturbations.
        UPDATED: Now uses the model's `predict_with_uncertainty` method.
        
        Args:
            perturbation_type: 'pulse', 'repeated_dosing', or 'gradual_increase'.
            n_mc_samples: Samples for uncertainty (used by ensemble).
            
        Returns:
            results: Perturbation simulation results.
        """
        time_points = np.linspace(0, 96, 400)  # Extended to 96h
        
        if perturbation_type == 'pulse':
            # Single pulse at t=24h
            dex_profile = self._create_pulse_profile(time_points, pulse_time=24, duration=6)
            
        elif perturbation_type == 'repeated_dosing':
            # Repeated doses every 24h
            dex_profile = self._create_repeated_dose_profile(time_points, interval=24, duration=6)
            
        elif perturbation_type == 'gradual_increase':
            # Gradually increasing dose
            dex_profile = self._create_gradual_increase_profile(time_points)
        
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        # Simulate response
        # CHANGED: Use predict_with_uncertainty for robustness
        try:
            mean_pred, std_pred, _ = self.model.predict_with_uncertainty(
                time_points,
                dex_profile,
                n_samples=n_mc_samples,
                device=str(self.device)
            )
        except AttributeError:
            print("Warning: Model does not have predict_with_uncertainty. Using deterministic prediction for perturbations.")
            with torch.no_grad():
                t_tensor = torch.tensor(
                    time_points, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                dex_tensor = torch.tensor(
                    dex_profile, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                mean_pred = self.model(t_tensor, dex_tensor).cpu().numpy()
                std_pred = np.zeros_like(mean_pred)
        
        results = {
            'time_points': time_points,
            'dex_profile': dex_profile,
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'perturbation_type': perturbation_type
        }
        
        return results
    
    def _create_pulse_profile(self, time_points, pulse_time, duration):
        """Create single pulse dexamethasone profile"""
        profile = np.zeros_like(time_points)
        mask = (time_points >= pulse_time) & (time_points < pulse_time + duration)
        profile[mask] = 3.0  # 3 mg/dl pulse
        return profile
    
    def _create_repeated_dose_profile(self, time_points, interval, duration):
        """Create repeated dosing profile"""
        profile = np.zeros_like(time_points)
        for start_time in range(0, int(time_points[-1]), interval):
            mask = (time_points >= start_time) & (time_points < start_time + duration)
            profile[mask] = 3.0
        return profile
    
    def _create_gradual_increase_profile(self, time_points):
        """Create gradually increasing dose profile"""
        max_dose = 10.0
        profile = (time_points / time_points[-1]) * max_dose
        return profile
    
    def analyze_transient_response(self,
                                  dex_dose: float = 3.0,
                                  n_mc_samples: int = 100) -> Dict:
        """
        Analyze transient response characteristics.
        UPDATED: Now uses the model's `predict_with_uncertainty` method.
        
        Args:
            dex_dose: Dexamethasone concentration.
            n_mc_samples: Samples for uncertainty (used by ensemble).
            
        Returns:
            analysis: Transient response characteristics.
        """
        time_points = np.linspace(0, 48, 500)
        
        # Simulate response
        # CHANGED: Use predict_with_uncertainty to get the mean prediction
        try:
            mean, std, _ = self.model.predict_with_uncertainty(
                time_points,
                np.full_like(time_points, dex_dose),
                n_samples=n_mc_samples,
                device=str(self.device)
            )
            renin = mean[:, 2]
            renin_std = std[:, 2]
        except AttributeError:
            print("Warning: Model does not have predict_with_uncertainty. Using deterministic prediction for transient analysis.")
            with torch.no_grad():
                t_tensor = torch.tensor(
                    time_points, dtype=torch.float32
                ).reshape(-1, 1).to(self.device)
                
                dex_tensor = torch.full_like(t_tensor, dex_dose)
                
                pred = self.model(t_tensor, dex_tensor).cpu().numpy()
                renin = pred[:, 2]
                renin_std = np.zeros_like(renin)
        
        # Analyze characteristics
        analysis = {}
        
        # Peak suppression time
        min_idx = np.argmin(renin)
        analysis['peak_suppression_time'] = time_points[min_idx]
        analysis['peak_suppression_value'] = renin[min_idx]
        
        # Time to reach 50% of steady-state suppression
        steady_state = renin[-1]
        target = (1.0 + steady_state) / 2
        above_target = renin > target
        if np.any(~above_target):
            cross_idx = np.where(~above_target)[0][0]
            analysis['t_half'] = time_points[cross_idx]
        else:
            analysis['t_half'] = None
        
        # Response rate (initial slope)
        if len(time_points) > 10:
            initial_slope = (renin[10] - renin[0]) / (time_points[10] - time_points[0])
            analysis['initial_response_rate'] = initial_slope
        
        # Overshoot (if any)
        if renin[-1] < renin[min_idx]:
            analysis['overshoot'] = False
        else:
            analysis['overshoot'] = True
            analysis['overshoot_magnitude'] = renin[-1] - renin[min_idx]
        
        # Store full trajectory
        analysis['time_points'] = time_points
        analysis['renin_trajectory'] = renin
        analysis['renin_uncertainty'] = renin_std
        
        return analysis


def generate_temporal_validation_report(validator: TemporalValidator,
                                       dex_doses: List[float] = [0.0, 0.3, 3.0, 30.0],
                                       output_dir: str = 'results/temporal_validation') -> Dict:
    """
    Generate comprehensive temporal validation report.
    (No changes needed).
    
    Args:
        validator: TemporalValidator instance.
        dex_doses: Doses to analyze.
        output_dir: Directory for output files.
        
    Returns:
        report: Complete validation report.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report = {}
    
    # 1. Time course simulations
    print("Simulating time courses...")
    time_course = validator.simulate_time_course(
        dex_doses=np.array(dex_doses),
        n_mc_samples=100
    )
    report['time_course'] = time_course
    
    # 2. Physiological plausibility
    print("Validating physiological plausibility...")
    plausibility = validator.validate_physiological_plausibility(time_course)
    report['plausibility'] = plausibility
    
    # 3. ODE comparison
    print("Comparing with ODE integration...")
    ode_comparisons = {}
    for dose in dex_doses[1:]:  # Skip control
        comparison = validator.compare_with_ode_integration(dose)
        ode_comparisons[dose] = comparison
    report['ode_comparison'] = ode_comparisons
    
    # 4. Perturbation analysis
    print("Analyzing perturbation responses...")
    perturbations = {}
    for ptype in ['pulse', 'repeated_dosing']:
        pert = validator.simulate_perturbations(ptype, n_mc_samples=100)
        perturbations[ptype] = pert
    report['perturbations'] = perturbations
    
    # 5. Transient response
    print("Analyzing transient characteristics...")
    transient = validator.analyze_transient_response(dex_dose=3.0, n_mc_samples=100)
    report['transient'] = transient
    
    print("Temporal validation complete!")
    
    # Save results to JSON
    import json
    json_path = os.path.join(output_dir, 'temporal_validation_results.json')
    
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
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
        else:
            return obj
    
    # Convert entire report to JSON-serializable format
    json_report = convert_to_serializable(report)
    
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    
    return report


if __name__ == "__main__":
    print("Temporal Validation Module (Deep Ensemble Version)")
    print("=" * 60)
    print("This module provides tools for:")
    print("  - Multi-timepoint simulation with uncertainty from Deep Ensemble")
    print("  - Physiological plausibility validation")
    print("  - ODE integration comparison")
    print("  - Perturbation response analysis")
    print("  - Transient characteristic analysis")
    print("\nUsage:")
    print("  from src.temporal_validation import TemporalValidator")
    print("  validator = TemporalValidator(trained_ensemble_model)")
    print("  results = validator.simulate_time_course([0, 3, 30])")