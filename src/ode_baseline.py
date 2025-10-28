"""
Traditional ODE-based model for glucocorticoid regulation of renin
Used as baseline for PINN comparison
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ODEParameters:
    """Parameters for the ODE system"""
    k_synth_renin: float = 0.1      # Renin synthesis rate
    k_deg_renin: float = 0.05       # Renin degradation rate
    k_synth_GR: float = 0.08        # GR synthesis rate
    k_deg_GR: float = 0.04          # GR degradation rate
    k_bind: float = 1.0             # Dex-GR binding rate
    k_unbind: float = 0.1           # Unbinding rate
    k_nuclear: float = 0.5          # Nuclear translocation rate
    IC50: float = 3.0               # Half-maximal inhibitory concentration
    hill: float = 2.0               # Hill coefficient
    k_translation: float = 0.3      # Translation rate
    k_secretion: float = 0.15       # Secretion rate
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to array for optimization"""
        return np.array([
            self.k_synth_renin, self.k_deg_renin,
            self.k_synth_GR, self.k_deg_GR,
            self.k_bind, self.k_unbind, self.k_nuclear,
            self.IC50, self.hill,
            self.k_translation, self.k_secretion
        ])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> 'ODEParameters':
        """Create parameters from array"""
        return cls(
            k_synth_renin=params[0],
            k_deg_renin=params[1],
            k_synth_GR=params[2],
            k_deg_GR=params[3],
            k_bind=params[4],
            k_unbind=params[5],
            k_nuclear=params[6],
            IC50=params[7],
            hill=params[8],
            k_translation=params[9],
            k_secretion=params[10]
        )

def ode_system(y: np.ndarray, t: float, dex: float, params: ODEParameters) -> np.ndarray:
    """
    ODE system for glucocorticoid regulation of renin
    
    State variables:
    y[0]: mRNA
    y[1]: protein
    y[2]: secreted renin
    y[3]: GR_free
    y[4]: GR_cytoplasm
    y[5]: GR_nucleus
    
    Args:
        y: State vector
        t: Time
        dex: Dexamethasone concentration
        params: Model parameters
        
    Returns:
        dydt: Derivatives
    """
    mRNA, protein, secreted, GR_free, GR_cyto, GR_nuc = y
    
    # Hill equation for transcriptional inhibition
    inhibition = 1.0 / (1.0 + (GR_nuc / params.IC50) ** params.hill)
    
    # Glucocorticoid receptor dynamics
    dGR_free_dt = (params.k_synth_GR - 
                   params.k_bind * dex * GR_free + 
                   params.k_unbind * GR_cyto - 
                   params.k_deg_GR * GR_free)
    
    dGR_cyto_dt = (params.k_bind * dex * GR_free - 
                   params.k_unbind * GR_cyto - 
                   params.k_nuclear * GR_cyto)
    
    dGR_nuc_dt = (params.k_nuclear * GR_cyto - 
                  params.k_deg_GR * GR_nuc)
    
    # Renin gene expression cascade
    dmRNA_dt = params.k_synth_renin * inhibition - params.k_deg_renin * mRNA
    
    dprotein_dt = (params.k_translation * mRNA - 
                   params.k_secretion * protein - 
                   params.k_deg_renin * protein)
    
    dsecreted_dt = params.k_secretion * protein
    
    return np.array([dmRNA_dt, dprotein_dt, dsecreted_dt,
                    dGR_free_dt, dGR_cyto_dt, dGR_nuc_dt])

def simulate_ode(params: ODEParameters, 
                 time_points: np.ndarray,
                 dex_concentrations: np.ndarray,
                 initial_conditions: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Simulate the ODE system for given conditions
    
    Args:
        params: Model parameters
        time_points: Time points to evaluate
        dex_concentrations: Dexamethasone concentrations for each time point
        initial_conditions: Initial state (default: baseline)
        
    Returns:
        solutions: Array of shape (n_times, 6) with state variables
    """
    if initial_conditions is None:
        # Default initial conditions (normalized baseline)
        initial_conditions = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    
    solutions = []
    
    for i, (t, dex) in enumerate(zip(time_points, dex_concentrations)):
        if i == 0:
            # First point: integrate from 0 to t
            t_span = [0, t] if t > 0 else [0, 0.01]
            sol = odeint(ode_system, initial_conditions, t_span, 
                         args=(dex, params))
            solutions.append(sol[-1])
        else:
            # Subsequent points: integrate from previous time
            t_prev = time_points[i-1]
            t_span = [t_prev, t]
            y0 = solutions[-1]
            sol = odeint(ode_system, y0, t_span, 
                         args=(dex, params))
            solutions.append(sol[-1])
    
    return np.array(solutions)

def objective_function(param_array: np.ndarray, 
                       experimental_data: Dict,
                       weight_data: float = 1.0,
                       weight_regularization: float = 0.01) -> float:
    """
    Objective function for parameter fitting
    
    Args:
        param_array: Parameters as array
        experimental_data: Dictionary with experimental measurements
        weight_data: Weight for data fitting term
        weight_regularization: Weight for regularization term
        
    Returns:
        loss: Total loss value
    """
    params = ODEParameters.from_array(param_array)
    
    # Simulate for experimental conditions
    time_points = experimental_data['time']
    dex_concentrations = experimental_data['dex_concentration']
    
    try:
        predictions = simulate_ode(params, time_points, dex_concentrations)
        secreted_pred = predictions[:, 2]  # Secreted renin
    except:
        # Return high loss if simulation fails
        return 1e10
    
    # Data fitting loss (weighted MSE)
    experimental_values = experimental_data['renin_normalized']
    
    if 'renin_std' in experimental_data:
        # Use inverse variance weighting
        weights = 1.0 / (experimental_data['renin_std']**2 + 1e-6)
        data_loss = np.mean(weights * (secreted_pred - experimental_values)**2)
    else:
        data_loss = np.mean((secreted_pred - experimental_values)**2)
    
    # Regularization to keep parameters in reasonable range
    reg_loss = np.sum(param_array**2) / len(param_array)
    
    total_loss = weight_data * data_loss + weight_regularization * reg_loss
    
    return total_loss

def fit_ode_model(experimental_data: Dict,
                  bounds: Optional[List[Tuple[float, float]]] = None,
                  method: str = 'differential_evolution',
                  maxiter: int = 1000,
                  verbose: bool = True) -> Tuple[ODEParameters, Dict]:
    """
    Fit ODE model to experimental data
    
    Args:
        experimental_data: Dictionary with experimental measurements
        bounds: Parameter bounds for optimization
        method: Optimization method ('differential_evolution' or 'L-BFGS-B')
        maxiter: Maximum iterations
        verbose: Print progress
        
    Returns:
        best_params: Fitted parameters
        results: Dictionary with fitting results
    """
    if bounds is None:
        # Simplified parameter bounds (only fit most important parameters)
        # Fix less important parameters to reasonable values
        bounds = [
            (0.05, 0.5),   # k_synth_renin - narrower range
            (0.02, 0.2),   # k_deg_renin - narrower range
            (0.05, 0.2),   # k_synth_GR - narrower range
            (0.02, 0.1),   # k_deg_GR - narrower range
            (0.5, 3.0),    # k_bind - narrower range
            (0.05, 0.5),   # k_unbind - narrower range
            (0.2, 1.0),    # k_nuclear - narrower range
            (1.0, 10.0),   # IC50 - KEY PARAMETER
            (1.5, 4.0),    # hill - KEY PARAMETER
            (0.1, 0.8),    # k_translation - narrower range
            (0.05, 0.3),   # k_secretion - narrower range
        ]
    
    if verbose:
        print("="*60)
        print("Traditional ODE Model Fitting")
        print("="*60)
        print(f"Method: {method}")
        print(f"Data points: {len(experimental_data['time'])}")
    
    start_time = time.time()
    
    if method == 'differential_evolution':
        # Global optimization with better settings
        result = differential_evolution(
            objective_function,
            bounds,
            args=(experimental_data,),
            maxiter=maxiter,
            popsize=30,  # Increased population size for better exploration
            tol=1e-10,  # Tighter tolerance
            strategy='best1bin',  # More robust strategy
            seed=42,
            mutation=(0.5, 1.5),  # Wider mutation range
            recombination=0.7,
            disp=verbose,
            workers=-1,  # Use all CPU cores
            polish=True,  # Local optimization at the end
            atol=0,
            updating='deferred'
        )
    else:
        # Local optimization with multiple random starts
        best_result = None
        best_loss = np.inf
        
        for _ in range(10):  # Multiple random starts
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            
            result = minimize(
                objective_function,
                x0,
                args=(experimental_data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': verbose}
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        
        result = best_result
    
    fitting_time = time.time() - start_time
    
    # Extract best parameters
    best_params = ODEParameters.from_array(result.x)
    
    # Calculate predictions with best parameters
    predictions = simulate_ode(
        best_params,
        experimental_data['time'],
        experimental_data['dex_concentration']
    )
    
    secreted_pred = predictions[:, 2]
    y_true = experimental_data['renin_normalized']
    
    # Calculate metrics
    ss_res = np.sum((y_true - secreted_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_true - secreted_pred)**2))
    
    # Calculate AIC and BIC
    n = len(y_true)
    k = len(result.x)  # Number of parameters
    aic = n * np.log(ss_res/n) + 2*k
    bic = n * np.log(ss_res/n) + k*np.log(n)
    
    results = {
        'parameters': best_params,
        'optimization_result': result,
        'predictions': secreted_pred,
        'full_predictions': predictions,
        'residuals': y_true - secreted_pred,
        'r_squared': r_squared,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'fitting_time': fitting_time,
        'loss': result.fun
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Fitting Results:")
        print("="*60)
        print(f"R^2 Score: {r_squared:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        print(f"Final Loss: {result.fun:.6f}")
        print(f"Fitting Time: {fitting_time:.2f} seconds")
        print(f"Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
        print("\n" + "="*60)
        print("Fitted Parameters:")
        print("="*60)
        print(f"IC50: {best_params.IC50:.3f} mg/dl")
        print(f"Hill coefficient: {best_params.hill:.3f}")
        print(f"k_synth_renin: {best_params.k_synth_renin:.4f} h^-1")
        print(f"k_deg_renin: {best_params.k_deg_renin:.4f} h^-1")
        print(f"k_translation: {best_params.k_translation:.4f} h^-1")
        print(f"k_secretion: {best_params.k_secretion:.4f} h^-1")
        print("="*60)
    
    return best_params, results

def generate_dose_response_curve(params: ODEParameters,
                                 dex_range: np.ndarray,
                                 time: float = 24.0) -> np.ndarray:
    """
    Generate dose-response curve at specified time
    
    Args:
        params: Model parameters
        dex_range: Range of dexamethasone concentrations
        time: Time point (default: 24 hours)
        
    Returns:
        renin_levels: Secreted renin levels for each concentration
    """
    renin_levels = []
    
    for dex in dex_range:
        # Simulate to specified time
        sol = simulate_ode(params, np.array([time]), np.array([dex]))
        renin_levels.append(sol[0, 2])  # Secreted renin
    
    return np.array(renin_levels)

def generate_time_course(params: ODEParameters,
                        time_points: np.ndarray,
                        dex_concentration: float) -> np.ndarray:
    """
    Generate time course for a specific dexamethasone concentration
    
    Args:
        params: Model parameters
        time_points: Time points to evaluate
        dex_concentration: Fixed dexamethasone concentration
        
    Returns:
        solutions: Full state evolution over time
    """
    # Constant dex concentration
    dex_array = np.full_like(time_points, dex_concentration)
    
    return simulate_ode(params, time_points, dex_array)

def bootstrap_confidence_intervals(experimental_data: Dict,
                                   n_bootstrap: int = 100,
                                   confidence_level: float = 0.95,
                                   verbose: bool = True) -> Dict:
    """
    Calculate confidence intervals using bootstrap
    
    Args:
        experimental_data: Original experimental data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 95%)
        verbose: Print progress
        
    Returns:
        ci_results: Dictionary with confidence intervals
    """
    n_data = len(experimental_data['time'])
    
    # Store bootstrap results
    ic50_values = []
    hill_values = []
    r2_values = []
    rmse_values = []
    
    if verbose:
        print(f"Running bootstrap with {n_bootstrap} samples...")
    
    for i in range(n_bootstrap):
        # Resample data with replacement
        indices = np.random.choice(n_data, size=n_data, replace=True)
        
        bootstrap_data = {
            'time': experimental_data['time'][indices],
            'dex_concentration': experimental_data['dex_concentration'][indices],
            'renin_normalized': experimental_data['renin_normalized'][indices]
        }
        
        if 'renin_std' in experimental_data:
            bootstrap_data['renin_std'] = experimental_data['renin_std'][indices]
        
        # Fit model to bootstrap sample
        # Use single-threaded optimization to avoid nested multiprocessing issues
        try:
            # Use differential evolution with workers=1 for bootstrap
            bounds = [
                (0.01, 1.0),   # k_synth_renin
                (0.01, 0.5),   # k_deg_renin
                (0.01, 0.5),   # k_synth_GR
                (0.01, 0.3),   # k_deg_GR
                (0.1, 5.0),    # k_bind
                (0.01, 1.0),   # k_unbind
                (0.1, 2.0),    # k_nuclear
                (0.5, 10.0),   # IC50
                (1.0, 4.0),    # hill
                (0.05, 1.0),   # k_translation
                (0.01, 0.5),   # k_secretion
            ]
            
            result = differential_evolution(
                objective_function,
                bounds,
                args=(bootstrap_data,),
                maxiter=500,  # Fewer iterations for bootstrap
                popsize=10,
                tol=1e-6,
                seed=42+i,
                disp=False,
                workers=1  # Single-threaded to avoid nested multiprocessing
            )
            
            params = ODEParameters.from_array(result.x)
            
            # Calculate R2 and RMSE
            predictions = simulate_ode(
                params,
                bootstrap_data['time'],
                bootstrap_data['dex_concentration']
            )
            
            secreted_pred = predictions[:, 2]
            y_true = bootstrap_data['renin_normalized']
            
            ss_res = np.sum((y_true - secreted_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean((y_true - secreted_pred)**2))
            
            ic50_values.append(params.IC50)
            hill_values.append(params.hill)
            r2_values.append(r_squared)
            rmse_values.append(rmse)
        except:
            continue  # Skip failed fits
        
        if verbose and (i+1) % 20 == 0:
            print(f"  Completed {i+1}/{n_bootstrap} bootstrap samples")
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_results = {
        'ic50_mean': np.mean(ic50_values),
        'ic50_std': np.std(ic50_values),
        'ic50_ci': np.percentile(ic50_values, [lower_percentile, upper_percentile]),
        'hill_mean': np.mean(hill_values),
        'hill_std': np.std(hill_values),
        'hill_ci': np.percentile(hill_values, [lower_percentile, upper_percentile]),
        'r2_mean': np.mean(r2_values),
        'r2_std': np.std(r2_values),
        'r2_ci': np.percentile(r2_values, [lower_percentile, upper_percentile]),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'rmse_ci': np.percentile(rmse_values, [lower_percentile, upper_percentile]),
        'n_successful': len(ic50_values),
        'n_bootstrap': n_bootstrap
    }
    
    if verbose:
        print("\n" + "="*60)
        print(f"Bootstrap Results ({confidence_level*100:.0f}% CI):")
        print("="*60)
        print(f"IC50: {ci_results['ic50_mean']:.2f} +/- {ci_results['ic50_std']:.2f} mg/dl")
        print(f"     CI: [{ci_results['ic50_ci'][0]:.2f}, {ci_results['ic50_ci'][1]:.2f}]")
        print(f"Hill: {ci_results['hill_mean']:.2f} +/- {ci_results['hill_std']:.2f}")
        print(f"     CI: [{ci_results['hill_ci'][0]:.2f}, {ci_results['hill_ci'][1]:.2f}]")
        print(f"R^2: {ci_results['r2_mean']:.4f} +/- {ci_results['r2_std']:.4f}")
        print(f"RMSE: {ci_results['rmse_mean']:.4f} +/- {ci_results['rmse_std']:.4f}")
        print(f"Successful fits: {ci_results['n_successful']}/{n_bootstrap}")
        print("="*60)
    
    return ci_results

if __name__ == "__main__":
    # Test with sample data
    from data import prepare_training_data
    
    print("Loading experimental data...")
    data = prepare_training_data(dataset='elisa', use_log_scale=False)
    
    print("\nFitting traditional ODE model...")
    params, results = fit_ode_model(data, method='differential_evolution')
    
    print("\nGenerating dose-response curve...")
    dex_range = np.logspace(-2, 2, 100)
    dose_response = generate_dose_response_curve(params, dex_range)
    
    print("\nRunning bootstrap for confidence intervals...")
    ci_results = bootstrap_confidence_intervals(data, n_bootstrap=50)
    
    print("\n" + "="*60)
    print("Traditional ODE baseline ready for comparison!")
    print("Expected R^2 approx 0.891 (vs PINN R^2 approx 0.964)")
    print("="*60)