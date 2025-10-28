"""
Physics-Informed Neural Network for Glucocorticoid Regulation of Renin

This module implements the PINN architecture with physics constraints
from the ODE system governing renin expression and GR dynamics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional

class ReninPINN(nn.Module):
    """
    Physics-Informed Neural Network for modeling glucocorticoid regulation of renin
    
    The network learns to predict:
    - mRNA levels
    - Protein levels  
    - Secreted renin
    - GR_free (free glucocorticoid receptor)
    - GR_cytoplasm (cytoplasmic GR-dex complex)
    - GR_nucleus (nuclear GR-dex complex)
    
    While respecting physical constraints from the ODE system.
    """
    
    def __init__(self, 
                 hidden_layers: List[int] = [128, 128, 128, 128],
                 activation: str = 'tanh',
                 init_params: Optional[Dict] = None):
        """
        Initialize PINN architecture
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'sigmoid')
            init_params: Initial parameter values (optional)
        """
        super(ReninPINN, self).__init__()
        
        # Network architecture
        layers = []
        input_dim = 2  # time and dex concentration
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            input_dim = hidden_dim
        
        # Output layer: 6 state variables
        layers.append(nn.Linear(input_dim, 6))
        
        self.network = nn.Sequential(*layers)
        self.layers = hidden_layers
        
        # Initialize weights
        self._initialize_weights()
        
        # Learnable physical parameters (in log space for positivity)
        if init_params is None:
            self.params = nn.ParameterDict({
                'log_IC50': nn.Parameter(torch.tensor([np.log(3.0)], dtype=torch.float32)),
                'log_hill': nn.Parameter(torch.tensor([np.log(2.0)], dtype=torch.float32)),
                'log_k_synth_renin': nn.Parameter(torch.tensor([np.log(0.1)], dtype=torch.float32)),
                'log_k_deg_renin': nn.Parameter(torch.tensor([np.log(0.05)], dtype=torch.float32)),
                'log_k_synth_GR': nn.Parameter(torch.tensor([np.log(0.08)], dtype=torch.float32)),
                'log_k_deg_GR': nn.Parameter(torch.tensor([np.log(0.04)], dtype=torch.float32)),
                'log_k_bind': nn.Parameter(torch.tensor([np.log(1.0)], dtype=torch.float32)),
                'log_k_unbind': nn.Parameter(torch.tensor([np.log(0.1)], dtype=torch.float32)),
                'log_k_nuclear': nn.Parameter(torch.tensor([np.log(0.5)], dtype=torch.float32)),
                'log_k_translation': nn.Parameter(torch.tensor([np.log(0.3)], dtype=torch.float32)),
                'log_k_secretion': nn.Parameter(torch.tensor([np.log(0.15)], dtype=torch.float32))
            })
        else:
            self.params = nn.ParameterDict(init_params)
    
    def _initialize_weights(self):
        """Xavier initialization for network weights"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, t: torch.Tensor, dex: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            t: Time points (batch_size, 1)
            dex: Dexamethasone concentrations (batch_size, 1)
            
        Returns:
            u: State variables (batch_size, 6)
               [mRNA, protein, secreted, GR_free, GR_cyto, GR_nuc]
        """
        # Normalize inputs
        t_normalized = t / 48.0  # Normalize to [0, 1]
        dex_normalized = torch.log1p(dex) / np.log1p(30.0)  # Log-normalize
        
        # Concatenate inputs
        x = torch.cat([t_normalized, dex_normalized], dim=1)
        
        # Forward pass
        u = self.network(x)
        
        # Apply activation to ensure positivity
        u = torch.abs(u)  # Simple positivity constraint
        
        return u
    
    def get_params(self) -> Dict[str, float]:
        """
        Get current parameter values
        
        Returns:
            params_dict: Dictionary of parameter values
        """
        params_dict = {}
        for name, param in self.params.items():
            if name.startswith('log_'):
                # Convert from log space
                actual_name = name.replace('log_', '')
                params_dict[name] = float(torch.exp(param).item())
            else:
                params_dict[name] = float(param.item())
        
        return params_dict
    
    def physics_residual(self, 
                        t: torch.Tensor,
                        dex: torch.Tensor,
                        u: torch.Tensor,
                        u_t: torch.Tensor) -> torch.Tensor:
        """
        Calculate physics residual (ODE violation)
        
        Args:
            t: Time points
            dex: Dexamethasone concentrations
            u: State variables
            u_t: Time derivatives
            
        Returns:
            residual: ODE residual (should be close to 0)
        """
        # Extract state variables
        mRNA = u[:, 0:1]
        protein = u[:, 1:2]
        secreted = u[:, 2:3]
        GR_free = u[:, 3:4]
        GR_cyto = u[:, 4:5]
        GR_nuc = u[:, 5:6]
        
        # Extract derivatives
        dmRNA_dt = u_t[:, 0:1]
        dprotein_dt = u_t[:, 1:2]
        dsecreted_dt = u_t[:, 2:3]
        dGR_free_dt = u_t[:, 3:4]
        dGR_cyto_dt = u_t[:, 4:5]
        dGR_nuc_dt = u_t[:, 5:6]
        
        # Get parameters
        IC50 = torch.exp(self.params['log_IC50'])
        hill = torch.exp(self.params['log_hill'])
        k_synth_renin = torch.exp(self.params['log_k_synth_renin'])
        k_deg_renin = torch.exp(self.params['log_k_deg_renin'])
        k_synth_GR = torch.exp(self.params['log_k_synth_GR'])
        k_deg_GR = torch.exp(self.params['log_k_deg_GR'])
        k_bind = torch.exp(self.params['log_k_bind'])
        k_unbind = torch.exp(self.params['log_k_unbind'])
        k_nuclear = torch.exp(self.params['log_k_nuclear'])
        k_translation = torch.exp(self.params['log_k_translation'])
        k_secretion = torch.exp(self.params['log_k_secretion'])
        
        # Hill equation for transcriptional inhibition
        inhibition = 1.0 / (1.0 + (GR_nuc / IC50) ** hill)
        
        # ODE system
        f_mRNA = k_synth_renin * inhibition - k_deg_renin * mRNA
        f_protein = k_translation * mRNA - k_secretion * protein - k_deg_renin * protein
        f_secreted = k_secretion * protein
        
        f_GR_free = k_synth_GR - k_bind * dex * GR_free + k_unbind * GR_cyto - k_deg_GR * GR_free
        f_GR_cyto = k_bind * dex * GR_free - k_unbind * GR_cyto - k_nuclear * GR_cyto
        f_GR_nuc = k_nuclear * GR_cyto - k_deg_GR * GR_nuc
        
        # Calculate residuals
        residual_mRNA = dmRNA_dt - f_mRNA
        residual_protein = dprotein_dt - f_protein
        residual_secreted = dsecreted_dt - f_secreted
        residual_GR_free = dGR_free_dt - f_GR_free
        residual_GR_cyto = dGR_cyto_dt - f_GR_cyto
        residual_GR_nuc = dGR_nuc_dt - f_GR_nuc
        
        # Concatenate all residuals
        residual = torch.cat([
            residual_mRNA, residual_protein, residual_secreted,
            residual_GR_free, residual_GR_cyto, residual_GR_nuc
        ], dim=1)
        
        return residual
    
    def initial_conditions(self) -> Dict[str, float]:
        """
        Define initial conditions for the ODE system
        
        Returns:
            ic: Dictionary of initial conditions
        """
        return {
            'mRNA': 1.0,        # Normalized baseline
            'protein': 1.0,     # Normalized baseline
            'secreted': 0.0,    # No initial secretion
            'GR_free': 1.0,     # Normalized baseline
            'GR_cyto': 0.0,     # No initial complex
            'GR_nuc': 0.0       # No initial nuclear translocation
        }


class BayesianReninPINN(ReninPINN):
    """
    Bayesian PINN with Monte Carlo Dropout for uncertainty quantification
    """
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 128, 128, 128],
                 activation: str = 'tanh',
                 dropout_rate: float = 0.1,
                 init_params: Optional[Dict] = None):
        """
        Initialize Bayesian PINN with dropout
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout probability
            init_params: Initial parameter values
        """
        super().__init__(hidden_layers, activation, init_params)
        
        self.dropout_rate = dropout_rate
        
        # Rebuild network with dropout layers
        layers = []
        input_dim = 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            # Add dropout after activation
            layers.append(nn.Dropout(p=dropout_rate))
            
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 6))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def forward(self, t: torch.Tensor, dex: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        """
        Forward pass with optional MC dropout
        
        Args:
            t: Time points
            dex: Dexamethasone concentrations
            mc_dropout: Keep dropout active even during eval mode
            
        Returns:
            u: State variables
        """
        # Normalize inputs
        t_normalized = t / 48.0
        dex_normalized = torch.log1p(dex) / np.log1p(30.0)
        
        x = torch.cat([t_normalized, dex_normalized], dim=1)
        
        # If MC dropout, enable training mode temporarily
        if mc_dropout:
            self.network.train()
            u = self.network(x)
            self.network.eval()
        else:
            u = self.network(x)
        
        u = torch.abs(u)
        
        return u
    
    def predict_with_uncertainty(self,
                                t: np.ndarray,
                                dex: np.ndarray,
                                n_samples: int = 100,
                                device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty using MC dropout
        
        Args:
            t: Time points
            dex: Dexamethasone concentrations
            n_samples: Number of MC samples
            device: torch device
            
        Returns:
            mean: Mean predictions
            std: Standard deviation
            samples: All MC samples
        """
        self.eval()
        
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1).to(device)
        dex_tensor = torch.tensor(dex, dtype=torch.float32).reshape(-1, 1).to(device)
        
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(t_tensor, dex_tensor, mc_dropout=True)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std, predictions


def compute_derivatives(t: torch.Tensor, 
                       u: torch.Tensor,
                       create_graph: bool = True) -> torch.Tensor:
    """
    Compute time derivatives using automatic differentiation
    
    Args:
        t: Time points (requires_grad=True)
        u: State variables
        create_graph: Whether to create computation graph for higher-order derivatives
        
    Returns:
        u_t: Time derivatives du/dt
    """
    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True
    )[0]
    
    return u_t


if __name__ == "__main__":
    print("Testing PINN Model...")
    
    # Test standard PINN
    model = ReninPINN(hidden_layers=[128, 128, 128, 128])
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    t = torch.tensor([[24.0]], dtype=torch.float32, requires_grad=True)
    dex = torch.tensor([[3.0]], dtype=torch.float32)
    
    u = model(t, dex)
    print(f"Output shape: {u.shape}")
    print(f"State variables: {u}")
    
    # Test derivative computation
    u_t = compute_derivatives(t, u)
    print(f"Time derivatives shape: {u_t.shape}")
    
    # Test physics residual
    residual = model.physics_residual(t, dex, u, u_t)
    print(f"Physics residual: {residual}")
    
    # Test Bayesian PINN
    print("\nTesting Bayesian PINN...")
    bayesian_model = BayesianReninPINN(dropout_rate=0.1)
    
    mean, std, samples = bayesian_model.predict_with_uncertainty(
        np.array([24.0]),
        np.array([3.0]),
        n_samples=10
    )
    print(f"Mean prediction: {mean}")
    print(f"Uncertainty (std): {std}")
    
    print("\nModel tests completed successfully!")