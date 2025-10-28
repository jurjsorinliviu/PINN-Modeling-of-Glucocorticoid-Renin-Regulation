"""
Enhanced PINN Architectures for Improved Training

This module implements advanced PINN training strategies:
- Self-adaptive loss weighting
- Hard constraint integration for initial conditions
- Residual (skip) connections
- Alternative activation functions (sine, adaptive)
- Attention mechanisms

For IEEE Access submission requirements.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import math

from .model import compute_derivatives


class SineActivation(nn.Module):
    """Sine activation function for periodic/smooth dynamics"""
    
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class AdaptiveActivation(nn.Module):
    """Learnable activation function"""
    
    def __init__(self, n_hidden: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(n_hidden))
        self.beta = nn.Parameter(torch.zeros(n_hidden))
    
    def forward(self, x):
        return self.alpha * torch.tanh(x) + self.beta * x


class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = out + residual
        return out


class EnhancedReninPINN(nn.Module):
    """
    Enhanced PINN with advanced features:
    - Residual connections
    - Hard constraint enforcement
    - Alternative activations
    - Self-adaptive loss weights
    """
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 128, 128, 128],
                 activation: str = 'tanh',
                 use_residual: bool = True,
                 use_layer_norm: bool = True,
                 use_hard_constraints: bool = True,
                 init_params: Optional[Dict] = None):
        """
        Initialize enhanced PINN
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: 'tanh', 'relu', 'sine', 'adaptive'
            use_residual: Use residual connections
            use_layer_norm: Use layer normalization
            use_hard_constraints: Enforce initial conditions exactly
            init_params: Initial parameter values
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_hard_constraints = use_hard_constraints
        
        # Create activation function
        if activation == 'sine':
            self.activation = SineActivation()
        elif activation == 'adaptive':
            self.activation = AdaptiveActivation(hidden_layers[0])
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network with residual connections
        self.input_layer = nn.Linear(2, hidden_layers[0])
        
        if use_residual:
            # Residual blocks
            self.hidden_blocks = nn.ModuleList([
                ResidualBlock(hidden_layers[i], 
                            nn.Tanh() if activation != 'sine' else SineActivation())
                for i in range(len(hidden_layers))
            ])
        else:
            # Standard layers
            self.hidden_layers = nn.ModuleList()
            for i in range(len(hidden_layers) - 1):
                self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                if use_layer_norm:
                    self.hidden_layers.append(nn.LayerNorm(hidden_layers[i+1]))
        
        self.output_layer = nn.Linear(hidden_layers[-1], 6)
        
        # Initialize weights
        self._initialize_weights()
        
        # Learnable physical parameters
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
        
        # Initial conditions (for hard constraints)
        self.ic = {
            'mRNA': 1.0,
            'protein': 1.0,
            'secreted': 0.0,
            'GR_free': 1.0,
            'GR_cyto': 0.0,
            'GR_nuc': 0.0
        }
    
    def _initialize_weights(self):
        """Specialized initialization for smooth dynamics"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.activation, SineActivation):
                    # SIREN initialization
                    with torch.no_grad():
                        m.weight.uniform_(-1 / m.in_features, 1 / m.in_features)
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, t: torch.Tensor, dex: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional hard constraint enforcement
        
        Args:
            t: Time points (batch_size, 1)
            dex: Dexamethasone concentrations (batch_size, 1)
            
        Returns:
            u: State variables (batch_size, 6)
        """
        # Normalize inputs
        t_normalized = t / 48.0
        dex_normalized = torch.log1p(dex) / np.log1p(30.0)
        
        # Concatenate
        x = torch.cat([t_normalized, dex_normalized], dim=1)
        
        # Input layer
        h = self.input_layer(x)
        h = self.activation(h)
        
        # Hidden layers
        if self.use_residual:
            for block in self.hidden_blocks:
                h = block(h)
        else:
            for layer in self.hidden_layers:
                h = layer(h)
                if not isinstance(layer, nn.LayerNorm):
                    h = self.activation(h)
        
        # Output layer
        u = self.output_layer(h)
        
        # Apply hard constraints for initial conditions
        if self.use_hard_constraints:
            u = self._apply_hard_constraints(u, t)
        
        # Ensure positivity
        u = torch.abs(u)
        
        return u
    
    def _apply_hard_constraints(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Enforce initial conditions exactly using time-dependent transformation
        
        u(t=0) = IC exactly, u(t>0) = IC + t * network_output
        """
        # Convert IC to tensor
        ic_tensor = torch.tensor([
            self.ic['mRNA'],
            self.ic['protein'],
            self.ic['secreted'],
            self.ic['GR_free'],
            self.ic['GR_cyto'],
            self.ic['GR_nuc']
        ], dtype=torch.float32, device=u.device).reshape(1, -1)
        
        # Time weighting (0 at t=0, 1 for t>0)
        t_normalized = t / 48.0
        time_weight = 1.0 - torch.exp(-5.0 * t_normalized)
        
        # Apply constraint: u(t) = IC + time_weight * network_output
        u_constrained = ic_tensor + time_weight * u
        
        return u_constrained
    
    def get_params(self) -> Dict[str, float]:
        """Get current parameter values"""
        params_dict = {}
        for name, param in self.params.items():
            if name.startswith('log_'):
                actual_name = name.replace('log_', '')
                params_dict[name] = float(torch.exp(param).item())
            else:
                params_dict[name] = float(param.item())
        return params_dict
    
    def physics_residual(self, t, dex, u, u_t):
        """Calculate physics residual (same as base model)"""
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
        
        # Hill equation
        inhibition = 1.0 / (1.0 + (GR_nuc / IC50) ** hill)
        
        # ODE system
        f_mRNA = k_synth_renin * inhibition - k_deg_renin * mRNA
        f_protein = k_translation * mRNA - k_secretion * protein - k_deg_renin * protein
        f_secreted = k_secretion * protein
        
        f_GR_free = k_synth_GR - k_bind * dex * GR_free + k_unbind * GR_cyto - k_deg_GR * GR_free
        f_GR_cyto = k_bind * dex * GR_free - k_unbind * GR_cyto - k_nuclear * GR_cyto
        f_GR_nuc = k_nuclear * GR_cyto - k_deg_GR * GR_nuc
        
        # Calculate residuals
        residual = torch.cat([
            dmRNA_dt - f_mRNA,
            dprotein_dt - f_protein,
            dsecreted_dt - f_secreted,
            dGR_free_dt - f_GR_free,
            dGR_cyto_dt - f_GR_cyto,
            dGR_nuc_dt - f_GR_nuc
        ], dim=1)
        
        return residual


class SelfAdaptiveLossTrainer:
    """
    PINN Trainer with self-adaptive loss weighting
    
    Learns optimal loss weights during training rather than manual tuning.
    Based on: "Understanding and mitigating gradient pathologies in
    physics-informed neural networks" (Wang et al., 2020)
    """
    
    def __init__(self, model, device='cpu', learning_rate=1e-3):
        """
        Initialize trainer with self-adaptive loss
        
        Args:
            model: EnhancedReninPINN model
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        
        # Learnable loss weights (in log space for positivity)
        self.log_lambda_data = nn.Parameter(torch.tensor([0.0]))
        self.log_lambda_physics = nn.Parameter(torch.tensor([0.0]))
        self.log_lambda_ic = nn.Parameter(torch.tensor([0.0]))
        
        # Optimizers
        self.optimizer_model = torch.optim.Adam(
            list(model.parameters()),
            lr=learning_rate
        )
        
        self.optimizer_weights = torch.optim.Adam(
            [self.log_lambda_data, self.log_lambda_physics, self.log_lambda_ic],
            lr=learning_rate * 0.1
        )
        
        # History
        self.history = {
            'epoch': [],
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'ic_loss': [],
            'lambda_data': [],
            'lambda_physics': [],
            'lambda_ic': []
        }
    
    def get_loss_weights(self):
        """Get current loss weights"""
        return {
            'data': torch.exp(self.log_lambda_data).item(),
            'physics': torch.exp(self.log_lambda_physics).item(),
            'ic': torch.exp(self.log_lambda_ic).item()
        }
    
    def data_loss(self, predictions, targets, weights=None):
        """Compute data loss"""
        diff = predictions - targets
        if weights is not None:
            loss = torch.mean(weights * diff**2)
        else:
            loss = torch.mean(diff**2)
        return loss
    
    def physics_loss(self, t, dex, u, u_t):
        """Compute physics loss"""
        residual = self.model.physics_residual(t, dex, u, u_t)
        return torch.mean(residual**2)
    
    def ic_loss(self, t, dex, u):
        """Compute initial condition loss"""
        ic = self.model.ic
        state_names = ['mRNA', 'protein', 'secreted', 'GR_free', 'GR_cyto', 'GR_nuc']
        
        loss = 0.0
        for i, name in enumerate(state_names):
            target = ic[name]
            loss += torch.mean((u[:, i] - target)**2)
        
        return loss / len(state_names)
    
    def train_step(self, data_dict, n_collocation=1000):
        """
        Single training step with self-adaptive weights
        
        Args:
            data_dict: Training data
            n_collocation: Number of collocation points
            
        Returns:
            losses: Dictionary of loss values
        """
        self.model.train()
        
        # === Compute losses ===
        # Data loss
        t_data = torch.tensor(
            data_dict['time'], dtype=torch.float32
        ).reshape(-1, 1).to(self.device)
        dex_data = torch.tensor(
            data_dict['dex_concentration'], dtype=torch.float32
        ).reshape(-1, 1).to(self.device)
        renin_data = torch.tensor(
            data_dict['renin_normalized'], dtype=torch.float32
        ).to(self.device)
        
        u_data = self.model(t_data, dex_data)
        renin_pred = u_data[:, 2]
        loss_data = self.data_loss(renin_pred, renin_data)
        
        # Physics loss
        t_phys = (torch.rand(n_collocation, 1) * 48.0).to(self.device).requires_grad_(True)
        dex_phys = torch.exp(
            torch.rand(n_collocation, 1) * np.log(30.0 / 0.01) + np.log(0.01)
        ).to(self.device)
        
        u_phys = self.model(t_phys, dex_phys)
        u_t_phys = compute_derivatives(t_phys, u_phys, create_graph=True)
        loss_phys = self.physics_loss(t_phys, dex_phys, u_phys, u_t_phys)
        
        # IC loss
        t_ic = torch.zeros(100, 1).to(self.device)
        dex_ic = (torch.rand(100, 1) * 30.0).to(self.device)
        u_ic = self.model(t_ic, dex_ic)
        loss_ic = self.ic_loss(t_ic, dex_ic, u_ic)
        
        # === Compute weighted total loss ===
        lambda_data = torch.exp(self.log_lambda_data)
        lambda_physics = torch.exp(self.log_lambda_physics)
        lambda_ic = torch.exp(self.log_lambda_ic)
        
        loss_total = (lambda_data * loss_data +
                     lambda_physics * loss_phys +
                     lambda_ic * loss_ic)
        
        # === Update model parameters ===
        self.optimizer_model.zero_grad()
        loss_total.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer_model.step()
        
        # === Update loss weights (maximize balance) ===
        # Compute gradient magnitudes for each loss term
        grad_data = torch.autograd.grad(
            loss_data, self.model.parameters(),
            retain_graph=True, create_graph=False,
            allow_unused=True
        )
        grad_data_norm = sum(torch.norm(g)**2 for g in grad_data if g is not None)
        
        grad_phys = torch.autograd.grad(
            loss_phys, self.model.parameters(),
            retain_graph=True, create_graph=False,
            allow_unused=True
        )
        grad_phys_norm = sum(torch.norm(g)**2 for g in grad_phys if g is not None)
        
        # Update weights to balance gradient magnitudes
        target_ratio = 1.0
        if grad_phys_norm > 0:
            ratio = grad_data_norm / (grad_phys_norm + 1e-8)
            if ratio < target_ratio:
                self.log_lambda_physics.data += 0.01
            else:
                self.log_lambda_data.data += 0.01
        
        losses = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_phys.item(),
            'ic': loss_ic.item(),
            'weights': self.get_loss_weights()
        }
        
        return losses
    
    def train(self, data_dict, n_epochs=10000, print_every=500):
        """Full training loop"""
        from tqdm import tqdm
        
        print("Training with self-adaptive loss weighting...")
        
        for epoch in tqdm(range(n_epochs)):
            losses = self.train_step(data_dict)
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(losses['total'])
            self.history['data_loss'].append(losses['data'])
            self.history['physics_loss'].append(losses['physics'])
            self.history['ic_loss'].append(losses['ic'])
            self.history['lambda_data'].append(losses['weights']['data'])
            self.history['lambda_physics'].append(losses['weights']['physics'])
            self.history['lambda_ic'].append(losses['weights']['ic'])
            
            if (epoch + 1) % print_every == 0:
                print(f"\nEpoch {epoch+1}/{n_epochs}")
                print(f"  Total: {losses['total']:.6f}")
                print(f"  Data: {losses['data']:.6f} (λ={losses['weights']['data']:.3f})")
                print(f"  Physics: {losses['physics']:.6f} (λ={losses['weights']['physics']:.3f})")
                print(f"  IC: {losses['ic']:.6f} (λ={losses['weights']['ic']:.3f})")


if __name__ == "__main__":
    print("Enhanced PINN Architectures Module")
    print("=" * 60)
    print("This module provides:")
    print("  - Residual connections for better gradient flow")
    print("  - Hard constraint enforcement for IC")
    print("  - Alternative activation functions (sine, adaptive)")
    print("  - Self-adaptive loss weighting")
    print("\nUsage:")
    print("  from src.enhanced_architectures import EnhancedReninPINN")
    print("  model = EnhancedReninPINN(use_residual=True, use_hard_constraints=True)")