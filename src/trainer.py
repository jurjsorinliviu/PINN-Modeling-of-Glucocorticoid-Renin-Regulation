"""
Training logic for Physics-Informed Neural Network
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .model import compute_derivatives
from .statistical_utils import calculate_metrics


@dataclass
class UnifiedTrainingConfig:
    """Configuration for the unified PINN training loop."""
    n_epochs: int = 2000
    print_every: int = 200
    loss_data: float = 1.0
    loss_physics: float = 5.0
    loss_ic: float = 0.5
    loss_parameter: float = 0.01
    loss_synthetic: float = 0.0
    loss_biological: float = 35.0
    physics_ramp_fraction: float = 0.10
    collocation_points: int = 512
    ic_points: int = 128
    synthetic_samples_per_epoch: int = 24
    synthetic_noise_std: float = 0.03
    max_grad_norm: float = 1.0
    monotonic_time_points: Tuple[float, ...] = ()
    monotonic_data_weight: float = 25.0
    monotonic_sample_count: int = 120
    monotonic_tolerance: float = 0.0
    high_dose_times: Tuple[float, ...] = (24.0, 36.0)
    high_dose_weight: float = 35.0
    monotonic_gradient_weight: float = 10.0
    temporal_suppression_weight: float = 20.0
    temporal_suppression_head_fraction: float = 0.15
    temporal_suppression_tail_fraction: float = 0.30
    temporal_derivative_weight: float = 8.0
    temporal_derivative_tolerance: float = 0.0
    biological_ramp_fraction: float = 0.15


@dataclass
class PlausibilityConfig:
    """Runtime plausibility checks evaluated during training."""
    doses: List[float] = field(default_factory=lambda: [0.0, 0.3, 3.0, 30.0])
    time_start: float = 0.0
    time_end: float = 48.0
    n_points: int = 120
    derivative_threshold: float = 0.12
    max_value: float = 1.6
    steady_state_window: int = 12
    steady_state_std: float = 0.05
    suppression_tolerance: float = 0.02  # Tolerance for temporal and dose-wise suppression checks


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping triggered by metrics or plausibility."""
    patience: int = 20
    min_epochs: int = 300
    r2_tolerance: float = 0.0025
    plausibility_patience: int = 5

class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Network
    
    Implements multi-objective loss function:
    L = λ_data * L_data + λ_physics * L_physics + λ_ic * L_ic
    
    where:
    - L_data: Mean squared error against experimental data
    - L_physics: ODE residual (physics violation)
    - L_ic: Initial condition violation
    """
    
    def __init__(self,
                 model,
                 device='cpu',
                 learning_rate=1e-3,
                 weight_decay=0.01):  # FIXED: Increased from 1e-5 to 0.01 for better regularization
        """
        Initialize trainer
        
        Args:
            model: ReninPINN model
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization (increased to prevent overfitting)
        """
        self.model = model.to(device)
        self.device = device
        
        # Separate learning rates for network and parameters
        self.optimizer = Adam([
            {'params': model.network.parameters(), 'lr': learning_rate},
            {'params': model.params.values(), 'lr': learning_rate * 0.1}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=500,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'ic_loss': [],
            'learning_rate': [],
            'parameters': []
        }
        
    def generate_collocation_points(self, n_points=1000):
        """
        Generate random collocation points for physics loss
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            t, dex: Tensors of collocation points
        """
        # Sample uniformly in time [0, 48] hours
        t = torch.rand(n_points, 1) * 48.0
        
        # Sample log-uniformly in dexamethasone [0.01, 30] mg/dl
        dex = torch.exp(torch.rand(n_points, 1) * np.log(30.0 / 0.01) + np.log(0.01))
        
        return t, dex
    
    def generate_ic_points(self, n_points=100):
        """
        Generate points at t=0 for initial conditions
        """
        t = torch.zeros(n_points, 1)
        dex = torch.rand(n_points, 1) * 30.0
        return t, dex
    
    def data_loss(self, predictions, targets, weights=None):
        """
        Compute loss on experimental data
        
        Args:
            predictions: Model predictions
            targets: Experimental measurements
            weights: Optional weights (inverse variance)
            
        Returns:
            loss: Weighted MSE
        """
        diff = predictions - targets
        
        if weights is not None:
            loss = torch.mean(weights * diff**2)
        else:
            loss = torch.mean(diff**2)
        
        return loss
    
    def physics_loss(self, t, dex, u, u_t):
        """
        Compute physics loss (ODE residual)
        """
        residual = self.model.physics_residual(t, dex, u, u_t)
        loss = torch.mean(residual**2)
        return loss
    
    def ic_loss(self, t, dex, u):
        """
        Compute initial condition loss
        """
        ic = self.model.initial_conditions()
        
        loss = 0.0
        state_names = ['mRNA', 'protein', 'secreted', 'GR_free', 'GR_cyto', 'GR_nuc']
        
        for i, name in enumerate(state_names):
            target = ic[name]
            loss += torch.mean((u[:, i] - target)**2)
        
        return loss / len(state_names)
    
    def train_step(self,
                   data_dict,
                   loss_weights={'data': 1.0, 'physics': 10.0, 'ic': 5.0},  # FIXED: Increased physics weight 100x
                   n_collocation=1000):
        """
        Single training step
        
        Args:
            data_dict: Dictionary with experimental data
            loss_weights: Weights for different loss components
            n_collocation: Number of collocation points
            
        Returns:
            losses: Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # === Data Loss ===
        t_data = torch.tensor(data_dict['time'], dtype=torch.float32).reshape(-1, 1).to(self.device)
        dex_data = torch.tensor(data_dict['dex_concentration'], dtype=torch.float32).reshape(-1, 1).to(self.device)
        renin_data = torch.tensor(data_dict['renin_normalized'], dtype=torch.float32).to(self.device)
        renin_std = torch.tensor(data_dict['renin_std'], dtype=torch.float32).to(self.device)
        
        u_data = self.model(t_data, dex_data)
        renin_pred = u_data[:, 2]  # Secreted renin
        
        # Inverse variance weighting
        weights = 1.0 / (renin_std**2 + 1e-6)
        loss_data = self.data_loss(renin_pred, renin_data, weights)
        
        # === Biological Constraint Loss ===
        # FIXED: Renin should DECREASE with increasing dexamethasone
        # Test this by sampling multiple doses at t=24h
        dex_test = torch.tensor([[0.0], [0.3], [3.0], [30.0]], dtype=torch.float32).to(self.device)
        t_test = torch.full((4, 1), 24.0, dtype=torch.float32).to(self.device)
        u_test = self.model(t_test, dex_test)
        renin_test = u_test[:, 2]
        
        # Penalize if renin increases with dose (should decrease)
        diff = renin_test[1:] - renin_test[:-1]  # Should be negative
        violation = torch.relu(diff)  # Penalize positive differences
        loss_biological = torch.mean(violation**2)
        
        # === Physics Loss ===
        t_phys, dex_phys = self.generate_collocation_points(n_collocation)
        t_phys = t_phys.to(self.device).requires_grad_(True)
        dex_phys = dex_phys.to(self.device)
        
        u_phys = self.model(t_phys, dex_phys)
        u_t_phys = compute_derivatives(t_phys, u_phys, create_graph=True)
        
        loss_phys = self.physics_loss(t_phys, dex_phys, u_phys, u_t_phys)
        
        # === Initial Condition Loss ===
        t_ic, dex_ic = self.generate_ic_points(100)
        t_ic = t_ic.to(self.device)
        dex_ic = dex_ic.to(self.device)
        
        u_ic = self.model(t_ic, dex_ic)
        loss_ic = self.ic_loss(t_ic, dex_ic, u_ic)
        
        # === Parameter Constraint Loss ===
        # FIXED: Add constraints for key parameters to guide them toward target values
        params = self.model.get_params()
        
        # IC50 constraint (target: 2.88, current: 1.78)
        ic50_target = 2.88
        ic50_current = params['log_IC50']
        ic50_constraint = (ic50_current - ic50_target) ** 2
        
        # Hill coefficient constraint (target: 1.92, current: 1.22)
        hill_target = 1.92
        hill_current = params['log_hill']
        hill_constraint = (hill_current - hill_target) ** 2
        
        # Parameter constraint loss (weighted to guide but not force)
        loss_params = 0.01 * ic50_constraint + 0.01 * hill_constraint
        
        # === Total Loss ===
        loss_total = (loss_weights['data'] * loss_data +
                     loss_weights['physics'] * loss_phys +
                     loss_weights['ic'] * loss_ic +
                     10.0 * loss_biological +  # FIXED: Added biological constraint
                     loss_params)  # FIXED: Added parameter constraints
        
        # Backpropagation
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        losses = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_phys.item(),
            'ic': loss_ic.item(),
            'biological': loss_biological.item(),
            'params': loss_params if isinstance(loss_params, float) else loss_params.item()
        }
        
        return losses
    
    def train(self,
              data_dict,
              n_epochs=10000,
              print_every=500,
              curriculum_learning=True):
        """
        Full training loop with curriculum learning
        
        Args:
            data_dict: Experimental data
            n_epochs: Number of training epochs
            print_every: Print frequency
            curriculum_learning: Use adaptive loss weights
        """
        print("="*70)
        print("Physics-Informed Neural Networks for Modeling")
        print("Glucocorticoid Regulation of Renin")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Data points: {data_dict['n_samples']}")
        print("="*70)
        
        for epoch in tqdm(range(n_epochs), desc="Training"):
            
            # Curriculum learning: FIXED schedule to emphasize physics
            if curriculum_learning:
                if epoch < 2000:
                    # Early: Focus on fitting data and initial conditions
                    weights = {'data': 10.0, 'physics': 1.0, 'ic': 5.0}
                elif epoch < 5000:
                    # Middle: Increase physics emphasis
                    weights = {'data': 5.0, 'physics': 10.0, 'ic': 2.0}
                elif epoch < 8000:
                    # Late: Balance but keep strong physics
                    weights = {'data': 2.0, 'physics': 50.0, 'ic': 1.0}
                else:
                    # Final: Maximum physics emphasis
                    weights = {'data': 1.0, 'physics': 100.0, 'ic': 0.5}
            else:
                weights = {'data': 1.0, 'physics': 10.0, 'ic': 5.0}
            
            # Training step
            losses = self.train_step(data_dict, loss_weights=weights)
            
            # Update scheduler
            self.scheduler.step(losses['total'])
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(losses['total'])
            self.history['data_loss'].append(losses['data'])
            self.history['physics_loss'].append(losses['physics'])
            self.history['ic_loss'].append(losses['ic'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['parameters'].append(self.model.get_params())
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                params = self.model.get_params()
                print(f"\nEpoch {epoch+1}/{n_epochs}")
                print(f"  Total Loss: {losses['total']:.6f}")
                print(f"  Data Loss:  {losses['data']:.6f}")
                print(f"  Physics Loss: {losses['physics']:.6f}")
                print(f"  IC Loss: {losses['ic']:.6f}")
                print(f"  Biological Loss: {losses['biological']:.6f}")
                print(f"  Parameter Loss: {losses['params']:.6f}")
                print(f"  IC50: {params['log_IC50']:.3f} mg/dl (target: 2.88)")
                print(f"  Hill: {params['log_hill']:.3f} (target: 1.92)")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)
        
        # Final parameters
        print("\nFinal Parameters:")
        final_params = self.model.get_params()
        for k, v in final_params.items():
            print(f"  {k}: {v:.6f}")
    
    def save_checkpoint(self, filepath):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'parameters': self.model.get_params()
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['parameters']


class UnifiedPINNTrainer:
    """
    Unified PINN trainer that keeps every loss component active from the first epoch,
    applies a short physics ramp-up, and performs inline validation so we can stop
    as soon as the fit degrades or plausibility checks fail.
    """

    def __init__(self,
                 model,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 config: Optional[UnifiedTrainingConfig] = None,
                 plausibility_config: Optional[PlausibilityConfig] = None,
                 early_stopping: Optional[EarlyStoppingConfig] = None,
                 baseline_temporal_path: Optional[str] = None,
                 parameter_targets: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None):
        self.model = model.to(device)
        self.device = device

        self.config = config or UnifiedTrainingConfig()
        self.plausibility_config = plausibility_config or PlausibilityConfig()
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.parameter_targets = parameter_targets or {'log_IC50': 2.88, 'log_hill': 1.92}

        self.high_dose_target: Optional[float] = None
        self.baseline_time_points: Optional[np.ndarray] = None
        self.baseline_predictions: Dict[float, np.ndarray] = {}
        self.dose_levels: List[float] = [0.0, 0.3, 3.0, 30.0]

        self.rng = np.random.default_rng(seed)

        self.optimizer = Adam([
            {'params': self.model.network.parameters(), 'lr': learning_rate},
            {'params': self.model.params.values(), 'lr': learning_rate * 0.1}
        ], weight_decay=weight_decay)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            min_lr=1e-6
        )

        self.history = {
            'epoch': [],
            'losses': [],
            'metrics': [],
            'plausibility': []
        }
        self.best_metrics = {'r2': -np.inf, 'rmse': np.inf, 'epoch': -1}
        self.r2_degrade_count = 0
        self.plaus_fail_count = 0
        self.stop_reason: Optional[str] = None
        self.latest_losses: Optional[Dict[str, float]] = None
        self.latest_metrics: Optional[Dict[str, float]] = None
        self.latest_plausibility: Optional[Dict[str, Dict]] = None

        self._baseline_path = Path(baseline_temporal_path).expanduser() if baseline_temporal_path else None
        self.synthetic_regions: Dict[float, Dict[str, np.ndarray]] = {}
        if self.config.loss_synthetic > 0:
            self.synthetic_regions = self._load_synthetic_regions()
            if not self.synthetic_regions:
                print("[UnifiedPINNTrainer] No plausible regions found for synthetic data; disabling synthetic loss.")
                self.config.loss_synthetic = 0.0
        elif self._baseline_path and self._baseline_path.exists():
            # Still load baseline metadata for biological penalties
            self._load_synthetic_regions()

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def generate_collocation_points(self, n_points: Optional[int] = None):
        n = n_points or self.config.collocation_points
        t = torch.rand(n, 1, device=self.device) * 48.0
        dex = torch.exp(torch.rand(n, 1, device=self.device) * np.log(30.0 / 0.01) + np.log(0.01))
        return t, dex

    def generate_ic_points(self, n_points: Optional[int] = None):
        n = n_points or self.config.ic_points
        t = torch.zeros(n, 1, device=self.device)
        dex = torch.rand(n, 1, device=self.device) * 30.0
        return t, dex

    def _physics_weight(self, epoch: int) -> float:
        base = self.config.loss_physics
        if self.config.physics_ramp_fraction <= 0:
            return base

        ramp_epochs = max(1, int(self.config.physics_ramp_fraction * self.config.n_epochs))
        if epoch < ramp_epochs:
            return base * float(epoch + 1) / float(ramp_epochs)
        return base

    def _biological_scale(self, epoch: int) -> float:
        """Linearly ramp biological penalties across the configured fraction of training."""
        fraction = float(self.config.biological_ramp_fraction)
        if fraction <= 0.0:
            return 1.0

        total_epochs = max(1, int(self.config.n_epochs))
        ramp_epochs = max(1, int(np.ceil(fraction * total_epochs)))
        progress = min(max(epoch, 0), ramp_epochs)

        return float(progress) / float(ramp_epochs)

    def _high_dose_weight_ramped(self, epoch: int) -> float:
        """
        Gentle plateau ramp for high_dose_weight to avoid immediate suppression failure.
        Keep low weight (7-10) for first 200 epochs, then gradually raise toward target.
        """
        base_weight = float(self.config.high_dose_weight)
        
        # Plateau parameters
        plateau_epochs = 200  # Keep low weight for first 200 epochs
        low_weight = min(10.0, base_weight * 0.5)  # Start at 50% of target (7-10 range)
        
        if epoch < plateau_epochs:
            # During plateau: keep at low_weight
            return low_weight
        else:
            # After plateau: linearly ramp from low_weight to base_weight
            total_epochs = max(1, int(self.config.n_epochs))
            ramp_epochs = max(1, total_epochs - plateau_epochs)
            progress = min(epoch - plateau_epochs, ramp_epochs)
            ramp_fraction = float(progress) / float(ramp_epochs)
            
            # Linear interpolation from low_weight to base_weight
            return low_weight + (base_weight - low_weight) * ramp_fraction

    def _parameter_penalty(self) -> torch.Tensor:
        ic50_target = torch.tensor(self.parameter_targets['log_IC50'], dtype=torch.float32, device=self.device)
        hill_target = torch.tensor(self.parameter_targets['log_hill'], dtype=torch.float32, device=self.device)

        ic50_current = torch.exp(self.model.params['log_IC50'])
        hill_current = torch.exp(self.model.params['log_hill'])

        loss_ic50 = (ic50_current - ic50_target).pow(2)
        loss_hill = (hill_current - hill_target).pow(2)
        return loss_ic50 + loss_hill

    def _biological_loss(self, epoch: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.dose_levels:
            self.dose_levels = [0.0, 0.3, 3.0, 30.0]

        base_dose_tensor = torch.tensor(self.dose_levels, dtype=torch.float32, device=self.device).unsqueeze(1)
        if self.config.monotonic_time_points:
            time_grid = torch.tensor(self.config.monotonic_time_points, dtype=torch.float32, device=self.device)
        else:
            steps = max(1, self.config.monotonic_sample_count)
            time_grid = torch.linspace(
                self.plausibility_config.time_start,
                self.plausibility_config.time_end,
                steps=steps,
                device=self.device,
                dtype=torch.float32
            )

        monotonic_penalty = torch.tensor(0.0, device=self.device)
        gradient_penalty = torch.tensor(0.0, device=self.device)
        suppression_penalty = torch.tensor(0.0, device=self.device)
        temporal_derivative_penalty = torch.tensor(0.0, device=self.device)
        renin_traces: List[torch.Tensor] = []
        if time_grid.numel() > 0:
            for t_value in time_grid:
                scalar_t = float(t_value.item() if hasattr(t_value, "item") else t_value)
                t_tensor = torch.full_like(base_dose_tensor, fill_value=scalar_t)
                renin = self.model(t_tensor, base_dose_tensor)[:, 2]
                renin_traces.append(renin)
                diff = renin[1:] - renin[:-1] + self.config.monotonic_tolerance
                monotonic_penalty = monotonic_penalty + torch.mean(torch.relu(diff) ** 2)

                if self.config.monotonic_gradient_weight > 0:
                    dose_tensor = base_dose_tensor.detach().clone().requires_grad_(True)
                    t_tensor_grad = torch.full_like(dose_tensor, fill_value=scalar_t)
                    renin_grad = self.model(t_tensor_grad, dose_tensor)[:, 2]
                    grad = torch.autograd.grad(
                        outputs=renin_grad,
                        inputs=dose_tensor,
                        grad_outputs=torch.ones_like(renin_grad),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    gradient_penalty = gradient_penalty + torch.mean(torch.relu(grad + self.config.monotonic_tolerance) ** 2)
            monotonic_penalty = monotonic_penalty / float(time_grid.numel())
            if self.config.monotonic_gradient_weight > 0:
                gradient_penalty = gradient_penalty / float(time_grid.numel())

        renin_matrix = torch.stack(renin_traces, dim=0) if renin_traces else None

        if self.config.monotonic_data_weight > 0:
            t_obs = torch.full_like(base_dose_tensor, fill_value=24.0)
            renin_obs = self.model(t_obs, base_dose_tensor)[:, 2]
            diff_obs = renin_obs[1:] - renin_obs[:-1] + self.config.monotonic_tolerance
            monotonic_penalty = monotonic_penalty + self.config.monotonic_data_weight * torch.mean(torch.relu(diff_obs) ** 2)

        if (
            renin_matrix is not None
            and self.config.temporal_suppression_weight > 0
            and renin_matrix.size(0) > 0
        ):
            head_frac = float(self.config.temporal_suppression_head_fraction)
            tail_frac = float(self.config.temporal_suppression_tail_fraction)
            head_len = max(1, int(round(head_frac * renin_matrix.size(0))))
            tail_len = max(1, int(round(tail_frac * renin_matrix.size(0))))
            head_mean = renin_matrix[:head_len].mean(dim=0)
            tail_mean = renin_matrix[-tail_len:].mean(dim=0)

            diff_tail = tail_mean - head_mean + self.config.monotonic_tolerance
            if diff_tail.numel() > 1:
                suppression_penalty = torch.mean(torch.relu(diff_tail[1:]) ** 2)

        if (
            renin_matrix is not None
            and self.config.temporal_derivative_weight > 0
            and renin_matrix.size(0) > 1
            and renin_matrix.size(1) > 1
        ):
            forward_diff = renin_matrix[1:, 1:] - renin_matrix[:-1, 1:] + self.config.temporal_derivative_tolerance
            temporal_derivative_penalty = torch.mean(torch.relu(forward_diff) ** 2)

        if self.high_dose_target is not None and self.dose_levels:
            if self.config.high_dose_times:
                high_times = torch.tensor(self.config.high_dose_times, dtype=torch.float32, device=self.device)
            else:
                high_times = torch.tensor([24.0], dtype=torch.float32, device=self.device)

            high_dose_penalty = torch.tensor(0.0, device=self.device)
            d_high = torch.tensor([[self.dose_levels[-1]]], dtype=torch.float32, device=self.device)
            target_tensor = torch.tensor(self.high_dose_target, dtype=torch.float32, device=self.device)
            for t_value in high_times:
                scalar_t = float(t_value.item() if hasattr(t_value, "item") else t_value)
                t_high = torch.tensor([[scalar_t]], dtype=torch.float32, device=self.device)
                renin_high = self.model(t_high, d_high)[:, 2]
                high_dose_penalty = high_dose_penalty + (renin_high - target_tensor).pow(2)

            high_dose_penalty = high_dose_penalty.mean()
        else:
            high_dose_penalty = torch.tensor(0.0, device=self.device)

        ramp_scale = self._biological_scale(epoch)
        suppression_penalty = suppression_penalty * ramp_scale
        high_dose_penalty = high_dose_penalty * ramp_scale

        # Use ramped high_dose_weight instead of flat config value
        high_dose_weight_current = self._high_dose_weight_ramped(epoch)

        total = (
            monotonic_penalty
            + self.config.monotonic_gradient_weight * gradient_penalty
            + self.config.temporal_suppression_weight * suppression_penalty
            + self.config.temporal_derivative_weight * temporal_derivative_penalty
            + high_dose_weight_current * high_dose_penalty
        )

        components = {
            'monotonic_fd': monotonic_penalty,
            'gradient': gradient_penalty,
            'suppression': suppression_penalty,
            'temporal': temporal_derivative_penalty,
            'highdose': high_dose_penalty,
        }
        return total, components

    def _load_synthetic_regions(self) -> Dict[float, Dict[str, np.ndarray]]:
        regions: Dict[float, Dict[str, np.ndarray]] = {}
        if self._baseline_path is None or not self._baseline_path.exists():
            return regions

        try:
            with self._baseline_path.open('r') as handle:
                baseline = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[UnifiedPINNTrainer] Failed to read baseline temporal file: {exc}")
            return regions

        time_points = np.asarray(baseline.get('time_course', {}).get('time_points', []), dtype=np.float32)
        predictions = baseline.get('time_course', {}).get('predictions', {})
        if time_points.size == 0 or not predictions:
            return regions

        renin_by_dose: Dict[float, np.ndarray] = {}
        for dose_str, values in predictions.items():
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
            if arr.ndim != 2 or arr.shape[0] != time_points.size or arr.shape[1] < 3:
                continue
            renin_by_dose[float(dose_str)] = arr[:, 2]

        if not renin_by_dose:
            return regions

        self.baseline_time_points = time_points
        self.baseline_predictions = renin_by_dose
        self.dose_levels = sorted(renin_by_dose.keys())
        try:
            self.plausibility_config.doses = self.dose_levels
        except AttributeError:
            pass

        base_time_mask = (
            (time_points >= self.plausibility_config.time_start) &
            (time_points <= self.plausibility_config.time_end)
        )
        max_value = self.plausibility_config.max_value + 0.05
        derivative_threshold = self.plausibility_config.derivative_threshold * 1.05
        tolerance = self.config.monotonic_tolerance

        for idx, dose in enumerate(self.dose_levels):
            renin = renin_by_dose[dose]
            derivative = np.gradient(renin, time_points, edge_order=2)
            mask = (
                base_time_mask &
                (renin >= 0.0) &
                (renin <= max_value) &
                (np.abs(derivative) <= derivative_threshold)
            )

            if idx > 0:
                prev_dose = self.dose_levels[idx - 1]
                renin_prev = renin_by_dose[prev_dose]
                mask &= renin <= renin_prev + tolerance

            if np.any(mask):
                regions[dose] = {
                    'time': time_points[mask],
                    'renin': renin[mask]
                }

        return regions

    def _ensure_targets(self, data_dict: Dict):
        if self.high_dose_target is not None:
            return

        dex = np.asarray(data_dict['dex_concentration'], dtype=np.float32)
        renin = np.asarray(data_dict['renin_normalized'], dtype=np.float32)
        if dex.size == 0:
            return

        unique_doses = sorted(float(x) for x in np.unique(dex))
        if unique_doses:
            self.dose_levels = unique_doses
            try:
                self.plausibility_config.doses = unique_doses
            except AttributeError:
                pass
        max_dose = unique_doses[-1] if unique_doses else None
        if max_dose is not None:
            mask = np.isclose(dex, max_dose)
            if np.any(mask):
                self.high_dose_target = float(np.mean(renin[mask]))
        if self.high_dose_target is None and renin.size:
            self.high_dose_target = float(np.mean(renin))

    def _sample_synthetic_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.config.loss_synthetic <= 0 or not self.synthetic_regions:
            return None

        total_available = int(sum(len(region['time']) for region in self.synthetic_regions.values()))
        if total_available == 0:
            return None

        n_samples = min(self.config.synthetic_samples_per_epoch, total_available)
        doses = list(self.synthetic_regions.keys())

        t_samples = []
        d_samples = []
        y_samples = []

        for _ in range(n_samples):
            dose = float(self.rng.choice(doses))
            region = self.synthetic_regions[dose]
            idx = int(self.rng.integers(0, len(region['time'])))
            t_val = float(region['time'][idx])
            renin_val = float(region['renin'][idx])
            noisy_target = renin_val + float(self.rng.normal(0.0, self.config.synthetic_noise_std))
            noisy_target = max(noisy_target, 0.0)

            t_samples.append(t_val)
            d_samples.append(dose)
            y_samples.append(noisy_target)

        t_tensor = torch.tensor(t_samples, dtype=torch.float32, device=self.device).unsqueeze(1)
        d_tensor = torch.tensor(d_samples, dtype=torch.float32, device=self.device).unsqueeze(1)
        y_tensor = torch.tensor(y_samples, dtype=torch.float32, device=self.device)
        return t_tensor, d_tensor, y_tensor

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------
    def train_step(self,
                   data_dict: Dict,
                   epoch: int,
                   synthetic_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        t_data = torch.tensor(data_dict['time'], dtype=torch.float32, device=self.device).unsqueeze(1)
        dex_data = torch.tensor(data_dict['dex_concentration'], dtype=torch.float32, device=self.device).unsqueeze(1)
        renin_data = torch.tensor(data_dict['renin_normalized'], dtype=torch.float32, device=self.device)
        renin_std = torch.tensor(data_dict['renin_std'], dtype=torch.float32, device=self.device)

        u_data = self.model(t_data, dex_data)
        renin_pred = u_data[:, 2]
        weights = 1.0 / (renin_std ** 2 + 1e-6)
        loss_data = torch.mean(weights * (renin_pred - renin_data) ** 2)

        if synthetic_batch is not None:
            t_syn, d_syn, y_syn = synthetic_batch
            u_syn = self.model(t_syn, d_syn)
            renin_syn = u_syn[:, 2]
            loss_synthetic = torch.mean((renin_syn - y_syn) ** 2)
        else:
            loss_synthetic = torch.tensor(0.0, device=self.device)

        t_phys, dex_phys = self.generate_collocation_points()
        t_phys.requires_grad_(True)
        u_phys = self.model(t_phys, dex_phys)
        u_t_phys = compute_derivatives(t_phys, u_phys, create_graph=True)
        physics_residual = self.model.physics_residual(t_phys, dex_phys, u_phys, u_t_phys)
        loss_physics = torch.mean(physics_residual ** 2)

        t_ic, dex_ic = self.generate_ic_points()
        u_ic = self.model(t_ic, dex_ic)
        loss_ic = self.ic_loss(t_ic, dex_ic, u_ic)

        loss_params = self._parameter_penalty()
        loss_bio_total, bio_components = self._biological_loss(epoch)

        physics_weight = self._physics_weight(epoch)

        loss_total = (
            self.config.loss_data * loss_data +
            physics_weight * loss_physics +
            self.config.loss_ic * loss_ic +
            self.config.loss_parameter * loss_params +
            self.config.loss_synthetic * loss_synthetic +
            self.config.loss_biological * loss_bio_total
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        losses = {
            'total': float(loss_total.item()),
            'data': float(loss_data.item()),
            'physics': float(loss_physics.item()),
            'ic': float(loss_ic.item()),
            'params': float(loss_params.item()),
            'synthetic': float(loss_synthetic.item()),
            'biological': float(loss_bio_total.item()),
            'biological_monotonic': float(bio_components['monotonic_fd'].item()),
            'biological_gradient': float(bio_components['gradient'].item()),
            'biological_suppression': float(bio_components['suppression'].item()),
            'biological_temporal': float(bio_components['temporal'].item()),
            'biological_highdose': float(bio_components['highdose'].item()),
            'physics_weight': float(physics_weight)
        }
        self.latest_losses = losses

        return losses

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate_metrics(self, data_dict: Dict) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(data_dict['time'], dtype=torch.float32, device=self.device).unsqueeze(1)
            dex = torch.tensor(data_dict['dex_concentration'], dtype=torch.float32, device=self.device).unsqueeze(1)
            renin_pred = self.model(t, dex)[:, 2].cpu().numpy()

        metrics = calculate_metrics(
            np.asarray(data_dict['renin_normalized']),
            renin_pred
        )

        params = self.model.get_params()
        metrics.update({
            'ic50': params.get('log_IC50', np.nan),
            'hill': params.get('log_hill', np.nan),
            'ic50_gap': abs(params.get('log_IC50', np.nan) - self.parameter_targets['log_IC50']),
            'hill_gap': abs(params.get('log_hill', np.nan) - self.parameter_targets['log_hill'])
        })
        self.latest_metrics = metrics
        return metrics

    def evaluate_plausibility(self) -> Dict:
        cfg = self.plausibility_config
        time_grid = np.linspace(cfg.time_start, cfg.time_end, cfg.n_points)

        results = {}
        dose_curves: Dict[float, np.ndarray] = {}
        tolerance = self.config.monotonic_tolerance
        all_passed = True

        for idx, dose in enumerate(cfg.doses):
            t_tensor = torch.tensor(time_grid, dtype=torch.float32, device=self.device).unsqueeze(1)
            dex_tensor = torch.full_like(t_tensor, dose)

            with torch.no_grad():
                pred = self.model(t_tensor, dex_tensor).cpu().numpy()

            renin = pred[:, 2]
            derivative = np.gradient(renin, time_grid, edge_order=2)

            checks = {
                'non_negative': bool(np.all(renin >= 0.0)),
                'bounded': bool(np.all(renin <= cfg.max_value)),
                'smooth': bool(np.max(np.abs(derivative)) <= cfg.derivative_threshold),
                'steady_state': True,
                'suppression': True,
                'dose_suppression': True
            }

            if len(renin) >= cfg.steady_state_window:
                tail = renin[-cfg.steady_state_window:]
                checks['steady_state'] = float(np.std(tail)) <= cfg.steady_state_std

            # Use plausibility config suppression tolerance
            plaus_tolerance = getattr(cfg, 'suppression_tolerance', 0.02)
            
            if dose > 0:
                checks['suppression'] = bool(renin[-1] <= renin[0] + plaus_tolerance)

            if idx > 0:
                prev_dose = cfg.doses[idx - 1]
                prev_curve = dose_curves.get(prev_dose)
                if prev_curve is not None and len(prev_curve) == len(renin):
                    checks['dose_suppression'] = bool(np.all(renin <= prev_curve + plaus_tolerance))

            passed = all(checks.values())
            if not passed:
                all_passed = False

            dose_curves[dose] = renin

            results[dose] = {
                'passed': passed,
                'checks': checks
            }

        plausibility = {'all_passed': all_passed, 'doses': results}
        self.latest_plausibility = plausibility
        return plausibility

    # ------------------------------------------------------------------
    # Training loop orchestration
    # ------------------------------------------------------------------
    def ic_loss(self, t: torch.Tensor, dex: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ic_values = self.model.initial_conditions()
        state_names = ['mRNA', 'protein', 'secreted', 'GR_free', 'GR_cyto', 'GR_nuc']
        loss = torch.tensor(0.0, device=self.device)
        for idx, name in enumerate(state_names):
            target = ic_values[name]
            loss = loss + torch.mean((u[:, idx] - target) ** 2)
        return loss / len(state_names)

    def _update_best_metrics(self, metrics: Dict[str, float], epoch: int):
        if metrics['r2'] is None or np.isnan(metrics['r2']):
            return

        if metrics['r2'] > self.best_metrics['r2'] + 1e-6:
            self.best_metrics = {'r2': metrics['r2'], 'rmse': metrics['rmse'], 'epoch': epoch}
            self.r2_degrade_count = 0
        else:
            if epoch >= self.early_stopping.min_epochs and metrics['r2'] < self.best_metrics['r2'] - self.early_stopping.r2_tolerance:
                self.r2_degrade_count += 1
            else:
                self.r2_degrade_count = 0

    def _check_early_stopping(self, plausibility: Dict, epoch: int) -> bool:
        if plausibility['all_passed']:
            self.plaus_fail_count = 0
        else:
            self.plaus_fail_count += 1

        should_stop = False
        reason = None

        if epoch >= self.early_stopping.min_epochs and self.r2_degrade_count >= self.early_stopping.patience:
            should_stop = True
            reason = f"r2 degraded for {self.r2_degrade_count} epochs"

        if self.plaus_fail_count >= self.early_stopping.plausibility_patience:
            should_stop = True
            reason = "plausibility checks failed repeatedly"

        if should_stop and reason:
            self.stop_reason = reason

        return should_stop

    def _print_progress(self, epoch: int):
        if not self.latest_losses or not self.latest_metrics or not self.latest_plausibility:
            return

        losses = self.latest_losses
        metrics = self.latest_metrics
        plausibility = self.latest_plausibility

        ic50_gap = metrics.get('ic50_gap', np.nan)
        hill_gap = metrics.get('hill_gap', np.nan)
        plaus_flag = "PASS" if plausibility['all_passed'] else "FAIL"

        print(f"\nEpoch {epoch + 1}")
        print(f"  Loss total / data / physics / synthetic: "
              f"{losses['total']:.5f} / {losses['data']:.5f} / {losses['physics']:.5f} / {losses['synthetic']:.5f}")
        print(f"  Metrics r2={metrics['r2']:.4f}  rmse={metrics['rmse']:.4f}  ic50_gap={ic50_gap:.3f}  hill_gap={hill_gap:.3f}")
        print(f"  Plausibility: {plaus_flag}")

    def train(self,
              data_dict: Dict,
              config: Optional[UnifiedTrainingConfig] = None) -> Dict[str, List]:
        if config is not None:
            self.config = config

        print("=" * 70)
        print("Unified PINN Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.n_epochs}")
        print(f"Synthetic weight: {self.config.loss_synthetic}")
        print("=" * 70)

        self._ensure_targets(data_dict)

        for epoch in tqdm(range(self.config.n_epochs), desc="Unified Training"):
            synthetic_batch = self._sample_synthetic_batch()
            losses = self.train_step(data_dict, epoch, synthetic_batch)

            self.scheduler.step(losses['total'])

            metrics = self.evaluate_metrics(data_dict)
            plausibility = self.evaluate_plausibility()

            self.history['epoch'].append(epoch)
            self.history['losses'].append(losses)
            self.history['metrics'].append(metrics)
            self.history['plausibility'].append(plausibility)

            self._update_best_metrics(metrics, epoch)
            should_stop = self._check_early_stopping(plausibility, epoch)

            if self.config.print_every and ((epoch + 1) % self.config.print_every == 0):
                self._print_progress(epoch)

            if should_stop:
                print(f"\n[UnifiedPINNTrainer] Early stopping triggered at epoch {epoch + 1}: {self.stop_reason}")
                break

        return self.history

    def save_checkpoint(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_metrics': self.best_metrics,
            'stop_reason': self.stop_reason
        }, filepath)
        print(f"[UnifiedPINNTrainer] Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)
        self.stop_reason = checkpoint.get('stop_reason', None)
        print(f"[UnifiedPINNTrainer] Checkpoint loaded from {filepath}")
        return self.history
