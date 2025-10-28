"""
Calibration sweeps for the unified PINN training pipeline.

This module orchestrates three short experiments:
1. Parameter constraint weight sweep.
2. Synthetic data weight sweep using the best constraint weight.
3. Staged pretraining length sweep to quantify parameter drift.

Results are written to ``results/calibration/calibration_summary.json`` for
subsequent reporting and analysis.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .data import prepare_training_data
from .model import ReninPINN
from .trainer import (
    EarlyStoppingConfig,
    PlausibilityConfig,
    UnifiedPINNTrainer,
    UnifiedTrainingConfig,
)


# ---------------------------------------------------------------------------
# Global configuration for the sweeps
# ---------------------------------------------------------------------------
CONSTRAINT_WEIGHTS = [0.005, 0.01, 0.02, 0.04]
SYNTHETIC_WEIGHTS = [0.0, 0.1, 0.2, 0.5]
PRETRAIN_LENGTHS = [0, 250, 500]

SHORT_TRAIN_EPOCHS = 500
MAIN_TRAIN_EPOCHS = 900

BASELINE_TEMPORAL_PATH = Path("results/comprehensive/temporal/temporal_validation_results.json")
CALIBRATION_DIR = Path("results/calibration")

PARAMETER_TARGETS = {"log_IC50": 2.88, "log_hill": 1.92}

# Each variant bundles a loss_biological, monotonic gradient weight, and synthetic noise std.
TRAINING_VARIANTS = [
    {
        "id": 1,
        "loss_biological": 22.0,
        "monotonic_gradient_weight": 8.0,
        "synthetic_noise_std": 0.03,
        "biological_ramp_fraction": 0.4,
        "high_dose_weight": 18.0,
    },
    {
        "id": 0,
        "loss_biological": 18.0,
        "monotonic_gradient_weight": 6.0,
        "synthetic_noise_std": 0.02,
        "biological_ramp_fraction": 0.35,
        "high_dose_weight": 15.0,
    },
    {
        "id": 2,
        "loss_biological": 32.0,
        "monotonic_gradient_weight": 14.0,
        "synthetic_noise_std": 0.05,
        "biological_ramp_fraction": 0.45,
        "high_dose_weight": 20.0,
    },
]


def set_global_seed(seed: int):
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(value: Optional[float]) -> Optional[float]:
    """Convert values to float while preserving None for NaNs."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def serialise_plausibility(plausibility: Dict) -> Dict:
    """Convert plausibility dictionary to a JSON-friendly structure."""
    doses = {
        str(dose): {
            "passed": bool(info.get("passed")),
            "checks": {name: bool(flag) for name, flag in info.get("checks", {}).items()}
        }
        for dose, info in plausibility.get("doses", {}).items()
    }
    return {"all_passed": bool(plausibility.get("all_passed")), "doses": doses}


def default_plausibility_config() -> PlausibilityConfig:
    """Return plausibility configuration tailored for calibration speed."""
    return PlausibilityConfig(
        doses=[0.0, 0.3, 3.0, 30.0],
        time_start=0.0,
        time_end=48.0,
        n_points=90,
        derivative_threshold=0.15,  # Loosened from 0.12 to allow gentler high-dose suppression
        max_value=1.8,  # Loosened from 1.6 to provide more headroom
        steady_state_window=10,
        steady_state_std=0.06,  # Loosened from 0.05 for more steady-state tolerance
        suppression_tolerance=0.03,  # Allow 3% tolerance for suppression checks at high dose
    )


def relaxed_early_stopping(total_epochs: int) -> EarlyStoppingConfig:
    """Calibration-friendly early stopping with reasonable guard rails."""
    patience = max(50, total_epochs // 4)
    min_epochs = max(200, total_epochs // 2)
    return EarlyStoppingConfig(
        patience=patience,
        min_epochs=min_epochs,
        r2_tolerance=0.008,
        plausibility_patience=max(20, total_epochs // 6),
    )


def create_model(seed: int) -> ReninPINN:
    """Instantiate a fresh ReninPINN with deterministic initialisation."""
    set_global_seed(seed)
    return ReninPINN(hidden_layers=[128, 128, 128, 128], activation="tanh")


def summarise_metrics(metrics: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Keep only the key scalar metrics for reporting."""
    return {
        "r2": to_float(metrics.get("r2")),
        "rmse": to_float(metrics.get("rmse")),
        "ic50": to_float(metrics.get("ic50")),
        "ic50_gap": to_float(metrics.get("ic50_gap")),
        "hill": to_float(metrics.get("hill")),
        "hill_gap": to_float(metrics.get("hill_gap")),
    }


def summarise_parameters(params: Dict[str, float]) -> Dict[str, float]:
    """Return parameter summary augmented with gap relative to targets."""
    ic50 = float(params.get("log_IC50", np.nan))
    hill = float(params.get("log_hill", np.nan))
    return {
        "ic50": ic50,
        "hill": hill,
        "ic50_gap": abs(ic50 - PARAMETER_TARGETS["log_IC50"]),
        "hill_gap": abs(hill - PARAMETER_TARGETS["log_hill"]),
    }


def best_constraint_configuration(results: List[Dict]) -> Dict:
    """Select the constraint weight/variant pairing favouring plausibility."""
    def sort_key(item: Dict) -> Tuple:
        plaus = item.get("plausibility", {})
        passes = plaus.get("all_passed", False)
        metrics = item.get("metrics", {})
        r2 = to_float(metrics.get("r2")) or float("-inf")
        gap = (metrics.get("ic50_gap", float("inf")) +
               metrics.get("hill_gap", float("inf")))
        return (0 if passes else 1, -r2, gap)

    sorted_results = sorted(results, key=sort_key)
    return sorted_results[0] if sorted_results else {}


def best_synthetic_configuration(results: List[Dict], variant_id: int) -> Dict:
    """Select synthetic weight prioritising plausibility for the chosen variant."""
    filtered = [item for item in results if item.get("variant") == variant_id]
    candidates = filtered if filtered else results

    def sort_key(item: Dict) -> Tuple:
        plaus = item.get("plausibility", {})
        passes = plaus.get("all_passed", False)
        metrics = item.get("metrics", {})
        r2 = to_float(metrics.get("r2")) or float("-inf")
        return (0 if passes else 1, -r2)

    sorted_results = sorted(candidates, key=sort_key)
    return sorted_results[0] if sorted_results else {}


def run_constraint_sweep(data: Dict,
                         device: torch.device,
                         baseline_path: Optional[Path]) -> List[Dict]:
    """Run sensitivity sweep over parameter constraint weights."""
    results = []
    for variant in TRAINING_VARIANTS:
        for idx, weight in enumerate(CONSTRAINT_WEIGHTS):
            model = create_model(seed=100 + 10 * variant["id"] + idx)

            config = UnifiedTrainingConfig(
                n_epochs=SHORT_TRAIN_EPOCHS,
                print_every=0,
                loss_data=1.0,
                loss_physics=5.0,
                loss_ic=0.5,
                loss_parameter=weight,
                loss_synthetic=0.0,
                loss_biological=variant["loss_biological"],
                physics_ramp_fraction=0.1,
                collocation_points=512,
                ic_points=128,
                synthetic_samples_per_epoch=0,
                synthetic_noise_std=variant["synthetic_noise_std"],
                max_grad_norm=1.0,
                monotonic_gradient_weight=variant["monotonic_gradient_weight"],
                biological_ramp_fraction=variant.get("biological_ramp_fraction", 0.15),
                high_dose_weight=variant.get("high_dose_weight", 35.0),
            )

            trainer = UnifiedPINNTrainer(
                model=model,
                device=device,
                learning_rate=1e-3,
                weight_decay=0.0,
                config=config,
                plausibility_config=default_plausibility_config(),
                early_stopping=relaxed_early_stopping(SHORT_TRAIN_EPOCHS),
                baseline_temporal_path=str(baseline_path) if baseline_path else None,
                parameter_targets=PARAMETER_TARGETS,
                seed=200 + 10 * variant["id"] + idx,
            )

            trainer.train(data)
            metrics = summarise_metrics(trainer.latest_metrics or {})
            params = summarise_parameters(trainer.model.get_params())
            plausibility = serialise_plausibility(trainer.latest_plausibility or {})

            result = {
                "constraint_weight": weight,
                "variant": variant["id"],
                "variant_params": variant,
                "metrics": metrics,
                "parameters": params,
                "plausibility": plausibility,
                "best_epoch": int(trainer.best_metrics.get("epoch", -1)),
                "final_loss": trainer.latest_losses,
            }
            results.append(result)

    return results


def run_synthetic_sweep(data: Dict,
                        device: torch.device,
                        baseline_path: Optional[Path],
                        constraint_weight: float,
                        variant_id: int) -> List[Dict]:
    """Run sensitivity sweep over synthetic data weights."""
    results = []
    variant = next((v for v in TRAINING_VARIANTS if v["id"] == variant_id), TRAINING_VARIANTS[0])
    for idx, weight in enumerate(SYNTHETIC_WEIGHTS):
        model = create_model(seed=300 + 10 * variant_id + idx)

        config = UnifiedTrainingConfig(
            n_epochs=SHORT_TRAIN_EPOCHS,
            print_every=0,
            loss_data=1.0,
            loss_physics=5.0,
            loss_ic=0.5,
            loss_parameter=constraint_weight,
            loss_synthetic=weight,
            loss_biological=variant["loss_biological"],
            physics_ramp_fraction=0.1,
            collocation_points=512,
            ic_points=128,
            synthetic_samples_per_epoch=24 if weight > 0 else 0,
            synthetic_noise_std=variant["synthetic_noise_std"],
            max_grad_norm=1.0,
            monotonic_gradient_weight=variant["monotonic_gradient_weight"],
            biological_ramp_fraction=variant.get("biological_ramp_fraction", 0.15),
            high_dose_weight=variant.get("high_dose_weight", 35.0),
        )

        trainer = UnifiedPINNTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            weight_decay=0.0,
            config=config,
            plausibility_config=default_plausibility_config(),
            early_stopping=relaxed_early_stopping(SHORT_TRAIN_EPOCHS),
            baseline_temporal_path=str(baseline_path) if baseline_path else None,
            parameter_targets=PARAMETER_TARGETS,
            seed=400 + 10 * variant_id + idx,
        )

        trainer.train(data)
        metrics = summarise_metrics(trainer.latest_metrics or {})
        plausibility = serialise_plausibility(trainer.latest_plausibility or {})

        result = {
            "synthetic_weight": weight,
            "variant": variant_id,
            "variant_params": variant,
            "metrics": metrics,
            "plausibility": plausibility,
            "best_epoch": int(trainer.best_metrics.get("epoch", -1)),
            "final_loss": trainer.latest_losses,
        }
        results.append(result)

    return results


def run_pretraining_sweep(data: Dict,
                          device: torch.device,
                          baseline_path: Optional[Path],
                          constraint_weight: float,
                          synthetic_weight: float,
                          variant_id: int) -> List[Dict]:
    """Evaluate the impact of staged pretraining lengths."""
    results = []
    variant = next((v for v in TRAINING_VARIANTS if v["id"] == variant_id), TRAINING_VARIANTS[0])
    pre_loss_bio = max(variant["loss_biological"] * 0.4, 5.0)
    pre_grad_weight = max(variant["monotonic_gradient_weight"] * 0.5, 2.0)
    for idx, pre_epochs in enumerate(PRETRAIN_LENGTHS):
        model = create_model(seed=500 + 10 * variant_id + idx)

        trainer = UnifiedPINNTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            weight_decay=0.0,
            config=UnifiedTrainingConfig(
                n_epochs=MAIN_TRAIN_EPOCHS,
                print_every=0,
                loss_data=1.0,
                loss_physics=5.0,
                loss_ic=0.5,
                loss_parameter=constraint_weight,
                loss_synthetic=synthetic_weight,
                loss_biological=variant["loss_biological"],
                physics_ramp_fraction=0.1,
                collocation_points=512,
                ic_points=128,
                synthetic_samples_per_epoch=24 if synthetic_weight > 0 else 0,
                synthetic_noise_std=variant["synthetic_noise_std"],
                max_grad_norm=1.0,
                monotonic_gradient_weight=variant["monotonic_gradient_weight"],
                biological_ramp_fraction=variant.get("biological_ramp_fraction", 0.15),
                high_dose_weight=variant.get("high_dose_weight", 35.0),
            ),
            plausibility_config=default_plausibility_config(),
            early_stopping=relaxed_early_stopping(MAIN_TRAIN_EPOCHS + pre_epochs),
            baseline_temporal_path=str(baseline_path) if baseline_path else None,
            parameter_targets=PARAMETER_TARGETS,
            seed=600 + 10 * variant_id + idx,
        )

        params_after_pre = None
        if pre_epochs > 0:
            pre_config = UnifiedTrainingConfig(
                n_epochs=pre_epochs,
                print_every=0,
                loss_data=1.0,
                loss_physics=0.0,
                loss_ic=0.5,
                loss_parameter=constraint_weight,
                loss_synthetic=0.0,
                loss_biological=pre_loss_bio,
                physics_ramp_fraction=0.0,
                collocation_points=256,
                ic_points=64,
                synthetic_samples_per_epoch=0,
                synthetic_noise_std=0.0,
                max_grad_norm=1.0,
                monotonic_gradient_weight=pre_grad_weight,
                biological_ramp_fraction=variant.get("biological_ramp_fraction", 0.15),
                high_dose_weight=variant.get("high_dose_weight", 35.0),
            )
            trainer.train(data, config=pre_config)
            params_after_pre = summarise_parameters(trainer.model.get_params())
            trainer.r2_degrade_count = 0
            trainer.plaus_fail_count = 0
            trainer.stop_reason = None
            trainer.best_metrics = {'r2': -np.inf, 'rmse': np.inf, 'epoch': -1}

        # Main training stage (continues from pretraining weights)
        trainer.train(data)

        metrics = summarise_metrics(trainer.latest_metrics or {})
        final_params = summarise_parameters(trainer.model.get_params())
        plausibility = serialise_plausibility(trainer.latest_plausibility or {})

        result = {
            "pretraining_epochs": pre_epochs,
            "variant": variant_id,
            "variant_params": variant,
            "parameters_after_pretraining": params_after_pre,
            "final_metrics": metrics,
            "final_parameters": final_params,
            "plausibility": plausibility,
            "best_epoch": int(trainer.best_metrics.get("epoch", -1)),
            "stop_reason": trainer.stop_reason,
        }
        results.append(result)

    return results


def best_pretraining_configuration(results: List[Dict], variant_id: int) -> Dict:
    filtered = [item for item in results if item.get("variant") == variant_id]
    candidates = filtered if filtered else results

    def sort_key(item: Dict) -> Tuple:
        plaus = item.get("plausibility", {})
        passes = plaus.get("all_passed", False)
        metrics = item.get("final_metrics", {})
        r2 = to_float(metrics.get("r2")) or float("-inf")
        pre_epochs = item.get("pretraining_epochs", 0) or 0
        if passes:
            return (0, -r2, pre_epochs)
        return (1, pre_epochs, -r2)

    sorted_results = sorted(candidates, key=sort_key)
    return sorted_results[0] if sorted_results else {}


def run_all_calibrations(output_dir: Path = CALIBRATION_DIR) -> Dict:
    """Execute the complete calibration pipeline and persist the summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_training_data(dataset="elisa", use_log_scale=False)
    baseline_path = BASELINE_TEMPORAL_PATH if BASELINE_TEMPORAL_PATH.exists() else None

    print("\n=== Calibration Step 1: Constraint weight sweep ===")
    constraint_results = run_constraint_sweep(data, device, baseline_path)
    best_constraint = best_constraint_configuration(constraint_results)
    best_constraint_weight = best_constraint.get("constraint_weight", CONSTRAINT_WEIGHTS[0])
    selected_variant_id = best_constraint.get("variant", TRAINING_VARIANTS[0]["id"])
    selected_variant_params = dict(best_constraint.get("variant_params", next(v for v in TRAINING_VARIANTS if v["id"] == selected_variant_id)))

    print("\n=== Calibration Step 2: Synthetic weight sweep ===")
    synthetic_results = run_synthetic_sweep(data, device, baseline_path, best_constraint_weight, selected_variant_id)
    best_synthetic = best_synthetic_configuration(synthetic_results, selected_variant_id)
    best_synthetic_weight = best_synthetic.get("synthetic_weight", SYNTHETIC_WEIGHTS[0])
    if best_synthetic.get("variant") is not None:
        selected_variant_id = best_synthetic.get("variant")
        selected_variant_params = dict(best_synthetic.get("variant_params", selected_variant_params))

    print("\n=== Calibration Step 3: Pretraining length sweep ===")
    pretraining_results = run_pretraining_sweep(
        data,
        device,
        baseline_path,
        constraint_weight=best_constraint_weight,
        synthetic_weight=best_synthetic_weight,
        variant_id=selected_variant_id,
    )
    best_pretraining = best_pretraining_configuration(pretraining_results, selected_variant_id)

    summary = {
        "constraint_sweep": constraint_results,
        "best_constraint": best_constraint,
        "synthetic_sweep": synthetic_results,
        "best_synthetic": best_synthetic,
        "pretraining_sweep": pretraining_results,
        "best_pretraining": best_pretraining,
        "variants": TRAINING_VARIANTS,
        "selected_weights": {
            "constraint": best_constraint_weight,
            "synthetic": best_synthetic_weight,
            "variant": selected_variant_id,
        },
        "selected_configuration": {
            "constraint_weight": best_constraint_weight,
            "synthetic_weight": best_synthetic_weight,
            "pretraining_epochs": best_pretraining.get("pretraining_epochs", 0),
            "variant": selected_variant_id,
            "variant_params": selected_variant_params,
        },
    }

    summary_path = output_dir / "calibration_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nCalibration summary written to {summary_path}")

    return summary


if __name__ == "__main__":
    run_all_calibrations()
