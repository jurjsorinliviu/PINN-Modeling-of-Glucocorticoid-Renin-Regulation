"""
Alternative ensemble with synthetic_weight=0.2 for tighter parameter alignment.

This script trains a 5-member ensemble using synthetic_weight=0.2 (instead of 0.5)
based on calibration results showing better parameter alignment while maintaining
plausibility.

Results are saved to results/unified_02/ for comparison with the main ensemble.
"""

import json
from pathlib import Path
import torch
import numpy as np

from src.data import prepare_training_data
from src.model import ReninPINN
from src.trainer import (
    EarlyStoppingConfig,
    PlausibilityConfig,
    UnifiedPINNTrainer,
    UnifiedTrainingConfig,
)
from src.unified_ensemble import (
    set_global_seed,
    create_model,
    summarise_metrics,
    summarise_parameters,
    serialise_plausibility,
    collect_predictions,
    compute_pareto_points,
    find_pareto_knee,
    plot_pareto,
    plot_dose_response,
    plot_time_courses,
    json_safe,
)
from src.statistical_utils import calculate_metrics, residual_analysis

# Configuration
ENSEMBLE_SIZE = 10  # Increase to get ~4-5 passing models with 30-40% success rate
ENSEMBLE_EPOCHS = 1400
SYNTHETIC_WEIGHT = 0.3  # Middle ground: better success rate than 0.2, better alignment than 0.5
PARAMETER_TARGETS = {"log_IC50": 2.88, "log_hill": 1.92}

OUTPUT_DIR = Path("results/unified_03")  # New directory for SW=0.3
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
BASELINE_TEMPORAL_PATH = Path("results/comprehensive/temporal/temporal_validation_results.json")

# Use variant 1 (same as main ensemble) for consistency

# Variant configuration from calibration (variant 1 with plateau ramp)
VARIANT_PARAMS = {
    "id": 1,
    "loss_biological": 22.0,
    "monotonic_gradient_weight": 8.0,
    "synthetic_noise_std": 0.03,
    "biological_ramp_fraction": 0.4,
    "high_dose_weight": 18.0,
}


def default_plausibility_config() -> PlausibilityConfig:
    return PlausibilityConfig(
        doses=[0.0, 0.3, 3.0, 30.0],
        time_start=0.0,
        time_end=48.0,
        n_points=120,
        derivative_threshold=0.15,
        max_value=1.8,
        steady_state_window=12,
        steady_state_std=0.06,
        suppression_tolerance=0.03,
    )


def relaxed_early_stopping(total_epochs: int) -> EarlyStoppingConfig:
    patience = max(60, total_epochs // 4)
    min_epochs = max(250, total_epochs // 2)
    return EarlyStoppingConfig(
        patience=patience,
        min_epochs=min_epochs,
        r2_tolerance=0.008,
        plausibility_patience=max(25, total_epochs // 6),
    )


def prepare_training_config(constraint_weight: float = 0.005) -> UnifiedTrainingConfig:
    return UnifiedTrainingConfig(
        n_epochs=ENSEMBLE_EPOCHS,
        print_every=0,
        loss_data=1.0,
        loss_physics=5.0,
        loss_ic=0.5,
        loss_parameter=constraint_weight,
        loss_synthetic=SYNTHETIC_WEIGHT,
        loss_biological=VARIANT_PARAMS["loss_biological"],
        physics_ramp_fraction=0.1,
        collocation_points=512,
        ic_points=128,
        synthetic_samples_per_epoch=24,
        synthetic_noise_std=VARIANT_PARAMS["synthetic_noise_std"],
        max_grad_norm=1.0,
        monotonic_gradient_weight=VARIANT_PARAMS["monotonic_gradient_weight"],
        biological_ramp_fraction=VARIANT_PARAMS["biological_ramp_fraction"],
        high_dose_weight=VARIANT_PARAMS["high_dose_weight"],
    )


def train_ensemble_03():
    """Train ensemble with synthetic_weight=0.3 as middle-ground compromise."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_training_data(dataset="elisa", use_log_scale=False)
    baseline_path = BASELINE_TEMPORAL_PATH if BASELINE_TEMPORAL_PATH.exists() else None

    print("\n" + "="*70)
    print(f"Training Middle-Ground Ensemble (synthetic_weight={SYNTHETIC_WEIGHT})")
    print("="*70)
    print(f"Target: Balance between accuracy and parameter alignment")
    print(f"Expected success rate: ~30-40% (better than SW=0.2's 20%)")
    print(f"Ensemble size: {ENSEMBLE_SIZE}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    member_results = []
    trained_models = []
    fallback_candidates = []

    for member_idx in range(ENSEMBLE_SIZE):
        print(f"\n--- Training member {member_idx + 1}/{ENSEMBLE_SIZE} ---")

        model = create_model(seed=1700 + member_idx)
        config = prepare_training_config()

        trainer = UnifiedPINNTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            weight_decay=0.0,
            config=config,
            plausibility_config=default_plausibility_config(),
            early_stopping=relaxed_early_stopping(ENSEMBLE_EPOCHS),
            baseline_temporal_path=str(baseline_path) if baseline_path else None,
            parameter_targets=PARAMETER_TARGETS,
            seed=1900 + member_idx,
        )

        trainer.train(data)

        metrics = summarise_metrics(trainer.latest_metrics or {})
        params = summarise_parameters(trainer.model.get_params())
        plausibility = serialise_plausibility(trainer.latest_plausibility or {})

        t_data = np.asarray(data["time"], dtype=np.float32)
        dex_data = np.asarray(data["dex_concentration"], dtype=np.float32)
        mean_pred, _ = collect_predictions([trainer.model], t_data, dex_data, device)

        member_info = {
            "member": member_idx,
            "metrics": metrics,
            "parameters": params,
            "plausibility": plausibility,
            "best_epoch": int(trainer.best_metrics.get("epoch", -1)),
            "stop_reason": trainer.stop_reason,
            "predictions": mean_pred[:, 2].tolist(),
            "variant": VARIANT_PARAMS["id"],
            "variant_params": VARIANT_PARAMS,
        }

        r2_value = metrics.get("r2")
        plaus_passed = plausibility.get("all_passed", False)
        accepted = r2_value is not None and r2_value >= 0.5 and plaus_passed

        if accepted:
            member_results.append(member_info)
            checkpoint_path = MODELS_DIR / f"unified_03_member_{member_idx}.pth"
            torch.save(trainer.model.state_dict(), checkpoint_path)
            trainer.model.to("cpu")
            trained_models.append(trainer.model)
            print(f"[OK] Member {member_idx} accepted (R2={r2_value:.3f}, plausibility=PASS)")
        else:
            print(f"[REJECT] Member {member_idx} rejected (R2={r2_value}, plausibility={plaus_passed})")
            trainer.model.to("cpu")
            fallback_candidates.append((r2_value if r2_value is not None else float("-inf"), member_info, trainer.model))

    if not trained_models:
        if fallback_candidates:
            fallback_candidates.sort(key=lambda item: item[0], reverse=True)
            best_r2, member_info, model = fallback_candidates[0]
            print(f"\n[Warning] Using fallback member with R²={best_r2}")
            fallback_checkpoint = MODELS_DIR / f"unified_03_member_{member_info['member']}_fallback.pth"
            torch.save(model.state_dict(), fallback_checkpoint)
            member_results.append(member_info)
            trained_models.append(model)
        else:
            raise RuntimeError("All ensemble members failed!")

    # Ensemble aggregation
    print(f"\n{'='*70}")
    print(f"Ensemble complete: {len(trained_models)}/{ENSEMBLE_SIZE} models accepted")
    print(f"{'='*70}\n")

    y_true = np.asarray(data["renin_normalized"], dtype=np.float32)
    t_data = np.asarray(data["time"], dtype=np.float32)
    dex_data = np.asarray(data["dex_concentration"], dtype=np.float32)

    ensemble_mean, ensemble_std = collect_predictions(trained_models, t_data, dex_data, torch.device("cpu"))
    ensemble_metrics = calculate_metrics(y_true, ensemble_mean[:, 2])
    residuals_stats = residual_analysis(y_true, ensemble_mean[:, 2], dex_data)

    pareto_points = compute_pareto_points(member_results)
    knee_idx, knee_point = find_pareto_knee(pareto_points)
    plot_pareto(pareto_points, knee_point, FIGURES_DIR / "pareto_frontier.png")

    # Dose-response predictions
    dex_range = np.logspace(-2, 2, 120)
    mean_dose, std_dose = collect_predictions(trained_models, np.full_like(dex_range, 24.0), dex_range, torch.device("cpu"))
    plot_dose_response(dex_range, mean_dose, std_dose, data, FIGURES_DIR / "dose_response.png")

    # Time-course predictions
    time_points = np.linspace(0.0, 48.0, 120)
    temporal_predictions = {}
    for dose in [0.0, 0.3, 3.0, 30.0]:
        dex_array = np.full_like(time_points, dose)
        mean_tc, std_tc = collect_predictions(trained_models, time_points, dex_array, torch.device("cpu"))
        temporal_predictions[dose] = (mean_tc, std_tc)
    plot_time_courses(time_points, temporal_predictions, FIGURES_DIR / "time_courses.png")

    summary = {
        "configuration": f"Middle-ground ensemble with synthetic_weight={SYNTHETIC_WEIGHT}",
        "n_members": len(trained_models),
        "hyperparameters": {
            "constraint_weight": 0.005,
            "synthetic_weight": SYNTHETIC_WEIGHT,
            "epochs": ENSEMBLE_EPOCHS,
            "variant": VARIANT_PARAMS["id"],
            "variant_params": VARIANT_PARAMS,
        },
        "member_results": member_results,
        "ensemble_metrics": {
            "r2": float(ensemble_metrics["r2"]),
            "rmse": float(ensemble_metrics["rmse"]),
            "mae": float(ensemble_metrics["mae"]),
            "ic50_mean": float(np.mean([m["parameters"]["ic50"] for m in member_results])),
            "ic50_std": float(np.std([m["parameters"]["ic50"] for m in member_results])),
            "hill_mean": float(np.mean([m["parameters"]["hill"] for m in member_results])),
            "hill_std": float(np.std([m["parameters"]["hill"] for m in member_results])),
            "ic50_gap_mean": float(np.mean([m["parameters"]["ic50_gap"] for m in member_results])),
            "ic50_gap_std": float(np.std([m["parameters"]["ic50_gap"] for m in member_results])),
            "hill_gap_mean": float(np.mean([m["parameters"]["hill_gap"] for m in member_results])),
            "hill_gap_std": float(np.std([m["parameters"]["hill_gap"] for m in member_results])),
        },
        "ensemble_predictions": {
            "mean": ensemble_mean[:, 2].tolist(),
            "std": ensemble_std[:, 2].tolist(),
        },
        "residual_diagnostics": {
            key: json_safe(value)
            for key, value in residuals_stats.items()
        },
        "pareto": {
            "points": pareto_points,
            "knee_index": int(knee_idx),
            "knee_point": knee_point,
        },
        "figures": {
            "pareto": str((FIGURES_DIR / "pareto_frontier.png").as_posix()),
            "dose_response": str((FIGURES_DIR / "dose_response.png").as_posix()),
            "time_courses": str((FIGURES_DIR / "time_courses.png").as_posix()),
        },
        "dose_response_curve": {
            "dex_range": dex_range.tolist(),
            "mean": mean_dose[:, 2].tolist(),
            "std": std_dose[:, 2].tolist(),
        },
        "time_courses": {
            str(dose): {
                "time": time_points.tolist(),
                "mean": temporal_predictions[dose][0][:, 2].tolist(),
                "std": temporal_predictions[dose][1][:, 2].tolist(),
            }
            for dose in temporal_predictions
        },
    }

    summary_path = OUTPUT_DIR / "unified_ensemble_03_results.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {summary_path}")
    print(f"{'='*70}\n")
    
    return summary


if __name__ == "__main__":
    train_ensemble_03()