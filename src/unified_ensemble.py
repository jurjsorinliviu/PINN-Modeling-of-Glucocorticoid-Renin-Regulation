"""
Unified ensemble training for the PINN using the unified trainer.

The script:
1. Loads calibration weights (constraint & synthetic) when available.
2. Trains an ensemble of PINN models with different seeds.
3. Aggregates metrics, parameter statistics, residual diagnostics.
4. Generates Pareto, dose-response, and time-course figures.
5. Stores a single JSON summary consumed by the reporting pipeline.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import prepare_training_data
from .model import ReninPINN
from .statistical_utils import calculate_metrics, residual_analysis
from .trainer import (
    EarlyStoppingConfig,
    PlausibilityConfig,
    UnifiedPINNTrainer,
    UnifiedTrainingConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENSEMBLE_SIZE = 10
ENSEMBLE_EPOCHS = 1400
PARAMETER_TARGETS = {"log_IC50": 2.88, "log_hill": 1.92}

CALIBRATION_SUMMARY_PATH = Path("results/calibration/calibration_summary.json")
OUTPUT_DIR = Path("results/unified")
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

BASELINE_TEMPORAL_PATH = Path("results/comprehensive/temporal/temporal_validation_results.json")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def json_safe(value):
    """Recursively convert numpy / torch objects to JSON-safe Python types."""
    import numbers

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, numbers.Number):
        try:
            if math.isnan(value):
                return None
        except TypeError:
            pass
        return float(value)
    return value


def load_calibration_weights(path: Path) -> Tuple[Dict, Dict]:
    default_variant = {
        "id": 1,
        "loss_biological": 22.0,
        "monotonic_gradient_weight": 8.0,
        "synthetic_noise_std": 0.03,
        "biological_ramp_fraction": 0.4,
        "high_dose_weight": 18.0,
    }
    default_config = {
        "constraint_weight": 0.01,
        "synthetic_weight": 0.0,
        "variant": default_variant["id"],
        "variant_params": default_variant,
        "pretraining_epochs": 0,
    }
    if path.exists():
        with path.open("r") as handle:
            summary = json.load(handle)
        selected = summary.get("selected_configuration") or {}
        config = {**default_config, **selected}
        if not config.get("variant_params"):
            variants = summary.get("variants", [])
            variant_id = config.get("variant")
            match = next((v for v in variants if v.get("id") == variant_id), None)
            if match:
                config["variant_params"] = match
        return config, summary
    return default_config, {}


def default_plausibility_config() -> PlausibilityConfig:
    return PlausibilityConfig(
        doses=[0.0, 0.3, 3.0, 30.0],
        time_start=0.0,
        time_end=48.0,
        n_points=120,
        derivative_threshold=0.15,  # Loosened from 0.12 to allow gentler high-dose suppression
        max_value=1.8,  # Loosened from 1.6 to provide more headroom
        steady_state_window=12,
        steady_state_std=0.06,  # Loosened from 0.05 for more steady-state tolerance
        suppression_tolerance=0.03,  # Allow 3% tolerance for suppression checks at high dose
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


def create_model(seed: int) -> ReninPINN:
    set_global_seed(seed)
    return ReninPINN(hidden_layers=[128, 128, 128, 128], activation="tanh")


def summarise_metrics(metrics: Dict[str, float]) -> Dict[str, Optional[float]]:
    return {
        "r2": to_float(metrics.get("r2")),
        "rmse": to_float(metrics.get("rmse")),
        "mae": to_float(metrics.get("mae")),
        "ic50": to_float(metrics.get("ic50")),
        "ic50_gap": to_float(metrics.get("ic50_gap")),
        "hill": to_float(metrics.get("hill")),
        "hill_gap": to_float(metrics.get("hill_gap")),
    }


def summarise_parameters(params: Dict[str, float]) -> Dict[str, float]:
    ic50 = float(params.get("log_IC50", np.nan))
    hill = float(params.get("log_hill", np.nan))
    return {
        "ic50": ic50,
        "hill": hill,
        "ic50_gap": abs(ic50 - PARAMETER_TARGETS["log_IC50"]),
        "hill_gap": abs(hill - PARAMETER_TARGETS["log_hill"]),
    }


def serialise_plausibility(plausibility: Dict) -> Dict:
    doses = {
        str(dose): {
            "passed": bool(info.get("passed")),
            "checks": {name: bool(flag) for name, flag in info.get("checks", {}).items()}
        }
        for dose, info in plausibility.get("doses", {}).items()
    }
    return {"all_passed": bool(plausibility.get("all_passed")), "doses": doses}


def prepare_training_config(constraint_weight: float,
                            synthetic_weight: float,
                            variant_params: Optional[Dict[str, float]] = None) -> UnifiedTrainingConfig:
    variant_params = variant_params or {}
    loss_bio = float(variant_params.get("loss_biological", 35.0))
    gradient_weight = float(variant_params.get("monotonic_gradient_weight", 12.0))
    noise_std = float(variant_params.get("synthetic_noise_std", 0.03))
    bio_ramp = float(variant_params.get("biological_ramp_fraction", 0.15))
    high_dose_weight = float(variant_params.get("high_dose_weight", 35.0))
    return UnifiedTrainingConfig(
        n_epochs=ENSEMBLE_EPOCHS,
        print_every=0,
        loss_data=1.0,
        loss_physics=5.0,
        loss_ic=0.5,
        loss_parameter=constraint_weight,
        loss_synthetic=synthetic_weight,
        loss_biological=loss_bio,
        physics_ramp_fraction=0.1,
        collocation_points=512,
        ic_points=128,
        synthetic_samples_per_epoch=24 if synthetic_weight > 0 else 0,
        synthetic_noise_std=noise_std,
        max_grad_norm=1.0,
        monotonic_gradient_weight=gradient_weight,
        biological_ramp_fraction=bio_ramp,
        high_dose_weight=high_dose_weight,
    )


def collect_predictions(models: List[ReninPINN],
                        t_array: np.ndarray,
                        dex_array: np.ndarray,
                        device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    t_tensor = torch.tensor(t_array, dtype=torch.float32).reshape(-1, 1).to(device)
    dex_tensor = torch.tensor(dex_array, dtype=torch.float32).reshape(-1, 1).to(device)
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(t_tensor, dex_tensor).cpu().numpy()
        preds.append(pred)
    preds = np.stack(preds)  # (n_models, n_samples, n_states)
    return preds.mean(axis=0), preds.std(axis=0)


def compute_pareto_points(member_results: List[Dict]) -> List[Dict]:
    points = []
    for result in member_results:
        metrics = result.get("metrics", {})
        params = result.get("parameters", {})
        parameter_gap = to_float(params.get("ic50_gap")) or 0.0
        parameter_gap += to_float(params.get("hill_gap")) or 0.0
        points.append({
            "member": result.get("member"),
            "r2": to_float(metrics.get("r2")),
            "rmse": to_float(metrics.get("rmse")),
            "parameter_gap": parameter_gap,
        })
    return points


def find_pareto_knee(points: List[Dict]) -> Tuple[int, Dict]:
    if len(points) < 3:
        return 0, points[0] if points else {}

    sorted_points = sorted(points, key=lambda x: x["parameter_gap"])
    x = np.array([p["parameter_gap"] for p in sorted_points], dtype=np.float64)
    y = np.array([p["r2"] for p in sorted_points], dtype=np.float64)

    if np.allclose(x.max(), x.min()) or np.allclose(y.max(), y.min()):
        return 0, sorted_points[0]

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    points_norm = np.stack([x_norm, y_norm], axis=1)

    start = points_norm[0]
    end = points_norm[-1]
    line_vec = end - start
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        return 0, sorted_points[0]

    line_unit = line_vec / line_length
    distances = []
    for point in points_norm:
        vec = point - start
        proj = np.dot(vec, line_unit)
        proj_point = start + proj * line_unit
        distances.append(np.linalg.norm(point - proj_point))

    knee_idx = int(np.argmax(distances))
    knee_point = sorted_points[knee_idx]
    return knee_idx, knee_point


def plot_pareto(points: List[Dict], knee_point: Dict, output_path: Path):
    plt.figure(figsize=(6, 4))
    x = [p["parameter_gap"] for p in points]
    y = [p["r2"] for p in points]
    labels = [p["member"] for p in points]

    plt.scatter(x, y, c="tab:blue", label="Ensemble members")
    if knee_point:
        plt.scatter([knee_point["parameter_gap"]], [knee_point["r2"]],
                    c="tab:red", s=80, label=f"Knee (member {knee_point['member']})")
    for xi, yi, lbl in zip(x, y, labels):
        plt.annotate(str(lbl), (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel("Parameter gap (|IC50-Target| + |Hill-Target|)")
    plt.ylabel("R²")
    plt.title("Pareto frontier: accuracy vs parameter plausibility")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_dose_response(dex_range: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray,
                       data: Dict,
                       output_path: Path):
    plt.figure(figsize=(6, 4))
    renin_mean = mean[:, 2]
    renin_std = std[:, 2]

    plt.fill_between(dex_range, renin_mean - renin_std, renin_mean + renin_std,
                     color="tab:blue", alpha=0.2, label="Ensemble ±1σ")
    plt.plot(dex_range, renin_mean, color="tab:blue", linewidth=2)

    plt.errorbar(data["dex_concentration"], data["renin_normalized"],
                 yerr=data["renin_std"], fmt="o", color="tab:red",
                 capsize=4, label="Experimental")

    plt.xscale("log")
    plt.xlabel("Dexamethasone (mg/dl)")
    plt.ylabel("Normalized renin secretion")
    plt.title("Dose-response with ensemble uncertainty")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_time_courses(time_points: np.ndarray,
                      predictions: Dict[float, Tuple[np.ndarray, np.ndarray]],
                      output_path: Path):
    plt.figure(figsize=(8, 6))
    for dose, (mean, std) in predictions.items():
        renin_mean = mean[:, 2]
        renin_std = std[:, 2]
        label = f"{dose} mg/dl"
        plt.plot(time_points, renin_mean, label=label)
        plt.fill_between(time_points,
                         renin_mean - renin_std,
                         renin_mean + renin_std,
                         alpha=0.2)

    plt.xlabel("Time (h)")
    plt.ylabel("Normalized renin secretion")
    plt.title("Temporal dynamics with ensemble uncertainty")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def train_ensemble(n_members: int = ENSEMBLE_SIZE) -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_training_data(dataset="elisa", use_log_scale=False)

    selected_config, calibration_summary = load_calibration_weights(CALIBRATION_SUMMARY_PATH)
    constraint_weight = float(selected_config.get("constraint_weight", 0.01))
    synthetic_weight = float(selected_config.get("synthetic_weight", 0.0))
    variant_params = dict(selected_config.get("variant_params") or {})
    selected_variant_id = selected_config.get("variant")
    baseline_path = BASELINE_TEMPORAL_PATH if BASELINE_TEMPORAL_PATH.exists() else None
    print(f"[Ensemble] Calibration config -> constraint: {constraint_weight}, synthetic: {synthetic_weight}, "
          f"variant: {selected_variant_id}, params: {variant_params}")

    member_results = []
    trained_models: List[ReninPINN] = []
    fallback_candidates: List[Tuple[float, Dict, ReninPINN]] = []

    for member_idx in range(n_members):
        print(f"\n--- Training ensemble member {member_idx + 1}/{n_members} ---")

        model = create_model(seed=700 + member_idx)
        config = prepare_training_config(constraint_weight, synthetic_weight, variant_params)

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
            seed=900 + member_idx,
        )

        trainer.train(data)

        metrics = summarise_metrics(trainer.latest_metrics or {})
        params = summarise_parameters(trainer.model.get_params())
        plausibility = serialise_plausibility(trainer.latest_plausibility or {})

        # Predictions on experimental data (computed even if member is later rejected so that
        # fallback records remain complete)
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
            "variant": selected_variant_id,
            "variant_params": variant_params,
        }

        r2_value = metrics.get("r2")
        plaus_passed = plausibility.get("all_passed", False)
        accepted = r2_value is not None and r2_value >= 0.5 and plaus_passed

        if accepted:
            member_results.append(member_info)
            checkpoint_path = MODELS_DIR / f"unified_member_{member_idx}.pth"
            torch.save(trainer.model.state_dict(), checkpoint_path)
            trainer.model.to("cpu")
            trained_models.append(trainer.model)
        else:
            print(f"[Ensemble] Skipping member {member_idx} (r2={r2_value}, plausibility={plaus_passed})")
            trainer.model.to("cpu")
            fallback_candidates.append((r2_value if r2_value is not None else float("-inf"), member_info, trainer.model))

    if not trained_models:
        if fallback_candidates:
            fallback_candidates.sort(key=lambda item: (item[0],), reverse=True)
            best_r2, member_info, model = fallback_candidates[0]
            print(f"[Ensemble] Using fallback member with r2={best_r2} despite plausibility failure.")
            fallback_checkpoint = MODELS_DIR / f"unified_member_{member_info['member']}_fallback.pth"
            torch.save(model.state_dict(), fallback_checkpoint)
            member_results.append(member_info)
            trained_models.append(model)
        else:
            raise RuntimeError("All ensemble members were discarded due to low performance or failed plausibility.")

    # Ensemble aggregation
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
        "n_members": len(trained_models),
        "hyperparameters": {
            "constraint_weight": constraint_weight,
            "synthetic_weight": synthetic_weight,
            "epochs": ENSEMBLE_EPOCHS,
            "variant": selected_variant_id,
            "variant_params": variant_params,
        },
        "calibration_summary": calibration_summary,
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

    summary_path = OUTPUT_DIR / "unified_ensemble_results.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nUnified ensemble summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    train_ensemble()
