"""
Reviewer-requested experiment suite for the deterministic PINN codebase.

This script fills the main computational gaps raised during peer review:
1. Traditional ML baselines on the sparse 24 h dose-response task.
2. Synthetic-augmentation ablation, including a no-synthetic setting.
3. Best single accepted PINN member versus ensemble summary.
4. Parameter-spread summary from accepted SW=0.3 ensemble checkpoints.
5. Coverage summary mapping reviewer requests to concrete output files.

Outputs are written under ``results/reviewer_experiments/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.data import prepare_training_data
from src.model import ReninPINN
from src.statistical_utils import calculate_metrics
from src.trainer import (
    EarlyStoppingConfig,
    PlausibilityConfig,
    UnifiedPINNTrainer,
)
from src.unified_ensemble import (
    PARAMETER_TARGETS,
    collect_predictions,
    create_model,
    json_safe,
    prepare_training_config,
    relaxed_early_stopping,
    serialise_plausibility,
    summarise_metrics,
    summarise_parameters,
)


OUTPUT_DIR = Path("results/reviewer_experiments")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

SW03_RESULTS_PATH = Path("results/unified_03/unified_ensemble_03_results.json")
SW05_RESULTS_PATH = Path("results/unified/unified_ensemble_results.json")
ODE_RESULTS_PATH = Path("results/ode_baseline_results.json")
SW03_MODEL_DIR = Path("results/unified_03/models")
BASELINE_TEMPORAL_PATH = Path("results/comprehensive/temporal/temporal_validation_results.json")

REVIEWER_VARIANT_PARAMS = {
    "id": 1,
    "loss_biological": 22.0,
    "monotonic_gradient_weight": 8.0,
    "synthetic_noise_std": 0.03,
    "biological_ramp_fraction": 0.4,
    "high_dose_weight": 18.0,
}

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_matrix(dose_values: np.ndarray) -> np.ndarray:
    dose_values = np.asarray(dose_values, dtype=np.float64).reshape(-1)
    safe_dose = np.where(dose_values <= 0.0, 0.01, dose_values)
    return np.column_stack([
        dose_values,
        np.log1p(safe_dose),
        np.sqrt(safe_dose),
    ])


def traditional_ml_models() -> Dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "svr_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=42,
        ),
        "gaussian_process": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4),
                alpha=1e-6,
                normalize_y=True,
                random_state=42,
            )),
        ]),
    }


def run_traditional_ml_baselines(data: Dict) -> Dict:
    x = feature_matrix(data["dex_concentration"])
    y = np.asarray(data["renin_normalized"], dtype=np.float64)
    dose_values = np.asarray(data["dex_concentration"], dtype=np.float64)
    dose_grid = np.logspace(-2, 2, 120)
    x_grid = feature_matrix(dose_grid)

    results = {
        "task": "24 h dose-response regression with leave-one-dose-out validation",
        "n_samples": int(len(y)),
        "models": {},
        "references": {},
    }

    if SW03_RESULTS_PATH.exists():
        sw03 = load_json(SW03_RESULTS_PATH)
        results["references"]["pinn_sw03"] = {
            "train_r2": sw03["ensemble_metrics"]["r2"],
            "train_rmse": sw03["ensemble_metrics"]["rmse"],
            "n_members": sw03["n_members"],
        }
    if ODE_RESULTS_PATH.exists():
        ode = load_json(ODE_RESULTS_PATH)
        metrics = ode.get("metrics", ode)
        results["references"]["ode_baseline"] = {
            "train_r2": metrics.get("r2"),
            "train_rmse": metrics.get("rmse"),
        }

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        dose_values,
        y,
        yerr=np.asarray(data["renin_std"], dtype=np.float64),
        fmt="o",
        color="black",
        label="Observed 24 h data",
        capsize=4,
    )

    for model_name, estimator in traditional_ml_models().items():
        estimator.fit(x, y)
        train_pred = estimator.predict(x)
        train_metrics = calculate_metrics(y, train_pred)

        held_out_predictions = np.zeros_like(y)
        folds = []
        for idx in range(len(y)):
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[idx] = False

            estimator_fold = traditional_ml_models()[model_name]
            estimator_fold.fit(x[train_mask], y[train_mask])
            pred = float(estimator_fold.predict(x[idx:idx + 1])[0])
            held_out_predictions[idx] = pred
            folds.append({
                "fold": idx + 1,
                "held_out_dose": float(dose_values[idx]),
                "observed": float(y[idx]),
                "predicted": pred,
                "absolute_error": float(abs(pred - y[idx])),
            })

        lodo_metrics = calculate_metrics(y, held_out_predictions)
        curve_pred = estimator.predict(x_grid)

        results["models"][model_name] = {
            "train_metrics": json_safe(train_metrics),
            "lodo_metrics": json_safe(lodo_metrics),
            "folds": folds,
            "train_predictions": train_pred.tolist(),
            "held_out_predictions": held_out_predictions.tolist(),
            "dose_response_curve": {
                "dose_grid": dose_grid.tolist(),
                "predictions": curve_pred.tolist(),
            },
        }

        plt.plot(dose_grid, curve_pred, label=model_name.replace("_", " ").title())

    plt.xscale("log")
    plt.xlabel("Dexamethasone concentration")
    plt.ylabel("Normalized renin")
    plt.title("Traditional ML baselines on the sparse 24 h dose-response task")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "traditional_ml_baselines.png", dpi=200)
    plt.close()

    summary_path = OUTPUT_DIR / "traditional_ml_baselines.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    save_traditional_ml_table(results)
    return results


def save_traditional_ml_table(results: Dict) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Traditional machine-learning baselines under leave-one-dose-out validation.}",
        r"\label{tab:traditional_ml_baselines}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Train R$^2$} & \textbf{LODO R$^2$} & \textbf{LODO RMSE} & \textbf{LODO MAE} \\",
        r"\hline",
    ]

    for model_name, info in results["models"].items():
        train_metrics = info["train_metrics"]
        lodo_metrics = info["lodo_metrics"]
        lines.append(
            f"{model_name.replace('_', ' ').title()} & "
            f"{train_metrics['r2']:.3f} & "
            f"{lodo_metrics['r2']:.3f} & "
            f"{lodo_metrics['rmse']:.3f} & "
            f"{lodo_metrics['mae']:.3f} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\multicolumn{5}{l}{\small Features: raw dose, log(1+dose), and sqrt(dose); all observations at 24 h.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (TABLES_DIR / "traditional_ml_baselines.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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


def train_reviewer_pinn_member(
    data: Dict,
    synthetic_weight: float,
    seed: int,
    device: str,
) -> Tuple[Dict, ReninPINN]:
    model = create_model(seed=seed)
    config = prepare_training_config(
        constraint_weight=0.005,
        synthetic_weight=synthetic_weight,
        variant_params=REVIEWER_VARIANT_PARAMS,
    )
    baseline_path = BASELINE_TEMPORAL_PATH if BASELINE_TEMPORAL_PATH.exists() else None

    trainer = UnifiedPINNTrainer(
        model=model,
        device=torch.device(device),
        learning_rate=1e-3,
        weight_decay=0.0,
        config=config,
        plausibility_config=default_plausibility_config(),
        early_stopping=relaxed_early_stopping(config.n_epochs),
        baseline_temporal_path=str(baseline_path) if baseline_path else None,
        parameter_targets=PARAMETER_TARGETS,
        seed=seed + 1000,
    )
    trainer.train(data)

    metrics = summarise_metrics(trainer.latest_metrics or {})
    params = summarise_parameters(trainer.model.get_params())
    plausibility = serialise_plausibility(trainer.latest_plausibility or {})

    entry = {
        "seed": seed,
        "metrics": metrics,
        "parameters": params,
        "plausibility": plausibility,
        "best_epoch": int(trainer.best_metrics.get("epoch", -1)),
        "stop_reason": trainer.stop_reason,
    }

    accepted = (
        entry["metrics"].get("r2") is not None
        and entry["metrics"]["r2"] >= 0.5
        and entry["plausibility"].get("all_passed", False)
    )
    entry["accepted"] = accepted

    return entry, trainer.model.to("cpu")


def summarize_ablation_config(entries: List[Dict], models: List[ReninPINN], data: Dict) -> Dict:
    accepted_entries = [entry for entry in entries if entry["accepted"]]
    summary = {
        "n_runs": len(entries),
        "accepted_runs": len(accepted_entries),
        "success_rate": float(100.0 * len(accepted_entries) / max(1, len(entries))),
        "accepted_member_metrics": None,
        "best_single_r2": None,
        "best_single_entry": None,
    }

    if entries:
        best_entry = max(entries, key=lambda item: item["metrics"].get("r2") or float("-inf"))
        summary["best_single_r2"] = best_entry["metrics"].get("r2")
        summary["best_single_entry"] = best_entry

    if accepted_entries and models:
        y_true = np.asarray(data["renin_normalized"], dtype=np.float32)
        t_data = np.asarray(data["time"], dtype=np.float32)
        dex_data = np.asarray(data["dex_concentration"], dtype=np.float32)
        ensemble_mean, ensemble_std = collect_predictions(models, t_data, dex_data, torch.device("cpu"))
        ensemble_metrics = calculate_metrics(y_true, ensemble_mean[:, 2])
        summary["accepted_member_metrics"] = {
            "r2": float(ensemble_metrics["r2"]),
            "rmse": float(ensemble_metrics["rmse"]),
            "mae": float(ensemble_metrics["mae"]),
            "prediction_std_mean": float(np.mean(ensemble_std[:, 2])),
            "ic50_mean": float(np.mean([entry["parameters"]["ic50"] for entry in accepted_entries])),
            "ic50_std": float(np.std([entry["parameters"]["ic50"] for entry in accepted_entries])),
            "hill_mean": float(np.mean([entry["parameters"]["hill"] for entry in accepted_entries])),
            "hill_std": float(np.std([entry["parameters"]["hill"] for entry in accepted_entries])),
            "ic50_gap_mean": float(np.mean([entry["parameters"]["ic50_gap"] for entry in accepted_entries])),
            "hill_gap_mean": float(np.mean([entry["parameters"]["hill_gap"] for entry in accepted_entries])),
        }

    return summary


def run_synthetic_ablation(data: Dict, device: str, n_runs: int = 5) -> Dict:
    synthetic_weights = [0.0, 0.2, 0.3, 0.5]
    results = {
        "description": "Synthetic-augmentation ablation with deterministic PINN training",
        "n_runs_per_setting": n_runs,
        "settings": {},
    }

    for weight_idx, synthetic_weight in enumerate(synthetic_weights):
        entries = []
        accepted_models = []
        for run_idx in range(n_runs):
            seed = 5000 + weight_idx * 100 + run_idx
            entry, model = train_reviewer_pinn_member(
                data=data,
                synthetic_weight=synthetic_weight,
                seed=seed,
                device=device,
            )
            entries.append(entry)
            if entry["accepted"]:
                accepted_models.append(model)
                checkpoint = MODELS_DIR / f"synthetic_sw_{str(synthetic_weight).replace('.', '_')}_seed_{seed}.pth"
                torch.save(model.state_dict(), checkpoint)

        results["settings"][str(synthetic_weight)] = {
            "synthetic_weight": synthetic_weight,
            "runs": entries,
            "summary": summarize_ablation_config(entries, accepted_models, data),
        }

    summary_path = OUTPUT_DIR / "synthetic_ablation.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(results), handle, indent=2)

    save_synthetic_ablation_table(results)
    plot_synthetic_ablation(results)
    return results


def save_synthetic_ablation_table(results: Dict) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Synthetic augmentation ablation for the deterministic PINN.}",
        r"\label{tab:synthetic_ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{SW} & \textbf{Accepted} & \textbf{Success \%} & \textbf{Ensemble R$^2$} & \textbf{RMSE} & \textbf{IC$_{50}$ Gap} & \textbf{Hill Gap} \\",
        r"\hline",
    ]

    for setting_key, info in results["settings"].items():
        summary = info["summary"]
        ensemble_metrics = summary["accepted_member_metrics"] or {}
        if ensemble_metrics:
            lines.append(
                f"{setting_key} & "
                f"{summary['accepted_runs']}/{summary['n_runs']} & "
                f"{summary['success_rate']:.1f} & "
                f"{ensemble_metrics['r2']:.3f} & "
                f"{ensemble_metrics['rmse']:.3f} & "
                f"{ensemble_metrics['ic50_gap_mean']:.3f} & "
                f"{ensemble_metrics['hill_gap_mean']:.3f} \\\\"
            )
        else:
            lines.append(
                f"{setting_key} & {summary['accepted_runs']}/{summary['n_runs']} & {summary['success_rate']:.1f} & -- & -- & -- & -- \\\\"
            )

    lines.extend([
        r"\hline",
        r"\multicolumn{7}{l}{\small SW=0.0 removes synthetic augmentation entirely; all other training settings are held fixed.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (TABLES_DIR / "synthetic_ablation.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_synthetic_ablation(results: Dict) -> None:
    weights = []
    success_rates = []
    r2_values = []

    for setting_key, info in results["settings"].items():
        weights.append(float(setting_key))
        success_rates.append(info["summary"]["success_rate"])
        ensemble_metrics = info["summary"]["accepted_member_metrics"]
        r2_values.append(ensemble_metrics["r2"] if ensemble_metrics else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(weights, success_rates, width=0.08, color="#4C78A8")
    axes[0].set_xlabel("Synthetic weight")
    axes[0].set_ylabel("Accepted runs (%)")
    axes[0].set_title("Training success versus synthetic weight")

    axes[1].plot(weights, r2_values, marker="o", color="#F58518")
    axes[1].set_xlabel("Synthetic weight")
    axes[1].set_ylabel("Accepted-ensemble R$^2$")
    axes[1].set_title("Accepted-model fit versus synthetic weight")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "synthetic_ablation.png", dpi=200)
    plt.close(fig)


def load_sw03_summary() -> Dict:
    if not SW03_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "results/unified_03/unified_ensemble_03_results.json not found. "
            "Run 9_ensemble_synthetic_03.py before the reviewer suite."
        )
    return load_json(SW03_RESULTS_PATH)


def summarize_single_model_vs_ensemble(sw03_summary: Dict) -> Dict:
    member_results = sw03_summary["member_results"]
    best_member = max(member_results, key=lambda item: item["metrics"].get("r2") or float("-inf"))
    ensemble_metrics = sw03_summary["ensemble_metrics"]
    observed = np.asarray(prepare_training_data(dataset="elisa", use_log_scale=False)["renin_normalized"], dtype=np.float64)
    best_predictions = np.asarray(best_member["predictions"], dtype=np.float64)
    ensemble_predictions = np.asarray(sw03_summary["ensemble_predictions"]["mean"], dtype=np.float64)

    dose_values = prepare_training_data(dataset="elisa", use_log_scale=False)["dex_concentration"]
    per_dose = []
    for idx, dose in enumerate(dose_values):
        per_dose.append({
            "dose": float(dose),
            "observed": float(observed[idx]),
            "best_single_prediction": float(best_predictions[idx]),
            "ensemble_prediction": float(ensemble_predictions[idx]),
            "best_single_abs_error": float(abs(best_predictions[idx] - observed[idx])),
            "ensemble_abs_error": float(abs(ensemble_predictions[idx] - observed[idx])),
        })

    results = {
        "best_single_member": {
            "member": best_member["member"],
            "metrics": best_member["metrics"],
            "parameters": best_member["parameters"],
        },
        "ensemble": {
            "n_members": sw03_summary["n_members"],
            "metrics": ensemble_metrics,
        },
        "per_dose_comparison": per_dose,
        "interpretation": (
            "The best single accepted model is reported descriptively only. "
            "The ensemble remains the main reported result because it averages "
            "across seed-dependent local minima."
        ),
    }

    (OUTPUT_DIR / "single_model_vs_ensemble.json").write_text(
        json.dumps(json_safe(results), indent=2),
        encoding="utf-8",
    )
    save_single_model_table(results)
    return results


def save_single_model_table(results: Dict) -> None:
    best_single = results["best_single_member"]
    ensemble = results["ensemble"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Best single accepted PINN member versus deterministic ensemble summary (SW=0.3).}",
        r"\label{tab:single_vs_ensemble}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Method} & \textbf{$R^2$} & \textbf{RMSE} & \textbf{MAE} & \textbf{IC$_{50}$ Gap} & \textbf{Hill Gap} \\",
        r"\hline",
        (
            f"Best single member (#{best_single['member']}) & "
            f"{best_single['metrics']['r2']:.3f} & "
            f"{best_single['metrics']['rmse']:.3f} & "
            f"{best_single['metrics']['mae']:.3f} & "
            f"{best_single['parameters']['ic50_gap']:.3f} & "
            f"{best_single['parameters']['hill_gap']:.3f} \\\\"
        ),
        (
            f"Ensemble mean (n={ensemble['n_members']}) & "
            f"{ensemble['metrics']['r2']:.3f} & "
            f"{ensemble['metrics']['rmse']:.3f} & "
            f"{ensemble['metrics']['mae']:.3f} & "
            f"{ensemble['metrics']['ic50_gap_mean']:.3f} & "
            f"{ensemble['metrics']['hill_gap_mean']:.3f} \\\\"
        ),
        r"\hline",
        r"\multicolumn{6}{l}{\small Single-model reporting is descriptive; manuscript conclusions should rely on the ensemble.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES_DIR / "single_model_vs_ensemble.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def load_sw03_models(device: str) -> List[ReninPINN]:
    checkpoints = sorted(
        path for path in SW03_MODEL_DIR.glob("unified_03_member_*.pth")
        if "fallback" not in path.name
    )
    models = []
    for checkpoint_path in checkpoints:
        model = ReninPINN(hidden_layers=[128, 128, 128, 128], activation="tanh")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def summarize_parameter_uncertainty(device: str) -> Dict:
    models = load_sw03_models(device)
    if not models:
        raise FileNotFoundError(
            "No accepted SW=0.3 checkpoints found in results/unified_03/models."
        )

    all_params = [model.get_params() for model in models]
    param_names = sorted(all_params[0].keys())
    summary = {
        "n_models": len(models),
        "parameters": {},
    }
    for param_name in param_names:
        values = np.asarray([params[param_name] for params in all_params], dtype=np.float64)
        summary["parameters"][param_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "cv_percent": float(100.0 * np.std(values) / np.mean(values)) if np.mean(values) != 0 else np.nan,
            "ci_95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "values": values.tolist(),
        }

    observed_data = prepare_training_data(dataset="elisa", use_log_scale=False)
    t_data = np.asarray(observed_data["time"], dtype=np.float32)
    dex_data = np.asarray(observed_data["dex_concentration"], dtype=np.float32)
    device_obj = torch.device(device)
    mean_pred, std_pred = collect_predictions(models, t_data, dex_data, device_obj)
    summary["prediction_spread_observed_doses"] = {
        "doses": dex_data.tolist(),
        "mean": mean_pred[:, 2].tolist(),
        "std": std_pred[:, 2].tolist(),
    }

    (OUTPUT_DIR / "parameter_uncertainty_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    save_parameter_table(summary)
    return summary


def save_parameter_table(summary: Dict) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Parameter spread across accepted SW=0.3 ensemble checkpoints.}",
        r"\label{tab:parameter_spread}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{Mean} & \textbf{Std. Dev.} & \textbf{95\% CI Lower} & \textbf{95\% CI Upper} \\",
        r"\hline",
    ]

    for param_name, stats in summary["parameters"].items():
        lines.append(
            f"{param_name} & {stats['mean']:.4f} & {stats['std']:.4f} & "
            f"{stats['ci_95'][0]:.4f} & {stats['ci_95'][1]:.4f} \\\\"
        )

    lines.extend([
        r"\hline",
        rf"\multicolumn{{5}}{{l}}{{\small Empirical spread across {summary['n_models']} accepted deterministic ensemble members.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (TABLES_DIR / "parameter_uncertainty_summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_coverage_summary(
    traditional_ml: Dict,
    synthetic_ablation: Dict,
    single_vs_ensemble: Dict,
    parameter_uncertainty: Dict,
) -> Dict:
    return {
        "reviewer_1": {
            "other_traditional_ml_models": {
                "status": "covered_by_experiment",
                "evidence": "results/reviewer_experiments/traditional_ml_baselines.json",
            },
            "ensemble_technique_and_single_model": {
                "status": "covered_by_experiment",
                "evidence": "results/reviewer_experiments/single_model_vs_ensemble.json",
            },
            "non_parametric_test": {
                "status": "covered_by_existing_result",
                "evidence": "results/statistical_analysis/wilcoxon_test_results.json",
            },
            "different_cross_validation": {
                "status": "covered_by_existing_result",
                "evidence": "results/supplementary_experiments/experiment_2_results.json",
            },
        },
        "reviewer_2": {
            "parameter_variability_and_uncertainty": {
                "status": "covered_by_experiment",
                "evidence": "results/reviewer_experiments/parameter_uncertainty_summary.json",
            },
            "synthetic_observations_removed_or_varied": {
                "status": "covered_by_experiment",
                "evidence": "results/reviewer_experiments/synthetic_ablation.json",
            },
            "loss_component_ablation": {
                "status": "covered_by_existing_result",
                "evidence": "results/comprehensive/ablation_study.json",
            },
            "overfitting_discussion_control": {
                "status": "covered_by_existing_result",
                "evidence": "results/pure_nn_baseline/pure_nn_results.json",
            },
        },
        "generated_files": {
            "traditional_ml": list(traditional_ml["models"].keys()),
            "synthetic_settings": list(synthetic_ablation["settings"].keys()),
            "single_model_member": single_vs_ensemble["best_single_member"]["member"],
            "n_parameter_models": parameter_uncertainty["n_models"],
        },
    }


def main() -> None:
    print("=" * 80)
    print("REVIEWER-REQUESTED EXPERIMENT SUITE")
    print("=" * 80)
    print(f"Device: {DEFAULT_DEVICE}")

    setup_directories()
    data = prepare_training_data(dataset="elisa", use_log_scale=False)

    print("\n[1/4] Running traditional ML baselines...")
    traditional_ml = run_traditional_ml_baselines(data)

    print("\n[2/4] Running synthetic augmentation ablation...")
    synthetic_ablation = run_synthetic_ablation(data, device=DEFAULT_DEVICE, n_runs=5)

    print("\n[3/4] Summarizing best single model versus ensemble...")
    sw03_summary = load_sw03_summary()
    single_vs_ensemble = summarize_single_model_vs_ensemble(sw03_summary)

    print("\n[4/4] Summarizing parameter spread across accepted ensemble checkpoints...")
    parameter_uncertainty = summarize_parameter_uncertainty(DEFAULT_DEVICE)

    coverage_summary = build_coverage_summary(
        traditional_ml=traditional_ml,
        synthetic_ablation=synthetic_ablation,
        single_vs_ensemble=single_vs_ensemble,
        parameter_uncertainty=parameter_uncertainty,
    )

    coverage_path = OUTPUT_DIR / "coverage_summary.json"
    with coverage_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(coverage_summary), handle, indent=2)

    print("\n[OK] Reviewer experiment suite complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
