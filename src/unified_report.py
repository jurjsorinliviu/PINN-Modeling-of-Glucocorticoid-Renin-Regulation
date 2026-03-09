"""
Generate a consolidated report based on calibration and unified ensemble results.
"""

import json
import math
from pathlib import Path
from typing import Dict, List

CALIBRATION_PATH = Path("results/calibration/calibration_summary.json")
ENSEMBLE_PATH = Path("results/unified/unified_ensemble_results.json")
REPORT_PATH = Path("results/unified/unified_report.txt")


def load_json(path: Path) -> Dict:
    if path.exists():
        with path.open("r") as handle:
            return json.load(handle)
    return {}


def fmt_numeric(value, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def format_constraint_section(calibration: Dict) -> List[str]:
    lines: List[str] = []
    sweep = calibration.get("constraint_sweep", [])
    if not sweep:
        return lines

    lines.append("Constraint Weight Sweep:")
    lines.append("  weight   r2      rmse    gap_IC50  gap_Hill")
    for item in sweep:
        metrics = item.get("metrics", {})
        weight = item.get("constraint_weight")
        lines.append(
            f"  {fmt_numeric(weight, 4):>6}  {fmt_numeric(metrics.get('r2')):>6}  "
            f"{fmt_numeric(metrics.get('rmse')):>6}  "
            f"{fmt_numeric(metrics.get('ic50_gap')):>9}  {fmt_numeric(metrics.get('hill_gap')):>9}"
        )
    best = calibration.get("best_constraint", {})
    if best:
        lines.append(f"  -> Selected constraint weight: {best.get('constraint_weight')}")
    lines.append("")
    return lines


def format_synthetic_section(calibration: Dict) -> List[str]:
    lines: List[str] = []
    sweep = calibration.get("synthetic_sweep", [])
    if not sweep:
        return lines

    lines.append("Synthetic Weight Sweep:")
    lines.append("  weight   r2      rmse    plausibility")
    for item in sweep:
        metrics = item.get("metrics", {})
        plaus = item.get("plausibility", {})
        weight = item.get("synthetic_weight")
        lines.append(
            f"  {fmt_numeric(weight, 2):>6}  {fmt_numeric(metrics.get('r2')):>6}  "
            f"{fmt_numeric(metrics.get('rmse')):>6}  "
            f"{'PASS' if plaus.get('all_passed') else 'FAIL'}"
        )
    best = calibration.get("best_synthetic", {})
    if best:
        lines.append(f"  -> Selected synthetic weight: {best.get('synthetic_weight')}")
    lines.append("")
    return lines


def format_pretraining_section(calibration: Dict) -> List[str]:
    lines: List[str] = []
    sweep = calibration.get("pretraining_sweep", [])
    if not sweep:
        return lines

    lines.append("Pretraining Length Sweep (parameter drift):")
    lines.append("  epochs   pre_gap_IC50  pre_gap_Hill  final r2  final rmse")
    for item in sweep:
        pre = item.get("parameters_after_pretraining") or {}
        final_metrics = item.get("final_metrics", {})
        lines.append(
            f"  {item.get('pretraining_epochs'):>6}  "
            f"{fmt_numeric(pre.get('ic50_gap')):>13}  {fmt_numeric(pre.get('hill_gap')):>12}  "
            f"{fmt_numeric(final_metrics.get('r2')):>8}  {fmt_numeric(final_metrics.get('rmse')):>10}"
        )
    lines.append("")
    return lines


def format_ensemble_section(ensemble: Dict) -> List[str]:
    lines: List[str] = []
    metrics = ensemble.get("ensemble_metrics", {})
    residuals = ensemble.get("residual_diagnostics", {})
    pareto = ensemble.get("pareto", {})
    figures = ensemble.get("figures", {})

    lines.append("Unified Ensemble Summary:")
    lines.append(f"  Members: {ensemble.get('n_members')}")
    lines.append(
        f"  R2={fmt_numeric(metrics.get('r2'))}  RMSE={fmt_numeric(metrics.get('rmse'))}  "
        f"gap(IC50)={fmt_numeric(metrics.get('ic50_gap_mean'))}+/-{fmt_numeric(metrics.get('ic50_gap_std'))}  "
        f"gap(Hill)={fmt_numeric(metrics.get('hill_gap_mean'))}+/-{fmt_numeric(metrics.get('hill_gap_std'))}"
    )
    normality = residuals.get("normality", {}) if isinstance(residuals, dict) else {}
    hetero = residuals.get("heteroscedasticity", {}) if isinstance(residuals, dict) else {}
    autocorr = residuals.get("autocorrelation", {}) if isinstance(residuals, dict) else {}
    lines.append("  Residual diagnostics:")
    lines.append(f"    Shapiro-Wilk p={fmt_numeric(normality.get('shapiro_p_value'))}")
    lines.append(f"    Jarque-Bera p={fmt_numeric(normality.get('jb_p_value'))}")
    lines.append(f"    Breusch-Pagan p={fmt_numeric(hetero.get('breusch_pagan_p_value'))}")
    lines.append(f"    Durbin-Watson={fmt_numeric(autocorr.get('durbin_watson'))}")

    knee = pareto.get("knee_point", {})
    if knee:
        lines.append(
            f"  Pareto knee at member {knee.get('member')} "
            f"(gap={fmt_numeric(knee.get('parameter_gap'))}, r2={fmt_numeric(knee.get('r2'))})"
        )
    lines.append("  Figures:")
    for name, path in figures.items():
        lines.append(f"    {name}: {path}")
    lines.append("")
    return lines


def generate_report(report_path: Path = REPORT_PATH):
    calibration = load_json(CALIBRATION_PATH)
    ensemble = load_json(ENSEMBLE_PATH)

    lines: List[str] = []
    lines.append("Unified PINN Calibration & Ensemble Report")
    lines.append("=" * 60)
    lines.append("")

    selected = calibration.get("selected_weights", {})
    if selected:
        lines.append("Selected calibration weights:")
        lines.append(f"  Constraint weight: {selected.get('constraint')}")
        lines.append(f"  Synthetic weight:  {selected.get('synthetic')}")
        lines.append("")

    lines.extend(format_constraint_section(calibration))
    lines.extend(format_synthetic_section(calibration))
    lines.extend(format_pretraining_section(calibration))
    lines.extend(format_ensemble_section(ensemble))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        handle.write("\n".join(lines))
    print(f"Unified report written to {report_path}")


if __name__ == "__main__":
    generate_report()
