#!/usr/bin/env python3
"""Family-score block regression and decomposition.

This implements the PDF-aligned family-score attribution step using the
decadal family-score table:

    family_score ~ thermal + demand + humid heat + solar + heatwave persistence

For each family, the script reports:
- full standardized ridge R2
- drop-one-block R2 loss
- grouped permutation R2 loss

GAM and mixed-effect models are recorded in a model-status table, but not used
as main results because the current dataset has one pathway and one city per
Koppen type.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_paper_common import FIGURES_DIR, ensure_figures_dir


SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY_DECADAL_CSV = SCRIPT_DIR / "paper_family_scores_decadal.csv"
MASTER_DECADAL_CSV = SCRIPT_DIR / "paper_master_panel_decadal.csv"
OUTPUT_CSV = SCRIPT_DIR / "paper_family_regression_decomposition.csv"
MODEL_STATUS_CSV = SCRIPT_DIR / "paper_family_regression_model_status.csv"
PLOT_OUTPUT = FIGURES_DIR / "family_regression_block_decomposition.png"
PLOT_DATA_OUTPUT = FIGURES_DIR / "family_regression_block_decomposition_data.csv"

RIDGE_ALPHA = 1.0
PERMUTATIONS = 500
RANDOM_SEED = 42

BLOCK_ORDER = [
    "thermal_shift",
    "demand_background",
    "humid_heat_stress",
    "solar_burden",
    "persistence_heatwave",
]

BLOCK_LABEL = {
    "thermal_shift": "Thermal\nshift",
    "demand_background": "Demand\nbackground",
    "humid_heat_stress": "Humid heat\nstress",
    "solar_burden": "Solar\nburden",
    "persistence_heatwave": "Heatwave\npersistence",
}

BLOCK_COLOR = {
    "thermal_shift": "#b55239",
    "demand_background": "#4f78a8",
    "humid_heat_stress": "#7b559c",
    "solar_burden": "#d1a33f",
    "persistence_heatwave": "#4f9a68",
}

FAMILY_ORDER = [
    "habitability",
    "operational_adequacy",
    "durability_cautious",
    "energy_burden_secondary",
]

FAMILY_LABEL = {
    "habitability": "Habitability",
    "operational_adequacy": "Operational adequacy",
    "durability_cautious": "Durability cautious",
    "energy_burden_secondary": "Energy burden secondary",
}


def r2_score(y: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 1e-12:
        return float("nan")
    residual = float(np.sum((y - predicted) ** 2))
    return max(0.0, min(1.0, 1.0 - residual / total))


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    return (values - mean) / std, mean, std


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, np.ndarray, float]:
    x_scaled, _, _ = standardize(x)
    y_scaled = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)
    regularizer = alpha * np.eye(x_scaled.shape[1])
    beta = np.linalg.solve(x_scaled.T @ x_scaled + regularizer, x_scaled.T @ y_scaled)
    predicted = x_scaled @ beta
    return beta, predicted, r2_score(y_scaled, predicted)


def prepare_model_table() -> pd.DataFrame:
    family = pd.read_csv(FAMILY_DECADAL_CSV)
    master = pd.read_csv(MASTER_DECADAL_CSV)

    drivers = master[master["role"] == "driver"].copy()
    drivers["delta_std_mean"] = pd.to_numeric(drivers["delta_std_mean"], errors="coerce")
    block = (
        drivers.groupby(["scenario", "building", "city", "climate_type", "decade", "subfamily"], dropna=False)
        ["delta_std_mean"]
        .mean()
        .reset_index()
    )
    block = block[block["subfamily"].isin(BLOCK_ORDER)]
    wide = block.pivot_table(
        index=["scenario", "building", "city", "climate_type", "decade"],
        columns="subfamily",
        values="delta_std_mean",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    for block_name in BLOCK_ORDER:
        if block_name not in wide.columns:
            wide[block_name] = 0.0
    wide[BLOCK_ORDER] = wide[BLOCK_ORDER].fillna(0.0)

    family = family.copy()
    family["family_score_mean"] = pd.to_numeric(family["family_score_mean"], errors="coerce")
    data = family.merge(wide, on=["scenario", "building", "city", "climate_type", "decade"], how="left")
    return data


def decompose_family(data: pd.DataFrame, family_name: str) -> list[dict[str, object]]:
    subset = data[data["score_family"] == family_name].copy()
    subset = subset.dropna(subset=["family_score_mean", *BLOCK_ORDER])
    if len(subset) < len(BLOCK_ORDER) + 8:
        return [
            {
                "score_family": family_name,
                "model": "standardized_ridge_block_regression",
                "status": "insufficient_rows",
                "n": len(subset),
            }
        ]

    y = subset["family_score_mean"].to_numpy(dtype=float)
    x = subset[BLOCK_ORDER].to_numpy(dtype=float)
    beta, predicted, full_r2 = fit_ridge(x, y)
    full_beta2 = {block: float(value) ** 2 for block, value in zip(BLOCK_ORDER, beta)}
    beta2_total = sum(full_beta2.values())

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for block_index, block in enumerate(BLOCK_ORDER):
        keep_indices = [idx for idx in range(len(BLOCK_ORDER)) if idx != block_index]
        _, _, reduced_r2 = fit_ridge(x[:, keep_indices], y)
        drop_loss = max(0.0, full_r2 - reduced_r2)

        losses = []
        for _ in range(PERMUTATIONS):
            x_perm = x.copy()
            x_perm[:, block_index] = rng.permutation(x_perm[:, block_index])
            x_perm_scaled, _, _ = standardize(x_perm)
            y_scaled = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)
            permuted_pred = x_perm_scaled @ beta
            losses.append(max(0.0, full_r2 - r2_score(y_scaled, permuted_pred)))

        rows.append(
            {
                "score_family": family_name,
                "model": "standardized_ridge_block_regression",
                "status": "completed",
                "n": len(subset),
                "ridge_alpha": RIDGE_ALPHA,
                "full_r2": full_r2,
                "block": block,
                "block_label": BLOCK_LABEL[block].replace("\n", " "),
                "standardized_beta": float(beta[block_index]),
                "beta2_share": (full_beta2[block] / beta2_total) if beta2_total > 1e-12 else np.nan,
                "drop_one_block_reduced_r2": reduced_r2,
                "drop_one_block_r2_loss": drop_loss,
                "permutation_r2_loss_mean": float(np.mean(losses)),
                "permutation_r2_loss_sd": float(np.std(losses)),
                "permutation_n": PERMUTATIONS,
                "interpretation_tier": (
                    "main"
                    if family_name in {"habitability", "operational_adequacy"}
                    else ("cautious_supplementary" if family_name == "durability_cautious" else "secondary_overlay")
                ),
            }
        )
    return rows


def build_model_status(data: pd.DataFrame) -> pd.DataFrame:
    scenario_count = data["scenario"].nunique()
    city_count = data["city"].nunique()
    family_count = data["score_family"].nunique()
    rows = [
        {
            "model": "family_score_standardized_ridge_block_regression",
            "status": "completed",
            "reason": "Uses decadal family scores and five driver-block predictors; suitable for current small prototype panel.",
            "output": str(OUTPUT_CSV.name),
        },
        {
            "model": "drop_one_block_r2_decomposition",
            "status": "completed",
            "reason": "Computed as full ridge R2 minus reduced ridge R2 with one block removed.",
            "output": str(OUTPUT_CSV.name),
        },
        {
            "model": "grouped_permutation_importance",
            "status": "completed",
            "reason": f"Computed with {PERMUTATIONS} permutations per block using the fitted standardized ridge model.",
            "output": str(OUTPUT_CSV.name),
        },
        {
            "model": "GAM",
            "status": "not_used_as_main_result",
            "reason": (
                "Current panel is small for smooth-term inference after decadal aggregation "
                f"({city_count} cities, {scenario_count} pathway, {family_count} score families). "
                "Use only after adding pathways/cities or keep as exploratory."
            ),
            "output": "",
        },
        {
            "model": "mixed_effect_model",
            "status": "not_used_as_main_result",
            "reason": (
                "City random effects are confounded with Koppen type because each Koppen type has one representative city; "
                f"pathway random/fixed effects are unavailable because scenario_count={scenario_count}."
            ),
            "output": "",
        },
    ]
    return pd.DataFrame(rows)


def plot_decomposition(results: pd.DataFrame) -> None:
    plot_data = results[results["status"] == "completed"].copy()
    plot_data = plot_data[plot_data["score_family"].isin(FAMILY_ORDER)]
    plot_data["family_order"] = plot_data["score_family"].map({name: idx for idx, name in enumerate(FAMILY_ORDER)})
    plot_data["block_order"] = plot_data["block"].map({name: idx for idx, name in enumerate(BLOCK_ORDER)})
    plot_data = plot_data.sort_values(["family_order", "block_order"])
    plot_data.to_csv(PLOT_DATA_OUTPUT, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharey=True)
    families = [family for family in FAMILY_ORDER if family in set(plot_data["score_family"])]
    y_positions = np.arange(len(families))
    bar_height = 0.13
    offsets = np.linspace(-0.28, 0.28, len(BLOCK_ORDER))

    for ax, value_col, title in [
        (axes[0], "drop_one_block_r2_loss", "Drop-one-block R2 loss"),
        (axes[1], "permutation_r2_loss_mean", "Grouped permutation R2 loss"),
    ]:
        for block, offset in zip(BLOCK_ORDER, offsets):
            values = []
            for family in families:
                row = plot_data[(plot_data["score_family"] == family) & (plot_data["block"] == block)]
                values.append(float(row.iloc[0][value_col]) if not row.empty else 0.0)
            ax.barh(y_positions + offset, values, height=bar_height, color=BLOCK_COLOR[block], label=BLOCK_LABEL[block].replace("\n", " "))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("R2 loss")
        ax.grid(True, axis="x", color="#e2e7ea", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([FAMILY_LABEL[family] for family in families])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.2)
    fig.suptitle("Family-score regression decomposition by driver block", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.02,
        0.04,
        "Models use decadal family scores and standardized driver-block predictors. Durability and energy burden are supplementary/secondary.",
        fontsize=8,
        color="#455a64",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(PLOT_OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_figures_dir()
    data = prepare_model_table()
    rows: list[dict[str, object]] = []
    for family_name in FAMILY_ORDER:
        rows.extend(decompose_family(data, family_name))
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    build_model_status(data).to_csv(MODEL_STATUS_CSV, index=False)
    plot_decomposition(results)
    print(f"Saved -> {OUTPUT_CSV}")
    print(f"Saved -> {MODEL_STATUS_CSV}")
    print(f"Saved -> {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
