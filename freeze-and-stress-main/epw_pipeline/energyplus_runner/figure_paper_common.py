"""Shared helpers for paper-facing figure scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "metric_exports" / "figures"

SUMMARY_CSV = SCRIPT_DIR / "paper_metrics_summary.csv"
REGISTRY_CSV = SCRIPT_DIR / "paper_metric_registry.csv"
SCREENING_CSV = SCRIPT_DIR / "paper_metric_screening.csv"
CLIMATE_SCREENING_CSV = SCRIPT_DIR / "paper_screening_by_climate_type.csv"
BLOCK_DECOMP_CSV = SCRIPT_DIR / "paper_block_decomposition.csv"

CLIMATE_ORDER = ["Am", "BWh", "Csb", "Cfb", "Dfa", "Dfb"]
CLIMATE_CITY = {
    "Am": "Miami",
    "BWh": "Phoenix",
    "Csb": "Los_Angeles",
    "Cfb": "Vancouver",
    "Dfa": "Toronto",
    "Dfb": "Montreal",
}
CITY_CLIMATE = {city: climate for climate, city in CLIMATE_CITY.items()}
CLIMATE_LABEL = {
    "Am": "Am\nMiami",
    "BWh": "BWh\nPhoenix",
    "Csb": "Csb\nLos Angeles",
    "Cfb": "Cfb\nVancouver",
    "Dfa": "Dfa\nToronto",
    "Dfb": "Dfb\nMontreal",
}

SUBFAMILY_LABEL = {
    "habitability": "Habitability",
    "operational_adequacy": "Operational\nadequacy",
    "energy_burden": "Energy\nburden",
    "durability": "Durability",
}
SUBFAMILY_COLOR = {
    "habitability": "#b44b5f",
    "operational_adequacy": "#4f6d8a",
    "energy_burden": "#8f7a3d",
    "durability": "#5b8a72",
}
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


def ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_year(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "never":
        return None
    return int(float(text))


def short_metric_label(label: str) -> str:
    replacements = {
        "Cooling setpoint not met occupied hours": "Cooling unmet\noccupied hours",
        "Heating unmet hours": "Heating\nunmet hours",
        "Annual cooling electricity": "Annual cooling\nelectricity",
        "Annual heating electricity": "Annual heating\nelectricity",
        "Peak operative temperature": "Peak operative\ntemperature",
        "Maximum daily EDH": "Maximum\ndaily EDH",
        "Daily EDH exceedance days": "Daily EDH\nexceedance days",
        "Cooling peak demand": "Cooling peak\ndemand",
        "Heating peak demand": "Heating peak\ndemand",
        "Maximum delta_t": "Maximum\ndelta T",
        "Indoor RH max": "Indoor RH\nmax",
        "Annual EDH": "Annual\nEDH",
    }
    return replacements.get(label, label.replace(" ", "\n", 1))


def response_metrics(registry: pd.DataFrame, screening: pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = registry[registry["role"] == "response"].copy()
    if screening is not None and "screening_selected_for_analysis" in screening.columns:
        selected = set(
            screening[
                (screening["role"] == "response")
                & (screening["screening_selected_for_analysis"].astype(str).str.lower() == "yes")
            ]["metric"]
        )
        metrics = metrics[metrics["metric"].isin(selected)].copy()
    family_order = {
        "habitability": 0,
        "operational_adequacy": 1,
        "energy_burden": 2,
        "durability": 3,
    }
    metrics["family_order"] = metrics["subfamily"].map(family_order).fillna(99)
    metrics["failure_order"] = ~metrics["failure_relevant"].astype(bool)
    return metrics.sort_values(["family_order", "failure_order", "metric"]).reset_index(drop=True)


def metric_prefix(row: pd.Series) -> str:
    return f"{row['family']}__{row['subfamily']}__{row['metric']}"


def first_trigger_keys(row: pd.Series, label_to_metric: dict[str, str]) -> set[str]:
    trigger = str(row.get("first_trigger_metric", "")).strip()
    if not trigger or trigger == "Never":
        return set()
    return {label_to_metric.get(part.strip(), part.strip()) for part in trigger.split(";")}


def wrap_text(text: object, width: int) -> str:
    words = str(text).replace("_", " ").split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)
