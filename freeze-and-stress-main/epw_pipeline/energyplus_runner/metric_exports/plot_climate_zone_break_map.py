#!/usr/bin/env python3
"""
Climate-Zone Break-Year Propagation Map for North America.

Maps ~80 major cities to their nearest Köppen prototype city, then shows
progressive building-energy-system failure across 2040 / 2070 / 2100 snapshots.

Prototype cities and their break years (RCP 8.5, office building):
  Phoenix   (BWh) — 2025    Miami      (Am)  — 2026
  Los Angeles (Csb) — 2028  Montreal   (Dfb) — 2030
  Toronto   (Dfa) — 2045    Vancouver  (Cfb) — Never
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_DIR = SCRIPT_DIR.parent
SUMMARY_CSV = RUNNER_DIR / "paper_metrics_summary.csv"
ANNUAL_METRICS_CSV = SCRIPT_DIR / "annual_metrics_extended.csv"
if not ANNUAL_METRICS_CSV.exists():
    ANNUAL_METRICS_CSV = SCRIPT_DIR / "annual_metrics.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
OUTPUT_PNG = FIGURES_DIR / "climate_zone_break_propagation.png"
MAP_XLIM = (-170, -50)
MAP_YLIM = (13, 72)
GRID_SHAPE = (260, 420)
PRIMARY_BREAK_YEAR_COLUMN = "failure_occurrence_5y_year"
PRIMARY_BREAK_YEAR_LABEL = "5Y rolling >=25% failure occurrence"
STRESS_COLORBAR_MAX = 0.60

# ---------------------------------------------------------------------------
# Break years from simulation (RCP 8.5, office, REMO2015)
# ---------------------------------------------------------------------------
FALLBACK_PROTOTYPE_BREAK_YEARS: dict[str, int | str] = {
    "Phoenix": 2029,
    "Miami": 2029,
    "Los_Angeles": 2044,
    "Montreal": 2081,
    "Toronto": "Never",
    "Vancouver": 2032,
}

STANDARD_RESPONSE_THRESHOLDS: dict[str, float] = {
    "annual_edh_c_h": 300.0,
    "daily_edh_max_k_h": 12.0,
    "daily_edh_exceed_6kh_count": 15.0,
    "facility_cooling_setpoint_not_met_occupied_time_total_hours": 500.0,
    "abups_occupied_heating_not_met_hours": 400.0,
}

RELATIVE_STRESS_COLUMNS = [
    "outdoor_drybulb_c_mean",
    "summer_mean_drybulb_c",
    "cdd_18",
    "hdd_18",
    "outdoor_wetbulb_c_max",
    "annual_ghi_kwh_m2",
    "peak_ghi_w_m2",
    "heatwave_days",
    "max_consec_hot_days_35c",
    "rep_zone_operative_temp_c_max",
    "delta_t_max_k",
    "cooling_electricity_peak_kw",
    "heating_electricity_peak_kw",
    "cooling_electricity_annual_kwh",
    "heating_electricity_annual_kwh",
    "rep_zone_rh_pct_max",
]

# ---------------------------------------------------------------------------
# Köppen → prototype mapping
# ---------------------------------------------------------------------------
KOPPEN_TO_PROTOTYPE: dict[str, str] = {
    # Tropical → Miami
    "Af": "Miami", "Am": "Miami", "Aw": "Miami",
    # Arid → Phoenix
    "BWh": "Phoenix", "BWk": "Phoenix", "BSh": "Phoenix", "BSk": "Phoenix",
    # Temperate – humid subtropical → Miami
    "Cfa": "Miami",
    # Temperate – Mediterranean → Los Angeles
    "Csa": "Los_Angeles", "Csb": "Los_Angeles",
    # Temperate – oceanic → Vancouver
    "Cfb": "Vancouver", "Cfc": "Vancouver",
    # Continental – hot summer → Toronto
    "Dfa": "Toronto", "Dwa": "Toronto",
    # Continental – warm/cool summer → Montreal
    "Dfb": "Montreal", "Dwb": "Montreal",
    "Dfc": "Montreal", "Dwc": "Montreal",
    "Dfd": "Montreal",
}

# ---------------------------------------------------------------------------
# Prototype visual identity
# ---------------------------------------------------------------------------
PROTOTYPE_COLORS: dict[str, str] = {
    "Phoenix": "#d95f02",
    "Miami": "#e7298a",
    "Los_Angeles": "#7570b3",
    "Montreal": "#1b9e77",
    "Toronto": "#66a61e",
    "Vancouver": "#377eb8",
}

PROTOTYPE_DISPLAY: dict[str, str] = {
    "Phoenix": "Phoenix (BWh)",
    "Miami": "Miami (Am)",
    "Los_Angeles": "Los Angeles (Csb)",
    "Montreal": "Montreal (Dfb)",
    "Toronto": "Toronto (Dfa)",
    "Vancouver": "Vancouver (Cfb)",
}

# ---------------------------------------------------------------------------
# North American cities database  (city, lat, lon, Köppen, is_prototype)
# ---------------------------------------------------------------------------
CITIES_DB: list[tuple[str, float, float, str, bool]] = [
    # ── Prototype cities ──────────────────────────────────────────────────
    ("Los Angeles",   34.05, -118.24, "Csb", True),
    ("Miami",         25.76,  -80.19, "Am",  True),
    ("Montreal",      45.50,  -73.57, "Dfb", True),
    ("Phoenix",       33.45, -112.07, "BWh", True),
    ("Toronto",       43.65,  -79.38, "Dfa", True),
    ("Vancouver",     49.28, -123.12, "Cfb", True),
    # ── USA — Tropical / Humid subtropical (Cfa → Miami) ─────────────────
    ("Houston",       29.76,  -95.37, "Cfa", False),
    ("New Orleans",   29.95,  -90.07, "Cfa", False),
    ("Atlanta",       33.75,  -84.39, "Cfa", False),
    ("Dallas",        32.78,  -96.80, "Cfa", False),
    ("Charlotte",     35.23,  -80.84, "Cfa", False),
    ("Nashville",     36.16,  -86.78, "Cfa", False),
    ("Memphis",       35.15,  -90.05, "Cfa", False),
    ("Jacksonville",  30.33,  -81.66, "Cfa", False),
    ("Tampa",         27.95,  -82.46, "Cfa", False),
    ("Orlando",       28.54,  -81.38, "Cfa", False),
    ("San Antonio",   29.42,  -98.49, "Cfa", False),
    ("Austin",        30.27,  -97.74, "Cfa", False),
    ("Raleigh",       35.78,  -78.64, "Cfa", False),
    ("Richmond",      37.54,  -77.44, "Cfa", False),
    ("Washington DC", 38.91,  -77.04, "Cfa", False),
    ("St. Louis",     38.63,  -90.20, "Cfa", False),
    ("Kansas City",   39.10,  -94.58, "Cfa", False),
    ("Oklahoma City", 35.47,  -97.52, "Cfa", False),
    ("Little Rock",   34.75,  -92.29, "Cfa", False),
    ("Birmingham",    33.52,  -86.81, "Cfa", False),
    ("Louisville",    38.25,  -85.76, "Cfa", False),
    ("Norfolk",       36.85,  -76.29, "Cfa", False),
    # ── USA — Arid / Semi-arid (→ Phoenix) ───────────────────────────────
    ("Las Vegas",     36.17, -115.14, "BWh", False),
    ("Tucson",        32.22, -110.93, "BWh", False),
    ("El Paso",       31.76, -106.44, "BWh", False),
    ("Albuquerque",   35.08, -106.65, "BSk", False),
    ("Denver",        39.74, -104.98, "BSk", False),
    ("Salt Lake City",40.76, -111.89, "BSk", False),
    ("Boise",         43.62, -116.21, "BSk", False),
    ("Reno",          39.53, -119.81, "BSk", False),
    # ── USA — Mediterranean (→ Los Angeles) ────────────────────────────
    ("San Francisco", 37.77, -122.42, "Csb", False),
    ("San Diego",     32.72, -117.16, "Csb", False),
    ("Sacramento",    38.58, -121.49, "Csa", False),
    ("San Jose",      37.34, -121.89, "Csb", False),
    ("Fresno",        36.74, -119.77, "Csa", False),
    ("Portland",      45.52, -122.68, "Csb", False),
    ("Santa Barbara", 34.42, -119.70, "Csb", False),
    # ── USA — Oceanic (→ Vancouver) ──────────────────────────────────────
    ("Seattle",       47.61, -122.33, "Cfb", False),
    ("Olympia",       47.04, -122.90, "Cfb", False),
    # ── USA — Hot-summer continental (Dfa → Toronto) ─────────────────────
    ("New York",      40.71,  -74.01, "Dfa", False),
    ("Chicago",       41.88,  -87.63, "Dfa", False),
    ("Philadelphia",  39.95,  -75.17, "Dfa", False),
    ("Boston",        42.36,  -71.06, "Dfa", False),
    ("Detroit",       42.33,  -83.05, "Dfa", False),
    ("Indianapolis",  39.77,  -86.16, "Dfa", False),
    ("Columbus",      39.96,  -82.99, "Dfa", False),
    ("Cincinnati",    39.10,  -84.51, "Dfa", False),
    ("Pittsburgh",    40.44,  -79.99, "Dfa", False),
    ("Cleveland",     41.50,  -81.69, "Dfa", False),
    ("Milwaukee",     43.04,  -87.91, "Dfa", False),
    ("Hartford",      41.76,  -72.68, "Dfa", False),
    ("Buffalo",       42.89,  -78.88, "Dfa", False),
    # ── USA — Warm-summer continental (Dfb → Montreal) ───────────────────
    ("Minneapolis",   44.98,  -93.27, "Dfb", False),
    ("Duluth",        46.79,  -92.10, "Dfb", False),
    ("Fargo",         46.88,  -96.79, "Dfb", False),
    ("Burlington VT", 44.48,  -73.21, "Dfb", False),
    ("Anchorage",     61.22, -149.90, "Dfc", False),
    ("Fairbanks",     64.84, -147.72, "Dfc", False),
    # ── Canada ───────────────────────────────────────────────────────────
    ("Calgary",       51.05, -114.07, "Dfb", False),
    ("Edmonton",      53.55, -113.49, "Dfb", False),
    ("Winnipeg",      49.90,  -97.14, "Dfb", False),
    ("Ottawa",        45.42,  -75.70, "Dfb", False),
    ("Quebec City",   46.81,  -71.21, "Dfb", False),
    ("Halifax",       44.65,  -63.57, "Dfb", False),
    ("Saskatoon",     52.13, -106.67, "Dfb", False),
    ("Regina",        50.45, -104.62, "Dfb", False),
    ("Victoria",      48.43, -123.37, "Cfb", False),
    ("Hamilton",      43.26,  -79.87, "Dfa", False),
    ("London ON",     42.98,  -81.25, "Dfa", False),
    ("Kelowna",       49.89, -119.50, "Dfb", False),
    ("St. John's",    47.56,  -52.71, "Dfb", False),
    ("Thunder Bay",   48.38,  -89.25, "Dfb", False),
    ("Whitehorse",    60.72, -135.06, "Dfc", False),
    ("Yellowknife",   62.45, -114.37, "Dfc", False),
    # ── Mexico (select cities) ───────────────────────────────────────────
    ("Mexico City",   19.43,  -99.13, "Cwb", False),
    ("Monterrey",     25.67, -100.31, "BSh", False),
    ("Guadalajara",   20.67, -103.35, "Cwa", False),
    ("Tijuana",       32.51, -117.02, "BSk", False),
    ("Hermosillo",    29.07, -110.97, "BWh", False),
    ("Cancun",        21.16,  -86.85, "Aw",  False),
    ("Merida",        20.97,  -89.59, "Aw",  False),
]

# Map uncommon Köppen codes that aren't in KOPPEN_TO_PROTOTYPE
_EXTRA_KOPPEN: dict[str, str] = {
    "Cwb": "Montreal",   # subtropical highland → cool continental proxy
    "Cwa": "Miami",      # monsoon-influenced humid subtropical
}
KOPPEN_TO_PROTOTYPE.update(_EXTRA_KOPPEN)


def parse_break_year(value: object) -> int | str:
    if value in (None, "", "Never"):
        return "Never"
    return int(float(value))


def load_prototype_break_years() -> dict[str, int | str]:
    if not SUMMARY_CSV.exists():
        return FALLBACK_PROTOTYPE_BREAK_YEARS.copy()

    summary = pd.read_csv(SUMMARY_CSV)
    result = FALLBACK_PROTOTYPE_BREAK_YEARS.copy()
    break_column = PRIMARY_BREAK_YEAR_COLUMN if PRIMARY_BREAK_YEAR_COLUMN in summary.columns else "main_break_year"
    for _, row in summary.iterrows():
        city = str(row.get("city", ""))
        if city in result:
            result[city] = parse_break_year(row.get(break_column))
    return result


def nearest_year_row(group: pd.DataFrame, snapshot: int) -> pd.Series:
    years = pd.to_numeric(group["year"], errors="coerce")
    eligible = group[years <= snapshot]
    if eligible.empty:
        eligible = group.iloc[[int(np.nanargmin((years - snapshot).abs().to_numpy()))]]
    return eligible.sort_values("year").iloc[-1]


def load_prototype_stress_scores(snapshot_years: list[int]) -> dict[int, dict[str, float]]:
    if not ANNUAL_METRICS_CSV.exists():
        return {year: {proto: 0.0 for proto in FALLBACK_PROTOTYPE_BREAK_YEARS} for year in snapshot_years}

    df = pd.read_csv(ANNUAL_METRICS_CSV)
    if "building" in df.columns:
        df = df[df["building"] == "office"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    relative_thresholds: dict[str, float] = {}
    for column in RELATIVE_STRESS_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            relative_thresholds[column] = float(np.percentile(values, 90))

    scores: dict[int, dict[str, float]] = {year: {} for year in snapshot_years}
    for city, group in df.groupby("city"):
        city_key = str(city)
        if city_key not in FALLBACK_PROTOTYPE_BREAK_YEARS:
            continue
        group = group.dropna(subset=["year"]).sort_values("year")
        if group.empty:
            continue
        for snapshot in snapshot_years:
            row = nearest_year_row(group, snapshot)

            standard_excess: list[float] = []
            for column, threshold in STANDARD_RESPONSE_THRESHOLDS.items():
                if column not in row or pd.isna(row[column]):
                    continue
                ratio = float(row[column]) / threshold if threshold else 0.0
                standard_excess.append(min(max(ratio - 1.0, 0.0), 1.0))
            mean_excess = float(np.mean(standard_excess)) if standard_excess else 0.0
            max_excess = float(np.max(standard_excess)) if standard_excess else 0.0
            standard_index = 0.5 * mean_excess + 0.5 * max_excess

            relative_flags = [
                float(row[column]) > threshold
                for column, threshold in relative_thresholds.items()
                if column in row and not pd.isna(row[column])
            ]
            relative_index = float(np.mean(relative_flags)) if relative_flags else 0.0
            scores[snapshot][city_key] = min(1.0, 0.65 * standard_index + 0.35 * relative_index)
    return scores


def build_city_dataframe(prototype_break_years: dict[str, int | str]) -> pd.DataFrame:
    """Build a DataFrame of all cities with prototype assignment."""
    rows = []
    for name, lat, lon, koppen, is_proto in CITIES_DB:
        proto = KOPPEN_TO_PROTOTYPE.get(koppen, "Montreal")  # default cold
        city_key = name.replace(" ", "_")
        break_year = prototype_break_years.get(proto, prototype_break_years.get(city_key, "Never"))
        rows.append({
            "city": name,
            "latitude": lat,
            "longitude": lon,
            "koppen": koppen,
            "prototype": proto,
            "break_year": break_year,
            "is_prototype": is_proto,
        })
    return pd.DataFrame(rows)


def load_basemap() -> gpd.GeoDataFrame:
    """Load North America polygons from local Natural Earth shapefile."""
    shp_path = SCRIPT_DIR / "ne_data" / "ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shp_path)
    return world[world["CONTINENT"] == "North America"].copy()


def draw_basemap(ax: plt.Axes, na: gpd.GeoDataFrame) -> None:
    """Draw the North America outline."""
    na.plot(ax=ax, color="#f0f4f8", edgecolor="#b0bec5", linewidth=0.5)
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def is_broken(break_year: int | str, snapshot: int) -> bool:
    return break_year != "Never" and int(break_year) <= snapshot


def build_masked_grid(
    na: gpd.GeoDataFrame,
    xlim: tuple[float, float] = MAP_XLIM,
    ylim: tuple[float, float] = MAP_YLIM,
    shape: tuple[int, int] = GRID_SHAPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a lon/lat grid masked to North American land polygons."""
    xs = np.linspace(xlim[0], xlim[1], shape[1])
    ys = np.linspace(ylim[0], ylim[1], shape[0])
    grid_x, grid_y = np.meshgrid(xs, ys)
    land_geom = na.geometry.unary_union
    grid_points = [Point(xy) for xy in zip(grid_x.ravel(), grid_y.ravel())]
    mask = np.fromiter((land_geom.covers(pt) for pt in grid_points), dtype=bool)
    return grid_x, grid_y, mask.reshape(shape)


def interpolate_idw(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.2,
) -> np.ndarray:
    """Inverse-distance interpolation for a smooth regional surface."""
    dx = grid_x[..., None] - x[None, None, :]
    dy = grid_y[..., None] - y[None, None, :]
    dist_sq = dx * dx + dy * dy
    exact = dist_sq == 0
    safe_dist_sq = np.where(exact, 1.0, dist_sq)
    weights = 1.0 / np.power(safe_dist_sq, power / 2.0)
    surface = np.sum(weights * values[None, None, :], axis=2) / np.sum(weights, axis=2)
    if exact.any():
        row_idx, col_idx, value_idx = np.where(exact)
        surface[row_idx, col_idx] = values[value_idx]
    return surface


def plot_regional_surface(
    ax: plt.Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mask: np.ndarray,
    surface: np.ndarray,
    *,
    levels: list[float] | np.ndarray,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize | None = None,
    alpha: float = 0.78,
    extend: str = "neither",
) -> None:
    """Draw a filled contour clipped to land by masking ocean cells."""
    masked_surface = np.ma.array(surface, mask=~mask)
    ax.contourf(
        grid_x,
        grid_y,
        masked_surface,
        levels=levels,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        extend=extend,
        antialiased=True,
        zorder=1,
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_years = [2040, 2070, 2100]
    prototype_break_years = load_prototype_break_years()
    prototype_stress_scores = load_prototype_stress_scores(snapshot_years)
    cities = build_city_dataframe(prototype_break_years)
    na = load_basemap()
    grid_x, grid_y, land_mask = build_masked_grid(na)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    proto_order = ["Phoenix", "Miami", "Los_Angeles", "Montreal", "Toronto", "Vancouver"]
    proto_to_idx = {proto: idx for idx, proto in enumerate(proto_order)}
    city_x = cities["longitude"].to_numpy(dtype=float)
    city_y = cities["latitude"].to_numpy(dtype=float)
    proto_values = cities["prototype"].map(proto_to_idx).to_numpy(dtype=float)
    proto_surface = interpolate_idw(city_x, city_y, proto_values, grid_x, grid_y, power=3.0)
    proto_cmap = mcolors.ListedColormap([PROTOTYPE_COLORS[key] for key in proto_order])
    proto_norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(proto_order) + 0.5, 1.0),
        proto_cmap.N,
    )
    stress_levels = np.linspace(0.0, STRESS_COLORBAR_MAX, 11)
    stress_cmap = mcolors.LinearSegmentedColormap.from_list(
        "break_plus_relative_stress",
        [
            "#2c7bb6",
            "#74add1",
            "#d9f0d3",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ],
    )
    stress_norm = mcolors.Normalize(vmin=0.0, vmax=STRESS_COLORBAR_MAX, clip=True)

    # ── Panel A: Climate-zone prototype classification ────────────────────
    ax = axes[0]
    draw_basemap(ax, na)
    plot_regional_surface(
        ax,
        grid_x,
        grid_y,
        land_mask,
        proto_surface,
        levels=np.arange(-0.5, len(proto_order) + 0.5, 1.0),
        cmap=proto_cmap,
        norm=proto_norm,
        alpha=0.64,
    )
    na.boundary.plot(ax=ax, color="#8fa1aa", linewidth=0.5, zorder=2)
    ax.set_title("(a)  Core-City Koppen Prototype Assignment", fontsize=11, fontweight="bold",
                 loc="left", pad=8)

    non_proto_cities = cities[~cities["is_prototype"]]
    for proto_key in proto_order:
        proto_cities = non_proto_cities[non_proto_cities["prototype"] == proto_key]
        ax.scatter(
            proto_cities["longitude"],
            proto_cities["latitude"],
            s=42,
            c="white",
            edgecolors="none",
            alpha=0.92,
            zorder=7,
        )
        ax.scatter(
            proto_cities["longitude"],
            proto_cities["latitude"],
            s=30,
            c=PROTOTYPE_COLORS[proto_key],
            edgecolors="#263238",
            linewidths=0.35,
            alpha=0.95,
            zorder=8,
        )

    # Label prototype cities
    label_cfg = {
        "Los Angeles":  ( 3.0, -3.0, "left"),
        "Miami":        ( 2.5, -2.0, "left"),
        "Montreal":     ( 2.5,  1.8, "left"),
        "Phoenix":      (-15.0,  3.5, "left"),
        "Toronto":      ( 2.5, -2.5, "left"),
        "Vancouver":    ( 2.5,  1.8, "left"),
    }
    for _, row in cities[cities["is_prototype"]].iterrows():
        ax.scatter(
            row["longitude"], row["latitude"], s=120,
            c=PROTOTYPE_COLORS.get(row["prototype"], "#999999"),
            edgecolors="black", linewidths=1.3, zorder=10,
        )
        cfg = label_cfg.get(row["city"], (1.5, 1.0, "left"))
        ax.annotate(
            f"{row['city']}\n{row['koppen']}",
            xy=(row["longitude"], row["latitude"]),
            xytext=(row["longitude"] + cfg[0], row["latitude"] + cfg[1]),
            fontsize=7.5, fontweight="bold", ha=cfg[2], va="center",
            arrowprops=dict(arrowstyle="-", color="#555", lw=0.6),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    # ── Panels B, C, D: Snapshot break maps ──────────────────────────────
    panel_labels = ["(b)", "(c)", "(d)"]
    snapshot_label_cfg = {
        "Los Angeles":  ( 3.0, -3.0, "left"),
        "Miami":        ( 2.5, -2.0, "left"),
        "Montreal":     ( 2.5,  1.8, "left"),
        "Phoenix":      (-15.0,  3.5, "left"),
        "Toronto":      ( 2.5, -2.5, "left"),
        "Vancouver":    ( 2.5,  1.8, "left"),
    }
    for idx, (snapshot, label) in enumerate(zip(snapshot_years, panel_labels)):
        ax = axes[idx + 1]
        draw_basemap(ax, na)
        stress_values = cities["prototype"].map(prototype_stress_scores.get(snapshot, {})).fillna(0.0).to_numpy(dtype=float)
        stress_surface = interpolate_idw(city_x, city_y, stress_values, grid_x, grid_y, power=2.0)
        plot_regional_surface(
            ax,
            grid_x,
            grid_y,
            land_mask,
            stress_surface,
            levels=stress_levels,
            cmap=stress_cmap,
            norm=stress_norm,
            alpha=0.70,
            extend="neither",
        )
        na.boundary.plot(ax=ax, color="#8fa1aa", linewidth=0.5, zorder=2)
        ax.set_title(f"{label}  {snapshot} {PRIMARY_BREAK_YEAR_LABEL} Snapshot", fontsize=11,
                     fontweight="bold", loc="left", pad=8)

        broken_count = int(cities["break_year"].apply(lambda by: is_broken(by, snapshot)).sum())
        total_count = len(cities)

        for _, row in cities[~cities["is_prototype"]].iterrows():
            broken = is_broken(row["break_year"], snapshot)
            stress_score = prototype_stress_scores.get(snapshot, {}).get(row["prototype"], 0.0)
            marker = "o" if broken else "^"
            ax.scatter(
                row["longitude"],
                row["latitude"],
                s=52,
                marker=marker,
                facecolors="white",
                edgecolors="none",
                alpha=0.82,
                zorder=7,
            )
            ax.scatter(
                row["longitude"],
                row["latitude"],
                s=30 if broken else 27,
                marker=marker,
                facecolors=stress_cmap(stress_norm(stress_score)),
                edgecolors="#7f1d1d" if broken else "#1565c0",
                linewidths=0.75 if broken else 1.2,
                alpha=0.92 if broken else 0.98,
                zorder=9 if broken else 8,
            )

        for _, row in cities[cities["is_prototype"]].iterrows():
            broken = is_broken(row["break_year"], snapshot)
            stress_score = prototype_stress_scores.get(snapshot, {}).get(row["prototype"], 0.0)

            marker = "o" if broken else "^"
            ax.scatter(row["longitude"], row["latitude"], s=128,
                       marker=marker, facecolors=stress_cmap(stress_norm(stress_score)), edgecolors="black",
                       linewidths=1.8, zorder=10)
            by_text = str(row["break_year"])
            ax.annotate(
                f"{row['city']}\n{'BREAK ' + by_text if broken else 'OK'} | stress {stress_score:.2f}",
                xy=(row["longitude"], row["latitude"]),
                xytext=(row["longitude"] + snapshot_label_cfg.get(row["city"], (1.5, 1.0, "left"))[0],
                        row["latitude"] + snapshot_label_cfg.get(row["city"], (1.5, 1.0, "left"))[1]),
                fontsize=7, fontweight="bold", ha="left", va="center",
                color="#d32f2f" if broken else "#2e7d32",
                arrowprops=dict(arrowstyle="-", color="#555", lw=0.5),
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

        # Summary annotation
        pct = 100 * broken_count / total_count
        ax.text(0.02, 0.02, f"{broken_count}/{total_count} cities transitioned ({pct:.0f}%)",
                transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                color="#d32f2f" if pct > 50 else "#555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))

    # ── Legends ──────────────────────────────────────────────────────────
    proto_handles = []
    for proto_key in proto_order:
        by = prototype_break_years[proto_key]
        by_str = str(by) if by != "Never" else "Never"
        proto_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=PROTOTYPE_COLORS[proto_key],
                   markeredgecolor="black", markersize=8,
                   label=f"{PROTOTYPE_DISPLAY[proto_key]}  —  Break: {by_str}")
        )

    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f46d43",
               markeredgecolor="#7f1d1d", markersize=7, label="City transitioned by snapshot"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="#1565c0", markersize=7, label="City not yet transitioned"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#111", markeredgewidth=1.8, markersize=9,
               label="Labeled large marker = Koppen prototype"),
    ]

    fig.legend(handles=proto_handles, loc="lower left", ncol=2, frameon=True,
               fontsize=8, bbox_to_anchor=(0.02, -0.01), framealpha=0.95,
               edgecolor="#ccc", title="Koppen Prototype Proxy Colors", title_fontsize=9)
    fig.legend(handles=status_handles, loc="lower right", ncol=1, frameon=True,
               fontsize=8, bbox_to_anchor=(0.98, -0.01), framealpha=0.95,
               edgecolor="#ccc", title="Break Status", title_fontsize=9)

    fig.suptitle(
        "Building Energy System Transition Propagation with Relative Stress Intensity\n5-year rolling >=25% failure occurrence via Koppen Prototype Mapping (RCP 8.5, Office)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    colorbar_ax = fig.add_axes([0.36, 0.018, 0.28, 0.014])
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=stress_norm, cmap=stress_cmap),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[0.0, 0.15, 0.30, 0.45, STRESS_COLORBAR_MAX],
    )
    colorbar.ax.set_xticklabels([
        "low",
        "0.15",
        "0.30",
        "0.45",
        "highest\nobserved",
    ])
    colorbar.set_label(
        "Severe stress score: threshold exceedance severity blended with p90 climate-response flags",
        fontsize=8,
    )
    colorbar.ax.tick_params(labelsize=7, length=2)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PNG}")

    # Print summary table
    print("\n── Break-Year Summary by Prototype ──")
    for proto_key in proto_order:
        count = len(cities[cities["prototype"] == proto_key])
        by = prototype_break_years[proto_key]
        stress_summary = "  ".join(
            f"{snapshot}:{prototype_stress_scores.get(snapshot, {}).get(proto_key, 0.0):.2f}"
            for snapshot in snapshot_years
        )
        print(f"  {PROTOTYPE_DISPLAY[proto_key]:30s}  Break: {str(by):6s}  Stress: {stress_summary}  Cities: {count}")
    print(f"  {'Total':30s}  {' ':13s}  Cities: {len(cities)}")


if __name__ == "__main__":
    main()
