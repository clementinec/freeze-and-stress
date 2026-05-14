#!/usr/bin/env python3
"""Generate Fig. 4: driver-sentinel-action decision map."""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from figure_paper_common import (
    BLOCK_DECOMP_CSV,
    BLOCK_LABEL,
    CLIMATE_CITY,
    CLIMATE_LABEL,
    CLIMATE_ORDER,
    FIGURES_DIR,
    REGISTRY_CSV,
    SUMMARY_CSV,
    ensure_figures_dir,
    first_trigger_keys,
    wrap_text,
)


ACTION_MAP = {
    "Am": "Envelope heat control + cooling adequacy check",
    "BWh": "Peak comfort, shading, and operating controls",
    "Csb": "Short-duration overheating controls",
    "Cfb": "Heating adequacy and heat-pump timing",
    "Dfa": "Monitor; no severe-transition retrofit trigger",
    "Dfb": "Intermittent overheating monitoring in cold climate",
}


def build_data(summary: pd.DataFrame, registry: pd.DataFrame, block_decomp: pd.DataFrame) -> pd.DataFrame:
    label_to_metric = dict(zip(registry["label"], registry["metric"]))
    rows = []
    for climate in CLIMATE_ORDER:
        city = CLIMATE_CITY[climate]
        source = summary[summary["city"] == city].iloc[0]
        trigger_metric_names = str(source.get("first_trigger_metric", "Never"))
        trigger_keys = first_trigger_keys(source, label_to_metric)
        driver_counter: Counter[str] = Counter()
        driver_share: dict[str, float] = {}
        for metric in trigger_keys:
            row = block_decomp[(block_decomp["climate_type"] == climate) & (block_decomp["metric"] == metric)]
            if row.empty:
                continue
            dominant = str(row.iloc[0]["block_dominant"])
            share = pd.to_numeric(row.iloc[0]["block_dominant_share"], errors="coerce")
            if dominant and dominant != "nan":
                driver_counter[dominant] += 1
                driver_share[dominant] = max(driver_share.get(dominant, 0.0), float(share) if pd.notna(share) else 0.0)
        dominant_drivers = [
            driver
            for driver, _ in sorted(
                driver_counter.items(),
                key=lambda item: (-item[1], -driver_share.get(item[0], 0.0), item[0]),
            )
        ]
        if not dominant_drivers and str(source.get("main_break_year")) == "Never":
            dominant_driver_label = "No severe-transition driver"
        else:
            dominant_driver_label = " + ".join(BLOCK_LABEL[d].replace("\n", " ") for d in dominant_drivers[:2])
        rows.append(
            {
                "climate_type": climate,
                "city": city,
                "dominant_driver_chain": dominant_driver_label,
                "sentinel_metric": trigger_metric_names,
                "main_break_year": source.get("main_break_year"),
                "main_sustained_failure_year": source.get("main_sustained_failure_year"),
                "adaptation_priority": ACTION_MAP[climate],
            }
        )
    return pd.DataFrame(rows)


def plot(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    ax.set_axis_off()
    ax.set_title("Fig. 4 | Climate-indexed adaptation priorities from driver-sentinel chains", loc="left", fontsize=13, fontweight="bold")

    columns = [
        ("Climate", 0.10),
        ("Dominant driver", 0.22),
        ("Sentinel metric(s)", 0.28),
        ("Timing", 0.13),
        ("Adaptation priority", 0.27),
    ]
    x0 = 0.02
    y_top = 0.88
    header_h = 0.08
    row_h = 0.122
    total_w = 0.96
    x_positions = [x0]
    for _, width in columns[:-1]:
        x_positions.append(x_positions[-1] + width * total_w)

    ax.add_patch(patches.Rectangle((x0, y_top), total_w, header_h, facecolor="#263238", edgecolor="none"))
    for (label, _), x in zip(columns, x_positions):
        ax.text(x + 0.008, y_top + header_h / 2, label, va="center", ha="left", fontsize=9, fontweight="bold", color="white")

    for row_index, climate in enumerate(CLIMATE_ORDER):
        row = data[data["climate_type"] == climate].iloc[0]
        y = y_top - (row_index + 1) * row_h
        face = "#f8fafb" if row_index % 2 == 0 else "#ffffff"
        ax.add_patch(patches.Rectangle((x0, y), total_w, row_h, facecolor=face, edgecolor="#d9e0e3", linewidth=0.8))
        ax.add_patch(patches.Rectangle((x0, y), 0.008, row_h, facecolor="#111111" if row["main_break_year"] != "Never" else "#aeb7bc", edgecolor="none"))

        timing = f"{row['main_break_year']} / {row['main_sustained_failure_year']}"
        values = [
            CLIMATE_LABEL[climate],
            wrap_text(row["dominant_driver_chain"], 24),
            wrap_text(row["sentinel_metric"], 31),
            timing,
            wrap_text(row["adaptation_priority"], 32),
        ]
        for (_, _), x, value in zip(columns, x_positions, values):
            ax.text(x + 0.010, y + row_h / 2, value, va="center", ha="left", fontsize=8.4, color="#172026")

    ax.text(
        x0,
        0.05,
        "Timing is main break year / sustained failure year. Never means the severe-transition screen is not sustained by 2100.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#455a64",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_driver_sentinel_action_map.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_figures_dir()
    summary = pd.read_csv(SUMMARY_CSV)
    registry = pd.read_csv(REGISTRY_CSV)
    block_decomp = pd.read_csv(BLOCK_DECOMP_CSV)
    data = build_data(summary, registry, block_decomp)
    data.to_csv(FIGURES_DIR / "fig4_driver_sentinel_action_map_data.csv", index=False)
    plot(data)
    print(f"Saved -> {FIGURES_DIR / 'fig4_driver_sentinel_action_map.png'}")


if __name__ == "__main__":
    main()
