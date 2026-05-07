from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ===== paths =====
hdd_cdd_path = Path(
    r"E:\00_research\03_freezethenstress\freeze-and-stress-main\epw_pipeline\epw_out\CORDEX_CMIP5_REMO2015_rcp85\Los_Angeles\hdd_cdd_timeseries.csv"
)
metrics_path = Path(
    r"E:\00_research\03_freezethenstress\analysis_outputs\annual_metrics_filtered.csv"
)
output_dir = Path(r"E:\00_research\03_freezethenstress\analysis_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== read =====
df_cdd = pd.read_csv(hdd_cdd_path)
df_metrics = pd.read_csv(metrics_path)

# ===== identify columns =====
def find_col(df, keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

cdd_col = find_col(df_cdd, "cdd")
ulh_col = find_col(df_metrics, "ulh_percent")
stress_col = find_col(df_metrics, "stress_per_1000kwh")

if cdd_col is None or ulh_col is None or stress_col is None:
    raise ValueError(f"未找到列：CDD={cdd_col}, ULH={ulh_col}, Stress={stress_col}")

# ===== preprocess =====
df_cdd["year"] = pd.to_numeric(df_cdd["year"], errors="coerce")
df_metrics["year"] = pd.to_numeric(df_metrics["year"], errors="coerce")

# merge by year
df = pd.merge(
    df_cdd[["year", cdd_col]],
    df_metrics[["year", ulh_col, stress_col]],
    on="year",
    how="inner"
).sort_values("year")

# ===== break year =====
threshold = 10.0
break_year = None
for _, row in df.iterrows():
    if row[ulh_col] > threshold:
        break_year = int(row["year"])
        break

# ===== plot =====
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左轴：CDD
ax1.plot(df["year"], df[cdd_col], marker="o", color="tab:blue", label="CDD")
ax1.set_xlabel("Year")
ax1.set_ylabel("CDD (°C·day)", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")

# 右轴：ULH & Stress
ax2 = ax1.twinx()
ax2.plot(df["year"], df[ulh_col], marker="s", color="tab:red", label="ULH (%)")
ax2.plot(df["year"], df[stress_col], marker="^", color="tab:green", label="Stress (unmet/1000 kWh)")
ax2.set_ylabel("ULH (%) / Stress", color="black")

# 阈值线
ax2.axhline(threshold, linestyle="--", color="tab:red", alpha=0.5, label="ULH Threshold")

# break-year
if break_year:
    ax1.axvline(break_year, linestyle="--", color="tab:gray", alpha=0.7)
    ax1.text(break_year, df[cdd_col].max()*0.9, f"Break: {break_year}", rotation=90, color="tab:gray")

# 图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper left")

plt.title("Climate forcing (CDD) vs Comfort failure (ULH) and System Stress")
plt.tight_layout()

# ===== save =====
out_path = output_dir / "cdd_ulh_stress_combined.png"
plt.savefig(out_path, dpi=200)
plt.close()

print("[OK] Figure saved:", out_path)
print("Break year:", break_year)