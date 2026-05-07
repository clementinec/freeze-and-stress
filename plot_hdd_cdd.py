from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ===== input =====
csv_path = Path(
    r"E:\00_research\03_freezethenstress\freeze-and-stress-main\epw_pipeline\epw_out\CORDEX_CMIP5_REMO2015_rcp85\Los_Angeles\hdd_cdd_timeseries.csv"
)

# ===== output =====
output_dir = Path(r"E:\00_research\03_freezethenstress\analysis_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== read =====
df = pd.read_csv(csv_path)

# 兼容列名
if "year" not in df.columns:
    raise ValueError("CSV 中缺少 year 列")

hdd_col = None
cdd_col = None

for col in df.columns:
    c = col.lower()
    if c.startswith("hdd"):
        hdd_col = col
    if c.startswith("cdd"):
        cdd_col = col

if hdd_col is None or cdd_col is None:
    raise ValueError(f"未找到 HDD/CDD 列，当前列名为: {df.columns.tolist()}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df[hdd_col] = pd.to_numeric(df[hdd_col], errors="coerce")
df[cdd_col] = pd.to_numeric(df[cdd_col], errors="coerce")
df = df.dropna(subset=["year", hdd_col, cdd_col]).sort_values("year")

# ===== save cleaned csv =====
clean_csv = output_dir / "hdd_cdd_timeseries_clean.csv"
df.to_csv(clean_csv, index=False)

# ===== plot 1: HDD and CDD together =====
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df[hdd_col], marker="o", label=hdd_col)
plt.plot(df["year"], df[cdd_col], marker="o", label=cdd_col)
plt.xlabel("Year")
plt.ylabel("Degree days")
plt.title("HDD/CDD Timeseries")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "hdd_cdd_timeseries.png", dpi=200)
plt.close()

# ===== plot 2: CDD only =====
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df[cdd_col], marker="o")
plt.xlabel("Year")
plt.ylabel(cdd_col)
plt.title("CDD Timeseries")
plt.tight_layout()
plt.savefig(output_dir / "cdd_timeseries.png", dpi=200)
plt.close()

# ===== plot 3: HDD only =====
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df[hdd_col], marker="o")
plt.xlabel("Year")
plt.ylabel(hdd_col)
plt.title("HDD Timeseries")
plt.tight_layout()
plt.savefig(output_dir / "hdd_timeseries.png", dpi=200)
plt.close()

print("[OK] Clean CSV:", clean_csv)
print("[OK] Figure:", output_dir / "hdd_cdd_timeseries.png")
print("[OK] Figure:", output_dir / "cdd_timeseries.png")
print("[OK] Figure:", output_dir / "hdd_timeseries.png")
print(df.head())