import pandas as pd
from pathlib import Path

EPW_DIR = Path(r"E:\00_research\03_freezethenstress\freeze-and-stress-main\epw_pipeline\epw_out\CORDEX_CMIP5_REMO2015_rcp85\Los_Angeles")
BASE_TEMP = 18.0  # °C


def read_epw(epw_path: Path) -> pd.DataFrame:
    # 只读前 7 列，避免整表列数不一致问题
    df = pd.read_csv(epw_path, skiprows=8, header=None, usecols=[0, 1, 2, 3, 6])

    df.columns = ["year", "month", "day", "hour", "drybulb"]
    df["drybulb"] = pd.to_numeric(df["drybulb"], errors="coerce")
    df = df.dropna(subset=["drybulb"])

    return df


def compute_hdd_cdd(df: pd.DataFrame, base_temp: float = 18.0) -> tuple[float, float]:
    # EPW 是小时数据，按小时 degree-hours 累加后 /24 转成 degree-days
    temp = df["drybulb"]

    cdd = (temp - base_temp).clip(lower=0).sum() / 24.0
    hdd = (base_temp - temp).clip(lower=0).sum() / 24.0

    return hdd, cdd


results = []

for epw_file in sorted(EPW_DIR.glob("*.epw")):
    # 从文件名提取年份；如果文件名最后不是年份，就改这里
    try:
        year = int(epw_file.stem.split("_")[-1])
    except ValueError:
        # 退一步：直接读文件里的 year 列取第一个值
        year = None

    df = read_epw(epw_file)

    if year is None:
        year = int(df["year"].iloc[0])

    hdd, cdd = compute_hdd_cdd(df, BASE_TEMP)

    results.append({
        "year": year,
        "HDD_18C": hdd,
        "CDD_18C": cdd,
        "epw_file": epw_file.name,
    })

out_df = pd.DataFrame(results).sort_values("year")

out_path = EPW_DIR / "hdd_cdd_timeseries.csv"
out_df.to_csv(out_path, index=False)

print("Saved:", out_path)
print(out_df.head())