"""Step 2: clean data, aggregate weather to weeks, engineer features, label outbreaks.

Output: data/processed/dengue_features.csv (one row per district-week)

Run:  py src/preprocess.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

WEATHER_COLS = {"precipitation_sum": "rain_mm",
                "temperature_2m_mean": "temp_c",
                "relative_humidity_2m_mean": "humidity"}
LAGS = [1, 2, 3, 4]
# Chronological split: no shuffling, so the model is always tested on the future.
SPLITS = {"train": (2007, 2019), "val": (2020, 2021), "test": (2022, 2025)}


def load_dengue() -> pd.DataFrame:
    d = pd.read_csv(RAW / "dengue_weekly_district.csv")
    d["start"] = pd.to_datetime(d["start.date"], format="mixed")
    d["end"] = pd.to_datetime(d["end.date"], format="mixed")
    d = d.drop(columns=["start.date", "end.date"])
    d = d.dropna(subset=["cases"]).drop_duplicates(["district", "start"])
    d[["year", "week"]] = d[["year", "week"]].astype(int)
    d["cases"] = d["cases"].astype(int)
    return d.sort_values(["district", "start"]).reset_index(drop=True)


def weekly_weather(dengue: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather onto each report week (start..end, inclusive)."""
    w = pd.read_csv(RAW / "weather_daily_district.csv").rename(columns=WEATHER_COLS)
    w["time"] = pd.to_datetime(w["time"])
    w = w.sort_values("time")
    weeks = dengue[["district", "start", "end"]].sort_values("start")
    # Attach each day to the latest report week that started on or before it.
    m = pd.merge_asof(w, weeks, left_on="time", right_on="start", by="district")
    m = m[m["time"] <= m["end"]]
    return (m.groupby(["district", "start"])
             .agg(rain_mm=("rain_mm", "sum"), temp_c=("temp_c", "mean"),
                  humidity=("humidity", "mean"), rain_days=("rain_mm", lambda s: (s > 1).sum()))
             .reset_index())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("district", group_keys=False)
    # Weather lags: mosquito breeding + incubation gives a 2-4 week delay.
    for col in ["rain_mm", "temp_c", "humidity"]:
        for k in LAGS:
            df[f"{col}_lag{k}"] = g[col].shift(k)
    df["rain_4wk_sum_lag1"] = g["rain_mm"].transform(lambda s: s.shift(1).rolling(4).sum())
    # EDA shows rain correlates most with cases 6-10 weeks later.
    df["rain_4wk_sum_lag6"] = g["rain_mm"].transform(lambda s: s.shift(6).rolling(4).sum())
    df["rain_mm_lag8"] = g["rain_mm"].shift(8)
    # Previous weeks' cases (log scale, since counts are heavily skewed).
    for k in [1, 2]:
        df[f"log_cases_lag{k}"] = np.log1p(g["cases"].shift(k))
    df["log_cases_mean4_lag1"] = np.log1p(g["cases"].transform(lambda s: s.shift(1).rolling(4).mean()))
    # Cyclical encoding of season (week 52 sits next to week 1).
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df["monsoon"] = df["start"].dt.month.map(
        lambda m: "SW" if 5 <= m <= 9 else "NE" if m in (12, 1, 2) else "inter")
    return df


def add_outbreak_label(df: pd.DataFrame) -> pd.DataFrame:
    """Outbreak week = cases above the district's own 'endemic channel':
    mean + 2*SD of the previous 52 weeks. Uses only past data, so no leakage."""
    g = df.groupby("district")["cases"]
    mu = g.transform(lambda s: s.shift(1).rolling(52, min_periods=26).mean())
    sd = g.transform(lambda s: s.shift(1).rolling(52, min_periods=26).std())
    df["outbreak_threshold"] = mu + 2 * sd
    df["outbreak"] = (df["cases"] > df["outbreak_threshold"]).astype(int)
    # How far last week was above/below this week's threshold (both known in advance).
    df["log_threshold"] = np.log1p(df["outbreak_threshold"].clip(lower=0))
    df["excess_lag1"] = df["log_cases_lag1"] - df["log_threshold"]
    return df


def assign_split(year: int) -> str:
    for name, (a, b) in SPLITS.items():
        if a <= year <= b:
            return name
    return "drop"


def main():
    dengue = load_dengue()
    weather = weekly_weather(dengue)
    df = dengue.merge(weather, on=["district", "start"], how="left")
    df = df.sort_values(["district", "start"]).reset_index(drop=True)
    df = add_features(df)
    df = add_outbreak_label(df)
    df["split"] = df["year"].map(assign_split)

    before = len(df)
    df = df[df["split"] != "drop"].dropna().reset_index(drop=True)
    print(f"Rows: {before:,} -> {len(df):,} after dropping lag warm-up / missing")
    print(df.groupby("split").agg(rows=("cases", "size"), outbreak_rate=("outbreak", "mean"),
                                  mean_cases=("cases", "mean")).round(3))
    print(f"Overdispersion check: mean={df.cases.mean():.1f}, variance={df.cases.var():.1f}")
    df.to_csv(OUT / "dengue_features.csv", index=False)
    print("Saved", OUT / "dengue_features.csv", df.shape)


if __name__ == "__main__":
    main()
