"""Step 3: exploratory data analysis. Saves figures to reports/figures/.

Run:  py src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print("saved", name)


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "dengue_features.csv", parse_dates=["start"])

    # 1. National weekly cases over time, with the train/val/test periods shaded.
    nat = df.groupby("start")["cases"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(nat.index, nat.values, lw=1, color="#c0392b")
    for split, c in [("val", "#f1c40f"), ("test", "#3498db")]:
        s = df.loc[df.split == split, "start"]
        plt.axvspan(s.min(), s.max(), alpha=0.15, color=c, label=split)
    plt.title("Weekly dengue cases, Sri Lanka (all districts)")
    plt.ylabel("Cases"); plt.legend()
    save("01_national_timeseries.png")

    # 2. Case distribution: heavy right skew, so we need a count model.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df.cases, bins=100, color="#7f8c8d"); ax[0].set_title("Weekly cases per district")
    ax[1].hist(np.log1p(df.cases), bins=50, color="#16a085"); ax[1].set_title("log(1 + cases)")
    fig.suptitle(f"Mean = {df.cases.mean():.1f}, variance = {df.cases.var():.0f}  (variance >> mean → over-dispersed)")
    save("02_case_distribution.png")

    # 3. Cases by district.
    order = df.groupby("district")["cases"].median().sort_values(ascending=False).index
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="district", y="cases", order=order, showfliers=False, color="#e67e22")
    plt.xticks(rotation=70); plt.title("Weekly cases by district (outliers hidden)")
    save("03_cases_by_district.png")

    # 4. Seasonality: average cases by week of year, with average rainfall.
    season = df[df.week <= 52].groupby("week").agg(cases=("cases", "mean"), rain=("rain_mm", "mean"))
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(season.index, season.cases, color="#c0392b", lw=2, label="Cases")
    ax1.set_xlabel("Week of year"); ax1.set_ylabel("Mean cases", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.bar(season.index, season.rain, alpha=0.3, color="#2980b9")
    ax2.set_ylabel("Mean weekly rain (mm)", color="#2980b9"); ax2.grid(False)
    plt.title("Seasonality: two yearly peaks (Jun-Jul after SW monsoon, Dec-Jan after NE monsoon)")
    save("04_seasonality.png")

    # 5. Lag correlation: rainfall k weeks ago vs this week's (log) cases, within districts.
    rows = []
    for k in range(0, 13):
        lagged = df.groupby("district")["rain_mm"].shift(k)
        sub = pd.DataFrame({"r": lagged, "y": np.log1p(df.cases), "d": df.district}).dropna()
        # Remove district-level differences so we see the timing effect only.
        sub["y"] -= sub.groupby("d")["y"].transform("mean")
        sub["r"] -= sub.groupby("d")["r"].transform("mean")
        rows.append((k, sub["r"].corr(sub["y"])))
    lag = pd.DataFrame(rows, columns=["lag", "corr"])
    plt.figure(figsize=(8, 4))
    plt.bar(lag.lag, lag["corr"], color="#2980b9")
    plt.xlabel("Rainfall lag (weeks)"); plt.ylabel("Correlation with log cases")
    plt.title("Rain affects dengue with a delay of several weeks")
    save("05_rain_lag_correlation.png")

    # 6. Outbreak rate by district.
    rate = df.groupby("district")["outbreak"].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 4))
    rate.plot.bar(color="#8e44ad"); plt.ylabel("Share of weeks flagged as outbreak")
    plt.title("Outbreak weeks (cases > past-52-week mean + 2 SD)")
    save("06_outbreak_rate_by_district.png")

    # 7. Example district with its outbreak threshold.
    c = df[df.district == "Colombo"]
    plt.figure(figsize=(12, 4))
    plt.plot(c.start, c.cases, lw=1, label="Cases")
    plt.plot(c.start, c.outbreak_threshold, lw=1, ls="--", color="k", label="Threshold")
    plt.scatter(c.start[c.outbreak == 1], c.cases[c.outbreak == 1], s=8, color="red", label="Outbreak week")
    plt.title("Colombo: weekly cases and endemic-channel threshold"); plt.legend()
    save("07_colombo_outbreak_label.png")

    # 8. Feature correlation heatmap.
    cols = ["cases", "log_cases_lag1", "log_cases_mean4_lag1", "rain_mm_lag1", "rain_mm_lag3",
            "rain_4wk_sum_lag1", "temp_c_lag2", "humidity_lag2", "week_sin", "week_cos"]
    plt.figure(figsize=(9, 7))
    from explain import label
    corr = df[cols].corr(method="spearman").rename(index=lambda c: label(c) if c != "cases" else "Cases (target)",
                                                     columns=lambda c: label(c) if c != "cases" else "Cases (target)")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Spearman correlation of key features")
    save("08_feature_correlation.png")

    print(lag.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
