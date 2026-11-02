"""Step 5: explainability (XAI).

1. Negative Binomial incidence rate ratios (IRR) - global, built-in interpretability
2. RuleFit rules table, translated into plain language
3. SHAP - NegBin (exact, log scale), RuleFit (model-agnostic), XGBoost (tree) for comparison
4. Partial dependence plots for rainfall lags (NegBin vs XGBoost)
5. LIME - local explanation of one real outbreak week

Run after train_evaluate.py:  py src/explain.py
"""
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from lime.lime_tabular import LimeTabularExplainer

from train_evaluate import load, pred_negbin

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RES = ROOT / "reports" / "results"
SEED = 42
rng = np.random.default_rng(SEED)

LABELS = {
    "log_cases_lag1": "Cases last week (log)", "log_cases_lag2": "Cases 2 wks ago (log)",
    "log_cases_mean4_lag1": "Avg cases past 4 wks (log)", "excess_lag1": "Last week vs threshold (log ratio)",
    "log_threshold": "Outbreak threshold (log)", "rain_4wk_sum_lag1": "Rain, past 4 wks (mm)",
    "rain_4wk_sum_lag6": "Rain, 6-9 wks ago (mm)", "rain_mm_lag8": "Rain 8 wks ago (mm)",
    "week_sin": "Season (sin)", "week_cos": "Season (cos)",
}


def label(c):
    if c in LABELS:
        return LABELS[c]
    m = re.match(r"(rain_mm|temp_c|humidity)_lag(\d)", c)
    if m:
        unit = {"rain_mm": "Rain (mm)", "temp_c": "Temp (°C)", "humidity": "Humidity (%)"}[m[1]]
        return f"{unit}, {m[2]} wk ago"
    return c.replace("district_", "District: ").replace("monsoon_", "Monsoon: ")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", name)


# ---------------------------------------------------------------- 1. NegBin IRR
def irr_plot(scaler, cols):
    c = pd.read_csv(RES / "negbin_coefficients.csv", index_col=0).drop("const")
    sd = pd.Series(scaler.scale_, index=cols)
    c["sd_of_feature"] = sd.reindex(c.index)
    # Dummies: per-SD is meaningless, so convert to the effect of being in that category
    # (vs the reference: district Ampara, monsoon NE). exp(coef / sd) = IRR per 1 unit.
    cat = c.index.str.startswith(("district_", "monsoon_"))
    c["IRR_per_unit"] = np.exp(c.coef / c.sd_of_feature)
    c.to_csv(RES / "negbin_irr_table.csv")
    for kind, mask, name, per_unit, xlab in [
            ("weather, season and case history", ~cat, "13_negbin_irr_features.png", False,
             "Incidence rate ratio per +1 SD of the feature"),
            ("district and monsoon", cat, "14_negbin_irr_districts.png", True,
             "Incidence rate ratio vs reference (Ampara district / NE monsoon)")]:
        d = c[mask].copy()
        f = (1 / d.sd_of_feature) if per_unit else 1
        est = np.exp(d.coef * f)
        lo, hi = d.ci_low ** f, d.ci_high ** f
        o = np.argsort(est.values)
        est, lo, hi, d = est.iloc[o], lo.iloc[o], hi.iloc[o], d.iloc[o]
        plt.figure(figsize=(8, 0.32 * len(d) + 1.2))
        y = np.arange(len(d))
        plt.errorbar(est, y, xerr=[est - lo, hi - est], fmt="none", ecolor="grey", lw=1)
        plt.scatter(est, y, c=np.where(d.p_value < 0.05, "#c0392b", "#95a5a6"), zorder=3)
        plt.axvline(1, color="k", ls="--", lw=1)
        plt.yticks(y, [label(i) for i in d.index]); plt.xscale("log")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        plt.xlabel(f"{xlab}\n(>1 = more cases; red = p < 0.05; given recent case history)")
        plt.title(f"Negative Binomial IRRs: {kind}")
        save(name)
    return c


# ---------------------------------------------------------------- 2. RuleFit rules
def readable_rule(rule):
    def cond(m):
        feat, op, v = m[1], m[2], float(m[3])
        if feat == "excess_lag1":
            return f"last week {op} {np.exp(v) * 100:.0f}% of outbreak threshold"
        if feat.startswith("log_cases_lag"):
            return f"cases {feat[-1]} wk ago {op} {np.expm1(v):.0f}"
        if feat == "log_cases_mean4_lag1":
            return f"avg cases past 4 wks {op} {np.expm1(v):.0f}"
        if feat == "log_threshold":
            return f"outbreak threshold {op} {np.expm1(v):.0f} cases"
        if feat.startswith(("district_", "monsoon_")):
            name = label(feat)
            return f"NOT {name}" if op == "<=" else name
        return f"{label(feat)} {op} {v:.1f}"
    return re.sub(r"(\w+) (<=|>) (-?[\d.e-]+)", cond, rule).replace(" and ", " AND ")


def rules_table():
    r = pd.read_csv(RES / "rulefit_rules.csv")
    r["readable"] = [readable_rule(x) if t == "rule" else f"[linear] {label(x)}"
                     for x, t in zip(r.rule, r.type)]
    r["effect"] = np.where(r.coef > 0, "raises risk", "lowers risk")
    r.to_csv(RES / "rulefit_rules_readable.csv", index=False)
    top = r.head(15).iloc[::-1]
    plt.figure(figsize=(11, 7))
    plt.barh(range(len(top)), top.importance, color=np.where(top.coef > 0, "#c0392b", "#2980b9"))
    plt.yticks(range(len(top)), [t if len(t) < 95 else t[:92] + "..." for t in top.readable], fontsize=8)
    plt.xlabel("Importance (|coef| x rule support)")
    plt.title("RuleFit: top 15 rules (red = raises outbreak risk, blue = lowers)")
    save("15_rulefit_top_rules.png")
    return r


# ---------------------------------------------------------------- 3. SHAP
def shap_negbin(nb, Xs_te, Xs_bg):
    """For a log-link GLM, SHAP on the log scale is exact: phi = beta * (x - E[x])."""
    beta = nb.params.drop("const")
    Xs_te, Xs_bg = Xs_te[beta.index], Xs_bg[beta.index]
    vals = (Xs_te - Xs_bg.mean()) * beta
    exp = shap.Explanation(values=vals.values, base_values=np.full(len(vals), nb.params["const"] + (Xs_bg.mean() * beta).sum()),
                           data=Xs_te.values, feature_names=[label(c) for c in Xs_te.columns])
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("SHAP - Negative Binomial (log expected cases)")
    save("16_shap_negbin_beeswarm.png")
    return exp


def shap_rulefit(rf, X_raw_te, X_raw_bg):
    sample = X_raw_te.sample(300, random_state=SEED)
    f = lambda a: rf.predict_proba(pd.DataFrame(a, columns=X_raw_te.columns))[:, 1]
    expl = shap.Explainer(f, shap.maskers.Independent(X_raw_bg.sample(100, random_state=SEED)),
                          feature_names=[label(c) for c in X_raw_te.columns], seed=SEED)
    sv = expl(sample, max_evals=2 * X_raw_te.shape[1] + 1)
    sv.feature_names = [label(c) for c in X_raw_te.columns]
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP - RuleFit (outbreak probability)")
    save("17_shap_rulefit_beeswarm.png")
    return sv


def shap_xgb(xgb, Xs_te):
    sv = shap.TreeExplainer(xgb)(Xs_te.sample(1000, random_state=SEED))
    sv.feature_names = [label(c) for c in Xs_te.columns]
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("SHAP - XGBoost outbreak classifier (baseline, for comparison)")
    save("18_shap_xgb_bar.png")
    return sv


def shap_ranking(*named):
    out = {}
    for name, sv in named:
        v = sv.values if hasattr(sv, "values") else sv
        out[name] = pd.Series(np.abs(v).mean(0), index=sv.feature_names).rank(ascending=False).astype(int)
    t = pd.DataFrame(out)
    t = t.loc[t.min(axis=1).sort_values().index].head(15)
    t.to_csv(RES / "shap_rank_comparison.csv")
    print("\nSHAP importance rank by model:\n", t.to_string())


# ---------------------------------------------------------------- 4. PDP
def pdp(df, X, count, feats):
    """Manual PDP: vary one raw feature over its range, keep others as observed, average prediction."""
    sc, cols = count["scaler"], count["columns"]
    nb, xgb = count["models"]["negbin"], count["models"]["xgboost"]
    base = X[df.split == "test"].sample(1500, random_state=SEED)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, f in zip(axes.ravel(), feats):
        grid = np.linspace(*np.percentile(X[f], [2, 98]), 25)
        nb_y, xg_y = [], []
        for v in grid:
            Z = base.copy(); Z[f] = v
            Zs = pd.DataFrame(sc.transform(Z), columns=cols)
            nb_y.append(pred_negbin(nb, Zs).mean())
            xg_y.append(xgb.predict(Zs).mean())
        ax.plot(grid, nb_y, color="#c0392b", lw=2, label="Negative Binomial")
        ax.plot(grid, xg_y, color="#2980b9", lw=2, ls="--", label="XGBoost")
        ax.set_xlabel(label(f)); ax.set_ylabel("Avg predicted weekly cases")
        ax2 = ax.twinx(); ax2.hist(X[f].clip(grid[0], grid[-1]), bins=40, alpha=0.12, color="grey")
        ax.set_xlim(grid[0], grid[-1]); ax2.set_yticks([])
    axes[0, 0].legend()
    # NegBin excludes the 4-week sum (it is the sum of the weekly lags), so its line is flat there.
    fig.suptitle("Partial dependence of predicted cases on rainfall lags (test period)")
    save("19_pdp_rain_lags.png")


# ---------------------------------------------------------------- 5. LIME
def lime_example(df, X, outbreak):
    rf = outbreak["models"]["rulefit"]
    thr = outbreak["thresholds"]["RuleFit"]
    tr = (df.split != "test").to_numpy()
    te = df[df.split == "test"]
    p = rf.predict_proba(X[df.split == "test"])[:, 1]
    # Pick a real outbreak week that the model correctly flagged, with high confidence.
    cand = te.assign(p=p)[(te.outbreak == 1) & (p >= thr)].sort_values("p", ascending=False)
    row = cand.iloc[len(cand) // 4]  # a typical (not the most extreme) true positive
    x = X.loc[row.name]
    # District/monsoon are held at this week's values, so LIME explains the
    # weather and case-history drivers within that district.
    num = [c for c in X.columns if not c.startswith(("district_", "monsoon_"))]

    def predict(a):
        Z = pd.DataFrame(np.repeat(x.values[None, :], len(a), axis=0), columns=X.columns)
        Z[num] = a
        return rf.predict_proba(Z)

    exp = LimeTabularExplainer(X.loc[tr, num].values, feature_names=[label(c) for c in num],
                               class_names=["Normal", "Outbreak"], discretize_continuous=True,
                               mode="classification", random_state=SEED)
    e = exp.explain_instance(x[num].values, predict, num_features=10, num_samples=3000)
    fig = e.as_pyplot_figure()
    fig.set_size_inches(10, 5)
    plt.title(f"LIME - {row.district}, week of {row.start:%d %b %Y}: "
              f"{int(row.cases)} cases (threshold {row.outbreak_threshold:.0f}), RuleFit score={row.p:.2f}")
    plt.xlabel("Contribution to RuleFit outbreak score")
    save("20_lime_outbreak_example.png")
    e.save_to_file(str(RES / "lime_outbreak_example.html"))
    pd.DataFrame(e.as_list(), columns=["condition", "weight"]).to_csv(RES / "lime_outbreak_example.csv", index=False)
    print(f"\nLIME example: {row.district} {row.start:%Y-%m-%d}, cases={int(row.cases)}, P={row.p:.2f}")
    for c, w in e.as_list():
        print(f"  {w:+.3f}  {c}")
    return row


def main():
    df, X = load()
    count = joblib.load(ROOT / "models" / "count_models.joblib")
    outbreak = joblib.load(ROOT / "models" / "outbreak_models.joblib")
    sc, cols = count["scaler"], count["columns"]
    te, bg = df.split == "test", df.split != "test"
    Xs = pd.DataFrame(sc.transform(X), columns=cols, index=X.index)
    Xs_o = pd.DataFrame(outbreak["scaler"].transform(X), columns=cols, index=X.index)

    irr = irr_plot(sc, cols)
    print("\nNegBin top IRRs (non-district):")
    print(irr[~irr.index.str.startswith("district_")].sort_values("IRR_per_SD", ascending=False)
          [["IRR_per_SD", "p_value"]].round(3).to_string())
    rules = rules_table()
    print("\nTop readable rules:\n", rules[["readable", "coef"]].head(8).to_string(index=False))

    sv_nb = shap_negbin(count["models"]["negbin"], Xs[te], Xs[bg])
    sv_rf = shap_rulefit(outbreak["models"]["rulefit"], X[te], X[bg])
    sv_xg = shap_xgb(outbreak["models"]["xgboost"], Xs_o[te])
    shap_ranking(("NegBin", sv_nb), ("RuleFit", sv_rf), ("XGBoost", sv_xg))

    pdp(df, X, count, ["rain_mm_lag1", "rain_mm_lag2", "rain_mm_lag4",
                       "rain_mm_lag8", "rain_4wk_sum_lag1", "rain_4wk_sum_lag6"])
    lime_example(df, X, outbreak)


if __name__ == "__main__":
    main()
