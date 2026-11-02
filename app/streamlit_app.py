"""Dengue early-warning dashboard (Sri Lanka).

Pick a district and week, adjust recent cases and weather, and see:
  - expected cases with a 90% interval (Negative Binomial)
  - an outbreak risk level (RuleFit) and the rules that fired
  - a SHAP breakdown of the count prediction

Run from the project root:  py -m streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explain import label, readable_rule  # noqa: E402
from train_evaluate import load, pred_negbin  # noqa: E402

st.set_page_config(page_title="Dengue early warning – Sri Lanka", page_icon=":material/coronavirus:", layout="wide")

RISK_LEVELS = [(0.05, "Low", "green"), (0.25, "Moderate", "orange"), (1.01, "High", "red")]


@st.cache_resource
def load_models():
    return (joblib.load(ROOT / "models" / "count_models.joblib"),
            joblib.load(ROOT / "models" / "outbreak_models.joblib"))


@st.cache_data
def load_data():
    df, X = load()
    rules = pd.read_csv(ROOT / "reports" / "results" / "rulefit_rules.csv")
    return df, X, rules


@st.cache_data
def risk_calibration(_rf, _X, _y):
    """Map RuleFit scores to observed outbreak rates on 2007-2021 (score deciles).
    RuleFit scores are not calibrated probabilities, so we report how often
    weeks with a similar score actually were outbreaks."""
    s = pd.Series(_rf.predict_proba(_X)[:, 1], index=_X.index)
    bins = pd.qcut(s, 10, duplicates="drop", retbins=True)[1]
    bins[0], bins[-1] = -np.inf, np.inf
    rate = _y.groupby(pd.cut(s, bins), observed=False).mean().to_numpy()
    return bins, rate


count, outbreak = load_models()
df, X, rules = load_data()
nb = count["models"]["negbin"]
alpha = count["best"]["NegBin"]["alpha"]
rf = outbreak["models"]["rulefit"]
sc, cols = count["scaler"], count["columns"]
hist_mask = df.split != "test"
bins, rates = risk_calibration(rf, X[hist_mask], df.loc[hist_mask, "outbreak"])

# Source data spells some names without spaces or in older forms.
DISPLAY_NAMES = {"NuwaraEliya": "Nuwara Eliya", "Hambanthota": "Hambantota", "Kalmune": "Kalmunai"}

# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header(":material/tune: Scenario")
    districts = sorted(df.district.unique())
    district = st.selectbox("District", districts, index=districts.index("Colombo"),
                            format_func=lambda n: DISPLAY_NAMES.get(n, n))
    d = df[df.district == district]
    weeks = d[d.split == "test"].start.dt.date.tolist()
    week = st.select_slider("Week starting", options=weeks, value=weeks[len(weeks) // 2],
                            format_func=lambda w: w.strftime("%d %b %Y"))
    row = d[d.start.dt.date == week].iloc[0]
    x0 = X.loc[row.name]

    with st.form("scenario"):
        st.caption("Pre-filled with the recorded values for this week. Change any value to test a what-if scenario.")
        st.markdown("**:material/groups: Recent cases**")
        c1 = st.number_input("Cases last week", 0, 5000, int(round(np.expm1(row.log_cases_lag1))))
        c2 = st.number_input("Cases 2 weeks ago", 0, 5000, int(round(np.expm1(row.log_cases_lag2))))
        c4 = st.number_input("Average cases, past 4 weeks", 0.0, 5000.0,
                             float(round(np.expm1(row.log_cases_mean4_lag1), 1)))
        st.markdown("**:material/rainy: Rainfall (mm per week)**")
        rain = [st.slider(f"{k} week{'s' if k > 1 else ''} ago", 0.0, 400.0, float(round(row[f"rain_mm_lag{k}"], 1)))
                for k in (1, 2, 3, 4)]
        rain8 = st.slider("8 weeks ago", 0.0, 400.0, float(round(row.rain_mm_lag8, 1)))
        rain6 = st.slider("Total over 6–9 weeks ago", 0.0, 1200.0, float(round(row.rain_4wk_sum_lag6, 1)))
        st.markdown("**:material/thermostat: Temperature and humidity (2 weeks ago)**")
        temp2 = st.slider("Mean temperature (°C)", 10.0, 35.0, float(round(row.temp_c_lag2, 1)))
        hum2 = st.slider("Mean relative humidity (%)", 40.0, 100.0, float(round(row.humidity_lag2, 1)))
        st.form_submit_button("Update prediction", icon=":material/refresh:", type="primary", width="stretch")

# Build the feature row, keeping derived features consistent with the inputs.
x = x0.copy()
x["log_cases_lag1"], x["log_cases_lag2"] = np.log1p(c1), np.log1p(c2)
x["log_cases_mean4_lag1"] = np.log1p(c4)
for k, r in zip((1, 2, 3, 4), rain):
    x[f"rain_mm_lag{k}"] = r
x["rain_4wk_sum_lag1"] = sum(rain)
x["rain_4wk_sum_lag6"], x["rain_mm_lag8"] = rain6, rain8
x["temp_c_lag2"], x["humidity_lag2"] = temp2, hum2
x["excess_lag1"] = x["log_cases_lag1"] - x["log_threshold"]
Xrow = pd.DataFrame([x])
Xs = pd.DataFrame(sc.transform(Xrow), columns=cols)

mu = float(pred_negbin(nb, Xs).iloc[0])
lo, hi = nbinom.ppf([0.05, 0.95], 1 / alpha, 1 / (1 + alpha * mu))
score = float(rf.predict_proba(Xrow)[0, 1])
hist_rate = float(rates[np.searchsorted(bins, score, side="right") - 1])
level, colour = next((n, c) for cut, n, c in RISK_LEVELS if hist_rate < cut)
threshold_cases = float(row.outbreak_threshold)
# Inputs are pre-filled with rounded values, so compare at the same rounding.
defaults = [int(round(np.expm1(row.log_cases_lag1))), int(round(np.expm1(row.log_cases_lag2))),
            round(np.expm1(row.log_cases_mean4_lag1), 1), *[round(row[f"rain_mm_lag{k}"], 1) for k in (1, 2, 3, 4)],
            round(row.rain_mm_lag8, 1), round(row.rain_4wk_sum_lag6, 1), round(row.temp_c_lag2, 1),
            round(row.humidity_lag2, 1)]
edited = not np.allclose([c1, c2, c4, *rain, rain8, rain6, temp2, hum2], defaults)

# ---------------------------------------------------------------- header
st.title(":material/coronavirus: Dengue early-warning dashboard")
st.caption(f"Sri Lanka · weekly dengue cases by district · **{DISPLAY_NAMES.get(district, district)}**, week of {row.start:%d %B %Y}"
           + (" · :orange-badge[what-if scenario]" if edited else " · :gray-badge[recorded data]"))

with st.expander("How to use this dashboard", icon=":material/help:"):
    st.markdown(
        "1. Choose a **district** and a **week** (2022–2025, data the models never saw during training).\n"
        "2. The inputs on the left are filled with what was recorded before that week. Change them and press "
        "**Update prediction** to explore a what-if scenario.\n"
        "3. Read the forecast (expected cases and a 90% range), the **outbreak risk level**, and the two "
        "explanation panels showing *why* the models made their predictions.\n\n"
        "An **outbreak week** is a week whose cases exceed the district's normal level "
        "(mean + 2 standard deviations over the previous 52 weeks).")

# ---------------------------------------------------------------- headline
with st.container(horizontal=True):
    st.metric("Expected cases this week", f"{mu:.0f}", border=True,
              help="Mean of the Negative Binomial forecast")
    st.metric("90% prediction range", f"{lo:.0f} – {hi:.0f}", border=True,
              help="The model expects the true count to fall in this range 9 times out of 10 "
                   "(observed coverage on 2022–25: 93.5%)")
    st.metric("Outbreak threshold", f"{threshold_cases:.0f}", border=True,
              help="Cases above this level count as an outbreak week")
    st.metric("Recorded cases", f"{int(row.cases)}", border=True,
              delta=f"{int(row.cases) - mu:+.0f} vs expected", delta_color="off",
              help="What was actually reported that week (for comparison only)")

with st.container(border=True):
    icon = {"Low": ":material/check_circle:", "Moderate": ":material/error:", "High": ":material/warning:"}[level]
    st.markdown(f"### {icon} Outbreak risk: :{colour}[{level}]")
    st.progress(min(hist_rate / 0.6, 1.0))
    st.markdown(f"In 2007–2021, **{hist_rate:.0%}** of weeks with a similar RuleFit score turned out to be "
                f"outbreak weeks (overall rate ≈ 10%). Expected cases are "
                f"**{mu / threshold_cases:.0%}** of the outbreak threshold.")
    st.caption(f"Recorded outcome for this week: **{'outbreak' if row.outbreak else 'normal'}**. "
               "Levels: Low < 5%, Moderate 5–25%, High > 25% historical outbreak rate.")

# ---------------------------------------------------------------- history chart
with st.container(border=True):
    st.subheader("Recent weeks and this forecast")
    h = d[(d.start <= row.start) & (d.start > row.start - pd.Timedelta(weeks=40))]
    long = pd.DataFrame({"week": h.start, "Recorded cases": h.cases,
                         "Outbreak threshold": h.outbreak_threshold}).melt("week", var_name="series", value_name="cases")
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("week:T", title=None), y=alt.Y("cases:Q", title="Weekly cases"),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Recorded cases", "Outbreak threshold"], range=["#3b82c4", "#e6a23c"])),
        strokeDash=alt.condition(alt.datum.series == "Outbreak threshold", alt.value([5, 4]), alt.value([1, 0])))
    pt = pd.DataFrame({"week": [row.start], "mu": [mu], "lo": [lo], "hi": [hi]})
    band = alt.Chart(pt).mark_rule(strokeWidth=4, color="#c0392b", opacity=0.5).encode(x="week:T", y="lo:Q", y2="hi:Q")
    dot = alt.Chart(pt).mark_point(filled=True, size=110, color="#c0392b").encode(
        x="week:T", y="mu:Q", tooltip=[alt.Tooltip("mu:Q", title="Expected", format=".0f"),
                                       alt.Tooltip("lo:Q", title="5th percentile"),
                                       alt.Tooltip("hi:Q", title="95th percentile")])
    st.altair_chart(lines + band + dot, alt="Weekly cases with threshold and the forecast range for the selected week")
    st.caption("Red point and bar: Negative Binomial forecast and its 90% range for the selected week.")

# ---------------------------------------------------------------- explanations
with st.container(border=True):
    st.subheader("Why this many cases? (SHAP)")
    beta = nb.params.drop("const")
    # Baseline = this district's average week in 2007-2021, so effects read as
    # "compared with a typical week in this district".
    base = pd.DataFrame(sc.transform(X[hist_mask & (df.district == district)]), columns=cols)[beta.index].mean()
    phi = (Xs[beta.index].iloc[0] - base) * beta
    phi = phi[~phi.index.str.startswith("district_")]
    top = phi.reindex(phi.abs().sort_values(ascending=False).index).head(10)
    s = pd.DataFrame({"feature": [label(i) for i in top.index], "effect": np.expm1(top.values) * 100})
    chart = alt.Chart(s).mark_bar().encode(
        x=alt.X("effect:Q", title="Change in expected cases vs a typical week in this district (%)"),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.condition(alt.datum.effect > 0, alt.value("#c0392b"), alt.value("#2980b9")),
        tooltip=["feature", alt.Tooltip("effect:Q", title="Effect (%)", format="+.1f")])
    st.altair_chart(chart, alt="Top 10 feature contributions to the expected case count")
    st.caption("Red raises expected cases, blue lowers them. Exact SHAP values of the Negative Binomial model "
               "(log scale), shown as a percentage change.")

with st.container(border=True):
    st.subheader("Which outbreak rules fired? (RuleFit)")
    active = []
    for r in rules[(rules.coef != 0) & (rules.type == "rule")].itertuples():
        try:
            if bool(Xrow.eval(r.rule).iloc[0]):
                active.append((readable_rule(r.rule), r.coef))
        except Exception:
            continue
    if active:
        t = pd.DataFrame(active, columns=["Rule", "Weight"]).sort_values("Weight", key=abs, ascending=False)
        t["Effect"] = np.where(t.Weight > 0, ":red[↑ raises risk]", ":blue[↓ lowers risk]")
        t["Weight"] = t.Weight.map("{:+.3f}".format)
        # st.table wraps long rule text instead of truncating it.
        st.table(t.head(10).set_index("Rule")[["Effect", "Weight"]], alt="Active RuleFit rules for this week")
        st.caption(f"{len(t)} of {int(((rules.coef != 0) & (rules.type == 'rule')).sum())} learned rules apply "
                   "to this week; the 10 with the largest weights are shown.")
    else:
        st.info("No rules apply to this scenario; the risk comes from the model's linear terms only.")

# ---------------------------------------------------------------- model info
with st.container(border=True):
    st.subheader("How the models work overall")
    st.caption("Saved results from the whole dataset. These charts do not change with the district or week selected.")
    fig_dir = ROOT / "reports" / "figures"
    tab_irr, tab_pdp, tab_lime = st.tabs(["Rate ratios", "Rainfall effect (PDP)", "One week explained (LIME)"])
    with tab_irr:
        st.image(fig_dir / "13_negbin_irr_features.png")
        st.caption("Incidence rate ratios from Negative Binomial: how much expected cases multiply when a feature "
                   "rises by one standard deviation. 1 means no effect.")
    with tab_pdp:
        st.image(fig_dir / "19_pdp_rain_lags.png")
        st.caption("Partial dependence: how the average forecast changes as each rainfall feature is varied.")
    with tab_lime:
        st.image(fig_dir / "20_lime_outbreak_example.png")
        st.caption("LIME for one fixed example: Hambantota, week of 2 July 2022, a real outbreak RuleFit flagged "
                   "correctly. Bars show what pushed its score up or down.")

with st.expander("Model performance, data and limitations", icon=":material/info:"):
    res = ROOT / "reports" / "results"
    st.markdown("**Case counts — test set 2022–2025**")
    count_tbl = pd.read_csv(res / "test_count_metrics.csv").rename(columns={
        "model": "Model", "R2": "R²", "PoissonDev": "Poisson deviance", "PI90_coverage": "90% range coverage"})
    count_tbl["90% range coverage"] = count_tbl["90% range coverage"].map(lambda v: "–" if pd.isna(v) else f"{v:.1%}")
    st.dataframe(count_tbl, hide_index=True, alt="Count model test metrics")
    st.markdown("**Outbreak classification — test set 2022–2025**")
    st.dataframe(pd.read_csv(res / "test_outbreak_metrics.csv").rename(columns={
        "model": "Model", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "threshold": "Alert cut-off"}),
        hide_index=True, alt="Outbreak model test metrics")
    st.markdown(
        "- **Data:** Epidemiology Unit (Ministry of Health) weekly reports, compiled in the `denguedatahub` "
        "R package; weather from Open-Meteo (ERA5) at each district's centre.\n"
        "- **Privacy:** aggregated district counts only, no personal data.\n"
        "- **Limitations:** reported cases undercount true infections; 2020–21 reporting was disrupted by "
        "COVID-19; weather at one point per district.")

