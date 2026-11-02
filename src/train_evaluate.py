"""Step 4: train and evaluate.

Task A (count):    predict weekly dengue cases per district
    New model:  Negative Binomial GLM (statsmodels)
    Baselines:  Linear Regression (on log cases), Random Forest, XGBoost
Task B (outbreak): classify outbreak weeks
    New model:  RuleFit (imodels)
    Baselines:  Logistic Regression, Random Forest, XGBoost

Protocol: fit on train (2007-2019), tune on validation (2020-2021),
refit the chosen settings on train+val, report once on test (2022-2025).

Run:  py src/train_evaluate.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import RuleFitClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_poisson_deviance,
                             mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RES = ROOT / "reports" / "results"
MOD = ROOT / "models"
for p in (FIG, RES, MOD):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42

NUMERIC = (["log_cases_lag1", "log_cases_lag2", "log_cases_mean4_lag1",
            "rain_4wk_sum_lag1", "rain_4wk_sum_lag6", "rain_mm_lag8", "week_sin", "week_cos",
            "log_threshold", "excess_lag1"]
           + [f"{c}_lag{k}" for c in ["rain_mm", "temp_c", "humidity"] for k in [1, 2, 3, 4]])
CATEGORICAL = ["district", "monsoon"]
# Exact linear combinations of other columns (4-wk sum = sum of 4 lags; excess = cases - threshold).
# Trees are fine with them, but they make linear/GLM coefficients unidentifiable.
LINEAR_DROP = ["rain_4wk_sum_lag1", "excess_lag1"]


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv(ROOT / "data" / "processed" / "dengue_features.csv", parse_dates=["start"])
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, drop_first=True, dtype=float)
    return df, X


def scale(train, *others):
    sc = StandardScaler().fit(train)
    f = lambda d: pd.DataFrame(sc.transform(d), columns=d.columns, index=d.index)
    return sc, f(train), *[f(o) for o in others]


# ---------------------------------------------------------------- Task A: counts
def count_metrics(y, p):
    p = np.clip(p, 1e-6, None)
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5,
                R2=r2_score(y, p), PoissonDev=mean_poisson_deviance(y, p))


def fit_negbin(X, y, alpha):
    X = X.drop(columns=LINEAR_DROP)
    return sm.GLM(y, sm.add_constant(X, has_constant="add"),
                  family=sm.families.NegativeBinomial(alpha=alpha)).fit()


def pred_negbin(m, X):
    return m.predict(sm.add_constant(X.drop(columns=LINEAR_DROP), has_constant="add"))


def tune_count(Xtr, ytr, Xva, yva):
    """Choose hyperparameters by validation Poisson deviance."""
    log = []
    best = {}
    # Negative Binomial: dispersion alpha (Var = mu + alpha*mu^2).
    for a in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        dev = mean_poisson_deviance(yva, np.clip(pred_negbin(fit_negbin(Xtr, ytr, a), Xva), 1e-6, None))
        log.append(("NegBin", f"alpha={a}", dev))
    # The grid above is kept for the report, but Poisson deviance favours near-Poisson fits,
    # so alpha is set by maximum likelihood on the training data (the standard estimator).
    mle = sm.NegativeBinomial(ytr, sm.add_constant(Xtr.drop(columns=LINEAR_DROP), has_constant="add")).fit(
        method="lbfgs", maxiter=500, disp=0)
    best["NegBin"] = dict(alpha=round(float(mle.params["alpha"]), 4))
    # Random Forest.
    for d in [8, 12, None]:
        for leaf in [5, 20]:
            m = RandomForestRegressor(300, max_depth=d, min_samples_leaf=leaf, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
            log.append(("RandomForest", json.dumps(dict(max_depth=d, min_samples_leaf=leaf)),
                        mean_poisson_deviance(yva, np.clip(m.predict(Xva), 1e-6, None))))
    # XGBoost (Poisson objective, fair comparison with a count model).
    for d in [3, 5]:
        for lr in [0.03, 0.1]:
            m = XGBRegressor(n_estimators=400, max_depth=d, learning_rate=lr, subsample=0.8,
                             objective="count:poisson", random_state=SEED).fit(Xtr, ytr)
            log.append(("XGBoost", json.dumps(dict(max_depth=d, learning_rate=lr)),
                        mean_poisson_deviance(yva, m.predict(Xva))))
    for name in ["RandomForest", "XGBoost"]:
        best[name] = json.loads(min((r for r in log if r[0] == name), key=lambda r: r[2])[1])
    pd.DataFrame(log, columns=["model", "params", "val_poisson_dev"]).to_csv(RES / "tuning_count.csv", index=False)
    return best


def build_count(name, params):
    if name == "RandomForest":
        return RandomForestRegressor(300, n_jobs=-1, random_state=SEED, **params)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=400, subsample=0.8, objective="count:poisson", random_state=SEED, **params)


def run_count(df, X):
    tr, va, te = (df.split == s for s in ("train", "val", "test"))
    trva = tr | va
    y = df["cases"].astype(float)
    _, Xtr_s, Xva_s = scale(X[tr], X[va])
    best = tune_count(Xtr_s, y[tr], Xva_s, y[va])
    print("Best count params:", best)

    # Refit on train+val, evaluate on test.
    sc, Xtv_s, Xte_s = scale(X[trva], X[te])
    preds, rows = {}, []
    nb = fit_negbin(Xtv_s, y[trva], best["NegBin"]["alpha"])
    preds["NegBin (new)"] = pred_negbin(nb, Xte_s)
    lr = LinearRegression().fit(Xtv_s.drop(columns=LINEAR_DROP), np.log1p(y[trva]))
    preds["LinearReg (log)"] = np.expm1(lr.predict(Xte_s.drop(columns=LINEAR_DROP))).clip(0)
    models = {"negbin": nb, "linreg": lr}
    for name in ["RandomForest", "XGBoost"]:
        m = build_count(name, best[name]).fit(Xtv_s, y[trva])
        preds[name] = m.predict(Xte_s)
        models[name.lower()] = m
    for name, p in preds.items():
        rows.append(dict(model=name, **count_metrics(y[te], p)))
    res = pd.DataFrame(rows).round(3)
    # NegBin gives a full distribution per week: check 90% prediction-interval coverage.
    from scipy.stats import nbinom
    a = best["NegBin"]["alpha"]
    mu = np.asarray(preds["NegBin (new)"])
    n_, p_ = 1 / a, 1 / (1 + a * mu)
    lo, hi = nbinom.ppf(0.05, n_, p_), nbinom.ppf(0.95, n_, p_)
    yt = y[te].to_numpy()
    res["PI90_coverage"] = [((yt >= lo) & (yt <= hi)).mean().round(3)] + [np.nan] * (len(res) - 1)
    res.to_csv(RES / "test_count_metrics.csv", index=False)
    print("\nTask A - test set (2022-2025):\n", res.to_string(index=False))

    # NegBin coefficient table: exp(coef) = incidence rate ratio per 1 SD.
    coef = pd.DataFrame({"coef": nb.params, "IRR_per_SD": np.exp(nb.params), "p_value": nb.pvalues,
                         "ci_low": np.exp(nb.conf_int()[0]), "ci_high": np.exp(nb.conf_int()[1])})
    coef.round(4).to_csv(RES / "negbin_coefficients.csv")
    joblib.dump(dict(models=models, scaler=sc, columns=list(X.columns), best=best), MOD / "count_models.joblib")

    # Plot: national weekly totals on test, actual vs predicted.
    t = df[te].assign(**{k: np.asarray(v) for k, v in preds.items()})
    nat = t.groupby("start")[["cases", *preds]].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(nat.index, nat.cases, color="k", lw=2, label="Actual")
    for k in preds:
        plt.plot(nat.index, nat[k], lw=1, label=k)
    plt.title("Test period: national weekly cases, actual vs predicted"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "09_count_test_predictions.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 5))
    p = np.asarray(preds["NegBin (new)"])
    plt.scatter(y[te], p, s=4, alpha=0.3)
    m = max(y[te].max(), p.max())
    plt.plot([0, m], [0, m], "k--"); plt.xscale("symlog"); plt.yscale("symlog")
    plt.xlabel("Actual cases"); plt.ylabel("Predicted cases"); plt.title("Negative Binomial: test set")
    plt.tight_layout(); plt.savefig(FIG / "10_negbin_pred_vs_actual.png", dpi=150); plt.close()
    return res


# ---------------------------------------------------------------- Task B: outbreak
def tune_threshold(y, p):
    """Pick the probability cut-off that maximises F1 on validation."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])])


def cls_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                F1=f1_score(y, yhat), Precision=precision_score(y, yhat, zero_division=0),
                Recall=recall_score(y, yhat), threshold=thr)


def build_cls(name, params):
    if name == "RuleFit":
        return RuleFitClassifier(random_state=SEED, **params)
    if name == "LogisticReg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", **params)
    if name == "RandomForest":
        return RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **params)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=400, subsample=0.8, eval_metric="aucpr", random_state=SEED, **params)


GRIDS = {
    "RuleFit": [dict(max_rules=m, tree_size=t) for m in [10, 20, 40] for t in [3, 4]],
    "LogisticReg": [dict(C=c) for c in [0.01, 0.1, 1.0]],
    "RandomForest": [dict(max_depth=d, min_samples_leaf=l) for d in [8, 12] for l in [5, 20]],
    "XGBoost": [dict(max_depth=d, learning_rate=lr, scale_pos_weight=8) for d in [3, 5] for lr in [0.03, 0.1]],
}


def run_outbreak(df, X):
    tr, va, te = (df.split == s for s in ("train", "val", "test"))
    trva = tr | va
    y = df["outbreak"]
    _, Xtr_s, Xva_s = scale(X[tr], X[va])
    raw = {"RuleFit"}  # trees don't need scaling; raw units give readable rules
    log, best, thresholds = [], {}, {}
    for name, grid in GRIDS.items():
        scores = []
        for params in grid:
            a, b = (X[tr], X[va]) if name in raw else (Xtr_s, Xva_s)
            p = build_cls(name, params).fit(a, y[tr]).predict_proba(b)[:, 1]
            ap = average_precision_score(y[va], p)
            scores.append((ap, params, tune_threshold(y[va], p)))
            log.append((name, json.dumps(params), ap))
        ap, best[name], thresholds[name] = max(scores, key=lambda s: s[0])
        print(f"{name}: best {best[name]} val PR-AUC={ap:.3f}")
    pd.DataFrame(log, columns=["model", "params", "val_pr_auc"]).to_csv(RES / "tuning_outbreak.csv", index=False)

    sc, Xtv_s, Xte_s = scale(X[trva], X[te])
    rows, probs, models = [], {}, {}
    for name in GRIDS:
        a, b = (X[trva], X[te]) if name in raw else (Xtv_s, Xte_s)
        m = build_cls(name, best[name]).fit(a, y[trva])
        p = m.predict_proba(b)[:, 1]
        label = f"{name} (new)" if name == "RuleFit" else name
        probs[label] = p
        models[name.lower()] = m
        rows.append(dict(model=label, **cls_metrics(y[te], p, thresholds[name])))
    res = pd.DataFrame(rows).round(3)
    res.to_csv(RES / "test_outbreak_metrics.csv", index=False)
    print("\nTask B - test set (2022-2025):\n", res.to_string(index=False))

    rules = models["rulefit"]._get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    rules.to_csv(RES / "rulefit_rules.csv", index=False)
    joblib.dump(dict(models=models, scaler=sc, columns=list(X.columns), best=best,
                     thresholds=thresholds), MOD / "outbreak_models.joblib")

    # ROC and PR curves.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, p in probs.items():
        fpr, tpr, _ = roc_curve(y[te], p); ax[0].plot(fpr, tpr, label=f"{k} ({roc_auc_score(y[te], p):.2f})")
        pr, rc, _ = precision_recall_curve(y[te], p); ax[1].plot(rc, pr, label=f"{k} ({average_precision_score(y[te], p):.2f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test)")
    ax[1].axhline(y[te].mean(), ls="--", color="k"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "11_outbreak_roc_pr.png", dpi=150); plt.close()

    cm = confusion_matrix(y[te], (probs["RuleFit (new)"] >= thresholds["RuleFit"]).astype(int))
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Normal", "Outbreak"]); plt.yticks([0, 1], ["Normal", "Outbreak"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("RuleFit confusion matrix (test)")
    plt.tight_layout(); plt.savefig(FIG / "12_rulefit_confusion.png", dpi=150); plt.close()
    return res, rules


def main():
    df, X = load()
    print(f"{len(df):,} rows, {X.shape[1]} features")
    run_count(df, X)
    _, rules = run_outbreak(df, X)
    print("\nTop RuleFit rules:\n", rules[["rule", "type", "coef", "importance"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
