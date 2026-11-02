# Dengue outbreak early warning for Sri Lanka

MSc in AI — Machine Learning assignment.

This project predicts weekly dengue cases and flags outbreak weeks for each Sri Lankan district, using recent case counts and lagged weather. It uses two algorithms that weren't covered in the lectures:

| Task | New algorithm | Lecture baselines |
|---|---|---|
| Weekly case count (regression) | **Negative Binomial GLM** (`statsmodels`) | Linear Regression, Random Forest, XGBoost |
| Outbreak week (classification) | **RuleFit** (`imodels`) | Logistic Regression, Random Forest, XGBoost |

The models are explained with incidence rate ratios, RuleFit rules, SHAP, PDP and LIME. A Streamlit app lets users enter a scenario and see the predictions and explanations.

## Data

| Source | What | How it was obtained |
|---|---|---|
| Epidemiology Unit, Ministry of Health, Sri Lanka: weekly epidemiological reports | Weekly notified dengue cases for 26 districts/health areas, Dec 2006 – Dec 2025 (25,766 rows) | Compiled tidy table `srilanka_weekly_data` from the CRAN package [`denguedatahub`](https://cran.r-project.org/package=denguedatahub), downloaded automatically |
| Open-Meteo historical weather API (ERA5 reanalysis) | Daily rainfall, mean temperature and humidity at each district's centre point | Free public API, no key needed |

All data is aggregated counts per district. It contains no personal or sensitive information.

## Project structure

```
data/raw/            downloaded data (created by collect_data.py)
data/processed/      model-ready features (created by preprocess.py)
src/
  collect_data.py    1. download dengue + weather data
  preprocess.py      2. weekly weather, lag features, outbreak label, time split
  eda.py             3. exploratory plots
  train_evaluate.py  4. tune, train and test all models
  explain.py         5. IRRs, RuleFit rules, SHAP, PDP, LIME
app/streamlit_app.py front end
models/              saved models (.joblib)
reports/figures/     all plots (01-20)
reports/results/     metric tables, tuning logs, coefficients, rules
```

## How to run

From the project folder (on Windows use `py`; elsewhere use `python`):

```bash
pip install -r requirements.txt
py src/collect_data.py       # ~5 min (downloads)
py src/preprocess.py
py src/eda.py
py src/train_evaluate.py     # ~10 min (RuleFit tuning)
py src/explain.py            # ~3 min (SHAP for RuleFit)
py -m streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501.

## Method summary

- **Split by time, no shuffling:** train 2007–2019, validation 2020–2021, test 2022–2025.
- **Outbreak label:** a week's cases exceed the district's mean + 2 SD over the previous 52 weeks. This "endemic channel" uses past data only.
- **Tuning:** hyperparameters are chosen on validation, then the models are refit on train+validation and evaluated once on test. NegBin's dispersion α is estimated by maximum likelihood.
- **Metrics:**
  - Counts: MAE, RMSE, R², Poisson deviance and 90% prediction-interval coverage.
  - Outbreaks: ROC-AUC, PR-AUC, F1, precision and recall.

## Results (test set, 2022–2025)

| Count model | MAE | RMSE | R² | 90% PI coverage |
|---|---|---|---|---|
| **Negative Binomial** | 12.8 | 27.9 | 0.80 | 93.5% |
| Linear Regression (log) | 11.9 | 26.7 | 0.81 | – |
| Random Forest | 11.8 | 26.1 | 0.82 | – |
| XGBoost | 11.8 | 26.1 | 0.82 | – |

| Outbreak model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **RuleFit** | 0.84 | 0.43 | 0.46 |
| Logistic Regression | 0.85 | 0.40 | 0.43 |
| Random Forest | 0.86 | 0.40 | 0.44 |
| XGBoost | 0.87 | 0.46 | 0.46 |

8.8% of test weeks are outbreak weeks, so a random classifier would score a PR-AUC of about 0.09.
