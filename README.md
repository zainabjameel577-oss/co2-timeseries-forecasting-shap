# CO₂ Time-Series Forecasting with SHAP Interpretability

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

An end-to-end data science project that forecasts atmospheric CO₂ (Mauna Loa, monthly, 1958–2001) with **SARIMA**, **Prophet** and a **gradient-boosting challenger**, then explains the models with **SHAP**. The pipeline tests for stationarity, applies differencing, produces 95% prediction intervals, and exports a whitepaper and a visual report.

![Dashboard](outputs/figures/11_dashboard.png)

## Highlights

- **Stationarity-aware modelling:** ADF and KPSS tests decide the differencing. The raw series is non-stationary, and a first difference (d=1) passes both tests.
- **Three models plus a baseline:** SARIMA(1,1,1)(1,0,1,12) chosen by AIC, Prophet, a lag-feature gradient-boosting model (GBM), and a seasonal-naive benchmark.
- **Calibrated uncertainty:** every model produces 95% prediction intervals, checked against a 36-month hold-out.
- **Interpretability layer:** exact TreeSHAP importance weights, plus a surrogate model that lets SHAP explain SARIMA.
- **Reproducible and modular:** fixed seed, separated modules, unit tests, auto-generated reports.

## Results (36-month hold-out)

| Model | MAE | RMSE | MAPE % | 95% PI coverage |
|---|---|---|---|---|
| Seasonal Naive | 2.87 | 3.11 | 0.774 | 13.9% |
| SARIMA | 0.95 | 1.05 | 0.257 | 94.4% |
| Prophet | 0.80 | 0.89 | 0.215 | 77.8% |
| **GBM (lag features)** | **0.31** | **0.37** | **0.083** | 100% |

Errors are in ppm. GBM is the most accurate model, and SARIMA has the best-calibrated intervals.

**What drives the forecasts?** The feature `diff_lag12` (the change in the same month one year earlier) accounts for about 67% of the GBM's SHAP attribution and about 73% of the SARIMA surrogate's. The annual cycle is the main signal, and both models agree on this.

**Outlook:** refitted on all data, the GBM forecasts about 373.4 ppm for Dec 2003 (95% interval 371.8–374.9).

## Project structure

```
├── config/settings.py        # paths, seeds, constants
├── data/
│   ├── raw/                  # source series
│   └── processed/            # cleaned monthly series
├── src/
│   ├── data_loader.py        # loading and train/test split
│   ├── stationarity.py       # ADF, KPSS, STL, differencing selection
│   ├── features.py           # lag features (single source of truth)
│   ├── models.py             # SARIMA, Prophet, GBM, seasonal naive
│   ├── evaluation.py         # metrics, interval coverage, Ljung-Box
│   ├── interpretability.py   # SHAP wrapper and SARIMA surrogate
│   └── visualization.py      # figures and dashboard
├── models/                   # saved best model
├── outputs/
│   ├── figures/              # all charts
│   ├── tables/               # CSV results
│   └── results.json          # every number used in the reports
├── reports/
│   ├── build_report.py       # builds both PDFs
│   ├── Whitepaper_TimeSeries_Forecasting_Interpretability.pdf
│   └── Project_Visual_Report.pdf
├── tests/                    # pytest suite
├── main.py                   # runs the full pipeline
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/<your-username>/co2-timeseries-forecasting-shap.git
cd co2-timeseries-forecasting-shap

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                   # full pipeline (about 10 seconds)
python reports/build_report.py   # whitepaper and visual report
pytest -q                        # run tests
```

## Methodology

1. **Data:** weekly CO₂ aggregated to monthly means, with 5 gaps filled by time interpolation. The last 36 months are held out and all decisions use the training window only.
2. **Stationarity:** a battery of ADF and KPSS tests on the raw, differenced and seasonally differenced series. The least-differenced option passing both tests is selected.
3. **Modelling:** SARIMA via a 24-candidate AIC grid search, Prophet with yearly seasonality, and a GBM predicting month-over-month change from eight lag and calendar features, with bootstrap intervals.
4. **Evaluation:** MAE, RMSE, MAPE, interval coverage and width, plus Ljung-Box residual checks (p = 0.63 for SARIMA).
5. **Interpretability:** TreeSHAP for the GBM, with a check that attributions sum exactly to each prediction. A surrogate GBM mimics SARIMA's predictions (in-sample R² = 0.995) so both models share one explanation vocabulary.

## Reports

- [Whitepaper (PDF)](reports/Whitepaper_TimeSeries_Forecasting_Interpretability.pdf): a 5-page write-up of method, results and interpretation.
- [Visual report (PDF)](reports/Project_Visual_Report.pdf): a 2-page dashboard and pipeline overview.

## Using your own data

Replace `load_co2_monthly()` in `src/data_loader.py` with a function that returns a monthly `pd.Series` with a `DatetimeIndex` (frequency `MS`). If your data is not monthly, change `SEASON` in `config/settings.py`. Everything else, including the reports, regenerates automatically.

## Limitations

- Model ranking rests on a single 36-month hold-out, so a rolling-origin backtest would make it more robust.
- GBM intervals reuse in-sample residuals and are likely optimistic out-of-sample.
- The SARIMA explanation comes from a surrogate, not from SARIMA's own parameters.
- The series is univariate and ends in 2001, and SHAP shows association rather than causation.

## Roadmap

- [ ] Rolling-origin cross-validation
- [ ] Ensemble of SARIMA and GBM
- [ ] Conformal prediction intervals
- [ ] Exogenous regressors
- [ ] GitHub Actions CI

## Tech stack

Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, Prophet, SHAP, Matplotlib, ReportLab, pytest.

## License

Released under the MIT License. See `LICENSE` for details.

## Author

**Your Name** | [GitHub](https://github.com/<your-username>) | [LinkedIn](https://www.linkedin.com/in/<your-profile>)
