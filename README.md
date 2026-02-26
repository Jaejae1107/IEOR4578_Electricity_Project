# Electricity Load Forecasting Preprocessing (LD2011-2014)

Reproducible preprocessing pipeline for `LD2011_2014.txt`.
Scope is preprocessing **Step 1-3** only.

## 1) What This Repository Produces

- Hourly wide master dataset (active clients only)
- Metadata describing preprocessing rules and removed entities
- Optional aggregate series
- CSV exports for model teams (Level 1/2/3)

## 2) Preprocessing Rules

- Input resolution: 15-min (`kW`)
- Target definition: `hourly_kW_mean_of_4`
- Processing order: downsample/filter in wide format before any melt
- DST policy: drop entire transition dates
- Inactive client filter:
  - `nonzero_rate < 0.01`, or
  - `max_consecutive_zeros_hours > 720`
- Numeric parsing: `decimal=','` (required for this dataset)

## 3) Run

```bash
make run
```

or

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 run_pipeline.py --config config.json
```

## 4) Test

```bash
make test
```

## 5) Complete File Inventory (Current)

### Root files

- `.gitignore`: ignore cache artifacts (`__pycache__`, `*.pyc`, `.DS_Store`)
- `LICENSE`: MIT license
- `Makefile`: run/test/clean commands
- `README.md`: project documentation
- `config.json`: pipeline configuration (input path, rules, output names)
- `requirements.txt`: Python dependencies
- `run_pipeline.py`: main CLI entrypoint (Step 1 -> Step 2 -> Step 3)
- `preprocess_ld.py`: legacy single-file runner (kept for compatibility)

### Source package (`src/ld_preprocessing`)

- `__init__.py`: package marker
- `pipeline.py`: orchestration logic and metadata assembly
- `step1_load_integrity.py`: Step 1 (load, parse, integrity checks)
- `step2_hourly_dst_inactive.py`: Step 2 (hourly downsample, DST drop, inactive filtering)
- `step3_save_outputs.py`: Step 3 (save master + metadata + optional aggregate)

### Tests (`tests`)

- `test_step1.py`: Step 1 unit tests
- `test_step2.py`: Step 2 unit tests
- `test_step3.py`: Step 3 unit tests
- `test_pipeline.py`: end-to-end pipeline test on tiny synthetic data
- `tests/__pycache__/...`: auto-generated cache files (not source)

### Generated output data

Core parquet outputs:
- `master_wide_hourly.parquet`: main preprocessed dataset (timestamp x active clients)
- `master_metadata.json`: preprocessing metadata/reproducibility record
- `aggregate_hourly.parquet`: optional aggregate series

CSV exports:
- `master_wide_hourly.csv`: CSV version of hourly wide master
- `aggregate_hourly.csv`: CSV version of aggregate series
- `calendar_features_hourly.csv`: timestamp-level calendar features for exogenous models
- `master_long_hourly.csv`: long-format panel (`timestamp`, `client_id`, `y`)
- `active_clients.csv`: active client ID list

## 6) CSV Usage by Model Level

1. Level 1: Pure Endogenous Baselines (AutoARIMA, AutoETS)
- Use: `master_long_hourly.csv` (`unique_id`, `ds`, `y` format expected by statsforecast)
- Optional: `aggregate_hourly.csv` for single-series benchmark

2. Level 2: Covariate Baselines (SARIMAX, Prophet)
- Use: `master_long_hourly.csv` + `calendar_features_hourly.csv` (join on `timestamp`)

3. Level 3: Global Deep Learning (e.g. iTransformer)
- Use: `master_wide_hourly.csv` (each client column is a variate for multivariate models)
- Join: `calendar_features_hourly.csv` on `timestamp`
- Optional indexing helper: `active_clients.csv`

## 7) GitHub Upload

```bash
git add .
git commit -m "Finalize LD2011-2014 preprocessing repo (steps 1-3)"
git push
```

## 8) Time-Based Data Split (Train / Validation / Test)

We additionally provide fixed chronological splits for model development:

- **Train:** January 2012 – December 2013  
- **Validation:** January 2014 – April 2014  
- **Test:** May 2014 – December 2014  

All split boundaries are inclusive at hourly resolution:
- Train: `2012-01-01 00:00:00` to `2013-12-31 23:00:00`
- Validation: `2014-01-01 00:00:00` to `2014-04-30 23:00:00`
- Test: `2014-05-01 00:00:00` to `2014-12-31 23:00:00`

No timestamp overlap exists between the three splits.

### Wide-format split files

- `master_wide_hourly_train_2012_2013.csv` — shape `(17448, 156)`
- `master_wide_hourly_validation_2014_01_04.csv` — shape `(2856, 156)`
- `master_wide_hourly_test_2014_05_12.csv` — shape `(5856, 156)`

### Long-format split files

- `master_long_hourly_train_2012_2013.csv` — shape `(2721888, 3)`
- `master_long_hourly_validation_2014_01_04.csv` — shape `(445536, 3)`
- `master_long_hourly_test_2014_05_12.csv` — shape `(913536, 3)`

## 9) Modeling Step 2 (`src/modeling_step2/`)

Three forecasting models are implemented for **24-hour-ahead** electricity load forecasting across 156 active clients.

### Models

#### SARIMAX.ipynb — AutoARIMA (Level 2)

- Library: `statsforecast` `AutoARIMA`
- Format: long-format (`master_long_hourly_*.csv`)
- Training window: last 672 hours (4-week lookback) per client
- Exogenous features: `hour_sin/cos`, `dow_sin/cos`, `is_weekend`, `month_sin/cos`
- Season length: 24 (daily); stepwise + approximation enabled for speed
- Parallel training: `n_jobs=-1`
- Saved models: `sarimax_val.joblib`, `sarimax_final.joblib`

**Test results (156 clients, overall):** MSE = 267,842 | MAE = 134.84 | WAPE = 0.197

#### Prophet.ipynb — Prophet (Level 2)

- Library: `prophet`
- Format: long-format (`master_long_hourly_*.csv`)
- One independent model per client (156 models total)
- Seasonalities: daily, weekly, yearly (additive mode)
- Exogenous regressor: `is_weekend`
- Saved models: `prophet_val.joblib`, `prophet_final.joblib`

**Test results (156 clients, overall):** MSE = 256,237 | MAE = 113.45 | WAPE = 0.166

#### iTransformer.ipynb — iTransformer (Level 3)

- Library: `neuralforecast` `iTransformer`
- Format: wide-format (`master_wide_hourly_*.csv`) converted to long format for NeuralForecast
- Global model shared across all 156 clients
- Horizon: 24 h | Input size: 672 h (4-week lookback)
- Architecture: hidden=512, heads=8, encoder layers=2, decoder layers=1, dropout=0.1
- Loss: MSE (train) / MAE (validation); early stopping patience=10 steps
- Exogenous features: `hour_sin/cos`, `dow_sin/cos`, `is_weekend`, `month_sin/cos`
- ~8.1M trainable parameters
- Saved models: `itransformer_val/`, `itransformer_final/`

**Test results (156 clients, overall):** MSE = 177,343 | MAE = 100.96 | WAPE = 0.143

### Model Comparison (Test Set)

| Model | Test MSE | Test MAE | Test WAPE |
|-------|----------|----------|-----------|
| iTransformer | **177,343** | **100.96** | **0.143** |
| Prophet | 256,237 | 113.45 | 0.166 |
| AutoARIMA | 267,842 | 134.84 | 0.197 |

iTransformer achieves the best overall performance. All models show high per-client variance; outlier clients (e.g., MT_196, MT_279, MT_235) produce significantly elevated errors.

## 10) Modeling Step 1 (`src/modeling_step1/`)

Three forecasting models are implemented for **24-hour-ahead** electricity load forecasting across 156 active clients, with an additional aggregate single-series benchmark.

### Models

#### AutoETS.ipynb — AutoETS (Level 1)

- Library: `statsforecast`
- Format: long-format (`master_long_hourly_*.csv`)
- Training window: last 672 hours (4-week lookback) per client
- Exogenous features: None
- Season length: 24 (daily); automatically selects error, trend, seasonality components
- Parallel training: `n_jobs=-1`
- Saved models: `autoets_val.joblib`, `autoets_final.joblib`

**Test results (156 clients, overall):** MSE = 612,003 | MAE = 145.52 | WAPE = 0.213

#### AutoARIMA.ipynb — AutoARIMA (Level 1)

- Library: `statsforecast` `AutoARIMA`
- Format: single time series (`master_long_hourly_*.csv`)
- Training window: last 672 hours (4-week lookback) per client
- Exogenous features: None
- Season length: 24 (daily); stepwise + approximation enabled for speed
- Parallel training: `n_jobs=-1`
- Saved models: `autoarima_val.joblib`, `autoarima_final.joblib`

**Test results (156 clients, overall):** MSE = 155,072 | MAE = 99.11 | WAPE = 0.145

#### AutoARIMA_Aggregate.ipynb — AutoARIMA (Level 1)

- Library: `statsforecast` `AutoARIMA`
- Format: long-format (`aggregate_hourly.csv`)
- Training window: last 672 hours (4-week lookback)
- Exogenous features: None
- Season length: 24 (daily); stepwise + approximation enabled for speed
- Saved models: `autoarima_agg_val.joblib`, `autoarima_agg_final.joblib`

**Test results (aggregate series):** MSE = 203,562,485 | MAE = 10,726.18 | WAPE = 0.100

### Model Comparison (Test Set)

| Model | Test MSE | Test MAE | Test WAPE |
|-------|----------|----------|-----------|
| AutoETS | 612,003 | 145.52 | 0.213 |
| AutoARIMA | 155,072 | 99.11 | 0.145 |
| AutoARIMA(agg) | 203,562,485 | 10,726.18 | 0.100 |

AutoARIMA outperformed AutoETS (WAPE 0.213); AutoETS likely anchored on the late December 2013 consumption peak, causing persistent overestimation early in the validation period. Aggregate benchmark (WAPE 0.1004) showed lower error than per-client (WAPE 0.1448)