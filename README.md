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
