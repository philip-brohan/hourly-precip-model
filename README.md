# hourly-precip-model-MO

XGBoost pipeline to estimate precipitation at any global location from local temperature, pressure, location, and time metadata.

## Model Objective

- Target output: `PRATE` (precipitation rate)
- Inputs:
	- `TMP2m` (current + lagged values)
	- `PRMSL` (current + lagged values)
	- Latitude, longitude
	- Cyclic time encodings (day-of-year and hour-of-day)

This implementation uses:

- Train years: 1961-1990
- Validation years: 1991-2000
- Test years: 2001-2014
- Ensemble members: `mem001` to `mem005`
- Lags: 8 (24 hours back at 3-hourly data spacing)
- Spatial sampling: random grid-point subsampling per timestep

## Data Source

Expected input directory structure:

`/data/scratch/philip.brohan/MLP/20CR/version_3/hourly/yyyy/VAR.yyyy_memNNN.nc`

with `VAR` in `{PRATE, TMP2m, PRMSL}`.

## Environment Setup (Conda)

```bash
conda env create -f environment.yml
conda activate precip-xgb
```

## Pipeline Scripts

- `src/01_explore_data.py`
	- Quick structural and metadata inspection for sample netCDF files.
- `src/02_make_training_data.py`
	- Creates sampled lagged datasets in HDF5: `data/train.h5`, `data/val.h5`, `data/test.h5`.
- `src/03_train_model.py`
	- Trains XGBoost with early stopping and writes model + feature importance.
- `src/04_evaluate_model.py`
	- Runs diagnostics and writes figures and summary metrics.

## Run Order

### 1) Inspect data files

```bash
python src/01_explore_data.py --year 2000 --member 1
```

### 2) Build sampled train/val/test datasets

```bash
python src/02_make_training_data.py \
	--base-dir /data/scratch/philip.brohan/MLP/20CR/version_3/hourly \
	--train-years 1961:1990 \
	--val-years 1991:2000 \
	--test-years 2001:2014 \
	--members 1-5 \
	--lags 8 \
	--samples-per-timestep 200 \
	--out-dir data
```

### 3) Train XGBoost model

```bash
python src/03_train_model.py \
	--train-file data/train.h5 \
	--val-file data/val.h5 \
	--model-out models/xgb_precip.json \
	--importance-out models/feature_importance.csv
```

### 4) Evaluate and generate diagnostics

```bash
python src/04_evaluate_model.py \
	--test-file data/test.h5 \
	--model-file models/xgb_precip.json \
	--fig-dir figures
```

## Diagnostics Produced

The evaluation stage writes:

- Scatter plot: observed vs predicted precipitation
- Distribution comparison (observed vs predicted)
- Q-Q plot
- Monthly (seasonal) mean cycle
- Bias map (predicted minus observed)
- Correlation map
- Wet-frequency maps (observed, predicted, and difference)
- Feature-importance plot
- JSON summary with key metrics (`RMSE`, `MAE`)

## Notes

- The target is modeled as `log1p(PRATE)` during training; diagnostics report raw-space metrics.
- The full archive is very large; tune `--samples-per-timestep`, `--max-train-rows`, and `--max-test-rows` to control memory/runtime.
