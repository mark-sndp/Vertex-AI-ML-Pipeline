# Data Stage — `data/`

## Purpose
Simulates readings from an industrial rotating machine (pump/motor) and produces
the train/test datasets consumed by every downstream stage.

## Design

`generate_dataset.py` exposes two functions:

- **`simulate_iiot_data(num_samples, seed, drift)`** — pure-numpy generator. Draws
  four independent sensor features from normal distributions, clipped to
  physically sensible ranges:
  - `vibration_mm_s` — mean 2.5, std 0.6
  - `pressure_bar` — mean 5.0, std 0.8
  - `rpm` — mean 1500, std 120
  - `load_pct` — mean 60, std 15

  The target, `temperature_c`, is a deterministic linear combination of the four
  features plus Gaussian noise (`std=1.5`):

  $$temperature\_c = 35 + 4.0\cdot vibration + 0.6\cdot pressure + 0.01\cdot rpm + 0.25\cdot load + \epsilon$$

  This keeps the regression task easy enough for a tiny model to learn (the POC
  goal), while still requiring the model to combine all four inputs.

- **`generate_and_save(output_dir, drift)`** — generates `NUM_SAMPLES` rows
  (`config.NUM_SAMPLES`, default 2000), takes a `TEST_SPLIT` (20%) slice as the
  holdout set, and writes `train.csv` / `test.csv` to `data/raw/` (git-ignored).
  If `config.USE_VERTEX_AI` is `True`, it additionally uploads both files to
  `gs://<GCS_BUCKET>/<GCS_PREFIX>/data/` via `google.cloud.storage`; otherwise it
  just logs the GCS path it *would* upload to.

## The `drift` parameter

`drift` shifts the vibration mean (`+drift`) and load mean (`+drift*10`) upward,
simulating equipment wear. It defaults to `0.0` for the main pipeline run and is
passed as a nonzero value by [`retraining/retrain.py`](../retraining/retrain.py)
to justify re-running the pipeline with "newer, degraded" sensor data.

## Why this design

- **Deterministic formula + noise** rather than a black-box simulator: makes it
  trivial to reason about what "good" model performance should look like, and
  keeps the POC self-contained (no external dataset dependency).
- **Local-first, GCS-optional**: the same function works whether or not a real
  GCP project is configured, which is what lets the whole pipeline be verified
  offline (`USE_VERTEX_AI=False`, the default — see the root [README](../README.md)).
