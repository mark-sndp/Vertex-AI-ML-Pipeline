# Evaluation Stage — `evaluation/`

## Purpose
Scores the trained model against the holdout test set and acts as the **gate**
that decides whether the pipeline proceeds to model registry/deployment/monitoring.

## Design

`evaluate.py`'s `evaluate(model_dir, test_data_path)`:

1. Loads `test.csv` (via `tf.io.gfile`, so local or `gs://` paths both work).
2. Loads `normalization.json` written by the training stage and applies the
   **same** mean/std used at train time — the test set must never be used to
   compute its own normalization statistics (that would leak information and
   also wouldn't match what a real served model receives).
3. Loads `saved_model.keras` and predicts.
4. Computes three standard regression metrics via scikit-learn:
   - `MAE` (mean absolute error, °C) — the primary gating metric, easy to
     communicate ("model is off by ~1.3°C on average").
   - `RMSE` — penalizes larger errors more, useful as a secondary signal.
   - `R²` — fraction of variance explained; sanity-checks that the model is
     actually learning the relationship rather than predicting the mean.
5. Compares `MAE` against `config.MAX_ACCEPTABLE_MAE` (default `3.0`) to produce
   a boolean `passed` verdict.
6. Writes the full result dict to `artifacts/evaluation_result.json`.

## Why this design

- **A single explicit threshold** (rather than e.g. comparing against a
  previous model's metrics) keeps the POC simple and easy to reason about; it's
  the same mechanism a real pipeline would use for a "does this model meet our
  bar" gate before promoting to registry.
- **Fail-closed**: [`pipeline/run_pipeline.py`](../pipeline/run_pipeline.py) and
  [`retraining/retrain.py`](../retraining/retrain.py) both check `passed` and
  stop *before* registry/deployment/monitoring if the gate fails — a bad model
  never reaches the registry stage, mirroring how a real CI/CD-for-ML gate
  should behave.
