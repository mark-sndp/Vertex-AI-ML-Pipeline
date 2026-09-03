# Training Stage — `training/`

## Purpose
Trains a small TensorFlow/Keras regression model that predicts `temperature_c`
from the four IIoT sensor features, and packages that model for Vertex AI.

## Files

### `task.py` — the trainable script

This is the actual training logic, deliberately written so it can run:
- standalone/locally (`python training/task.py --train-data ... --output-dir ...`), or
- as the entrypoint of a Vertex AI `CustomJob` training container (same script, same args).

Key pieces:

- **`build_model(num_features, learning_rate)`** — a plain `tf.keras.Sequential`:
  `Input(4) → Dense(16, relu) → Dense(8, relu) → Dense(1)`, compiled with
  `Adam` + `mse` loss and `mae` metric. Intentionally tiny — this is a POC, not
  a tuned production model.
- **`load_dataset(path)`** — uses `tf.io.gfile.GFile`, which transparently
  supports both local paths and `gs://` URIs. This is what lets `task.py` run
  unmodified whether the data lives on disk or in Cloud Storage.
- **`train(...)`**:
  1. Loads the CSV and splits into features `x` / target `y`.
  2. **Normalizes features using train-set mean/std** (important: a 4-feature
     input with wildly different scales — e.g. `rpm` ~1500 vs `vibration` ~2.5 —
     would otherwise dominate the loss and slow convergence).
  3. Fits the model (`validation_split=0.1`) for `config.EPOCHS` epochs.
  4. Saves two model artifacts to `output_dir`:
     - `saved_model.keras` — native Keras format, used by `evaluation/evaluate.py`
       for local scoring.
     - `1/` — a `SavedModel` export (`model.export(...)`), the format Vertex AI's
       prebuilt TensorFlow serving container expects (`artifact_uri` must point
       at a directory containing a numbered version subfolder).
  5. Saves `normalization.json` (the mean/std used) next to the model, so
     evaluation/serving can apply the *exact same* normalization at inference
     time — never recompute stats on eval/serving data.
  6. Writes `metrics.json` twice: once under `output_dir` (co-located with the
     model artifact, useful if `output_dir` is a fresh Vertex AI job directory)
     and once at the shared `config.METRICS_PATH` (`artifacts/metrics.json`) so
     other stages/scripts have a single well-known place to read it from.

### `train_launcher.py` — the Vertex AI submission wrapper

Decouples "how do I run training" from `task.py`'s "what does training do":

- **`run_local(...)`** — imports `task.train` directly and calls it in-process.
  Used when `config.USE_VERTEX_AI` is `False` (the default), so the pipeline can
  be verified without any GCP credentials or network calls.
- **`run_on_vertex(...)`** — uses
  `aiplatform.CustomJob.from_local_script(...)` to package `task.py` into a
  prebuilt TensorFlow training container (`config.TRAINING_CONTAINER_IMAGE`) and
  submit it as a real Vertex AI `CustomJob`, then blocks (`job.run(sync=True)`)
  until it completes and reads back `metrics.json`.
- **`launch_training(...)`** — the single entrypoint other stages/the pipeline
  orchestrator call; picks one of the above based on `config.USE_VERTEX_AI`.

## Why this design

- Splitting "trainable script" from "launcher" mirrors how you'd actually
  structure a Vertex AI training job: `task.py` is what gets containerized,
  `train_launcher.py` is what you'd otherwise run from a notebook or a Vertex AI
  Pipeline `CustomTrainingJobOp`.
- Feature normalization stats are persisted rather than hardcoded, so
  evaluation and (real) serving stay consistent with what the model was
  trained on.

## Tuning note

With the default toy architecture, `EPOCHS=150` / `LEARNING_RATE=0.01` are
needed to converge below the evaluation gate's default MAE threshold
(`config.MAX_ACCEPTABLE_MAE = 3.0`); far fewer epochs/lower LR under-fits (MAE
stayed around 7–8, verified during implementation).
