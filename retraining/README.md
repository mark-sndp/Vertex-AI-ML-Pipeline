# Retraining Stage — `retraining/`

## Purpose
Closes the MLOps loop: a manually-triggered script that regenerates data,
retrains, re-evaluates, and only re-registers/redeploys/re-monitors if the new
model clears the evaluation gate.

## Design

`retrain.py`'s `retrain(drift=0.5)`:

1. **Regenerate data with simulated drift** —
   [`data/generate_dataset.py`](../data/generate_dataset.py)'s `generate_and_save(drift=drift)`
   shifts the vibration/load distributions upward, standing in for "the
   machine has degraded since the model was first trained" — the realistic
   reason a retraining pipeline would fire in production (alongside a real
   `ModelDeploymentMonitoringJob` drift alert).
2. **Retrain** — calls `training/train_launcher.py`'s `launch_training()`
   exactly as the main pipeline does, on the new data.
3. **Re-evaluate** — calls `evaluation/evaluate.py`'s `evaluate()`.
4. **Conditional promotion** — only if `result["passed"]` is `True` does it call
   `register_model()` → `deploy_model(model)` → `setup_monitoring(endpoint)`,
   in that order, reusing the exact same stage functions the main pipeline
   uses. If the new model fails the gate, it stops and reports the failure —
   the previously deployed model is left untouched.

## Why this design

- **Manual trigger, not scheduled/event-driven** — this keeps the POC's scope
  small (no Cloud Scheduler / Pub/Sub / Cloud Functions wiring) while still
  demonstrating the concept: run `python retraining/retrain.py` any time you
  want to simulate "new data arrived, let's see if we should update the model."
  A production system would instead wire this to a Cloud Scheduler cron job or
  a Pub/Sub message triggered by the monitoring job's drift alert.
- **Reuses the same stage functions as the main pipeline** (`register_model`,
  `deploy_model`, `setup_monitoring`) rather than duplicating logic — retraining
  is "the same pipeline, run again," not a separate code path.
- **Gate re-checked on every retrain**, not skipped — guards against silently
  deploying a worse model just because retraining happened.
