# Monitoring Stage — `monitoring/`

## Purpose
Configures skew/drift detection on the deployed endpoint's input features, so
that a real deployment would get alerted when incoming sensor readings drift
away from the training distribution.

## Design

`setup_monitoring.py`'s `setup_monitoring(endpoint)`:

- When `config.USE_VERTEX_AI` is `False` (default): logs the exact
  `aiplatform.ModelDeploymentMonitoringJob.create(...)` call it would make,
  including the feature list, drift threshold, sample rate, and alert email.
- When `config.USE_VERTEX_AI` is `True`, builds:
  - `model_monitoring.DriftDetectionConfig` — one `drift_threshold`
    (`config.DRIFT_THRESHOLD`, default `0.3`) per input feature
    (`config.FEATURE_COLUMNS`: vibration, pressure, rpm, load).
  - `model_monitoring.EmailAlertConfig` — sends alerts to
    `config.MONITORING_ALERT_EMAIL`.
  - `aiplatform.model_monitoring.RandomSampleConfig` — samples
    `config.MONITORING_SAMPLE_RATE` (80%) of prediction requests for analysis
    rather than every single request, trading monitoring completeness for cost.
  - Then calls `aiplatform.ModelDeploymentMonitoringJob.create(...)` against the
    `endpoint` passed in (whatever [`deployment/deploy_endpoint.py`](../deployment/deploy_endpoint.py) returned).

## Why this design

- **Feature drift, not just prediction drift**: since this is a regression on
  physical sensor readings, the interesting signal is "are incoming vibration/
  pressure/rpm/load values drifting from what the model was trained on" — which
  is exactly what would precede real equipment degrading and the model's
  predictions becoming unreliable. This is also the story
  [`retraining/retrain.py`](../retraining/retrain.py) simulates via its `drift`
  parameter.
- Kept as its own stage (rather than folded into deployment) because in Vertex
  AI it's a genuinely separate long-running resource
  (`ModelDeploymentMonitoringJob`) with its own lifecycle.
