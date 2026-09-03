# Model Registry Stage — `registry/`

## Purpose
Registers the (evaluation-passed) trained model as a versioned artifact in
Vertex AI Model Registry.

## Design

`register_model.py`'s `register_model(model_dir)`:

- When `config.USE_VERTEX_AI` is `False` (default): computes the `artifact_uri`
  that *would* be used (`model_dir/1`, the SavedModel export directory) and
  **logs** the exact `aiplatform.Model.upload(...)` call it would make, then
  returns a local stub object (`{"resource_name": "local-stub-model", ...}`) so
  downstream stages (deployment) have something to pass along without special-casing.
- When `config.USE_VERTEX_AI` is `True`: calls the real
  `aiplatform.Model.upload()`:
  - `artifact_uri` — the model's GCS location (must be a directory, hence the
    `1/` SavedModel export from the training stage, not the `.keras` file).
  - `serving_container_image_uri` — `config.SERVING_CONTAINER_IMAGE`, Google's
    prebuilt TensorFlow prediction container (no custom serving code needed for
    this POC).
  - `labels` — `{"stage": "poc", "dataset": "iiot-temperature"}`, useful for
    filtering/auditing models in the registry.

## Why this design

- **Logging instead of no-op'ing** when offline: the log line shows the *exact*
  parameters that would be sent to Vertex AI, so switching to
  `USE_VERTEX_AI=True` is a config change, not a code change — and reviewers can
  see what "going live" would actually do without needing a GCP project.
- **Prebuilt serving container** over a custom one keeps the POC's surface area
  small; a real project might swap this for a custom container if it needs
  custom pre/post-processing at serving time.
