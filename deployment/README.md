# Deployment Stage — `deployment/`

## Purpose
Creates a Vertex AI `Endpoint` and deploys the registered model to it for
online prediction.

## Design

`deploy_endpoint.py`'s `deploy_model(model)`:

- When `config.USE_VERTEX_AI` is `False` (default): logs the
  `aiplatform.Endpoint.create(...)` + `model.deploy(...)` calls it would make
  and returns a local stub endpoint object.
- When `config.USE_VERTEX_AI` is `True`:
  1. `aiplatform.Endpoint.create(display_name=config.ENDPOINT_DISPLAY_NAME)`
  2. `model.deploy(endpoint=endpoint, machine_type=config.DEPLOY_MACHINE_TYPE, min_replica_count=1, max_replica_count=1)`

  `machine_type` (`n1-standard-2`) and replica counts are deliberately small/fixed
  — this is a demo endpoint, not a production autoscaling deployment.

`model` is whatever [`registry/register_model.py`](../registry/register_model.py)
returned (a real `aiplatform.Model` or the local stub), so this stage composes
directly with the registry stage — see how [`pipeline/run_pipeline.py`](../pipeline/run_pipeline.py)
chains `register_model()` → `deploy_model(model)`.

## Why this design

- Kept as a thin, single-purpose stage so it's obvious in the pipeline
  narrative where "registry" ends and "deployment" begins — in Vertex AI these
  are genuinely separate resources/API calls (a `Model` can exist in the
  registry without ever being deployed).
- Fixed `min_replica_count=max_replica_count=1`: avoids accidentally
  autoscaling/costing more than necessary for a demo; a real deployment would
  tune this per expected traffic.
