# Pipeline Orchestrator — `pipeline/`

## Purpose
Wires the six main stages together into a single runnable command that
demonstrates the full flow: Data → Training → Evaluation → Model Registry →
Deployment → Monitoring.

## Design

`run_pipeline.py`'s `run_pipeline()` is a straight-line sequence of calls into
each stage's public function, in order:

```python
generate_and_save()                 # data/
launch_training()                   # training/
eval_result = evaluate()            # evaluation/
if not eval_result["passed"]:
    return                          # stop before registry/deploy/monitoring
model = register_model()            # registry/
endpoint = deploy_model(model)      # deployment/
setup_monitoring(endpoint)          # monitoring/
```

It prints a labeled banner before each stage and a final summary (MAE/RMSE/R²)
at the end, plus a reminder that `retraining/retrain.py` is the separate
entrypoint for stage 7 (retraining).

## Why standalone scripts instead of a DAG/Pipelines SDK

This project deliberately wires stages together with plain Python function
calls rather than Vertex AI Pipelines (Kubeflow/TFX DSL — components, a
compiled pipeline spec, `PipelineJob.submit()`, etc.). That keeps:

- the code approachable without needing to understand KFP's component/DSL
  model,
- the "does it work" verification loop fast (`python pipeline/run_pipeline.py`
  with no compile step),
- and the offline/local (`USE_VERTEX_AI=False`) execution mode straightforward,
  since there's no pipeline execution backend involved.

A production version of this project could compile the same stage functions
into a Vertex AI Pipeline (each stage becomes a `@component`) to get
Vertex-native lineage tracking, caching, and a visual DAG in the console — that
would replace `run_pipeline.py`'s manual sequencing with a compiled
`PipelineJob`, without needing to change the stage logic itself.

## Why the evaluation gate short-circuits here (and in retraining)

Both `run_pipeline.py` and `retraining/retrain.py` implement the exact same
"stop if evaluation fails" check independently, rather than sharing a single
gating helper — at this scale (two call sites, ~3 lines each) a shared
abstraction would add more indirection than it saves.
