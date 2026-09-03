"""Orchestrates the full IIoT Vertex AI demo pipeline end to end:
Data -> Training -> Evaluation -> Registry -> Deployment -> Monitoring.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.generate_dataset import generate_and_save
from deployment.deploy_endpoint import deploy_model
from evaluation.evaluate import evaluate
from monitoring.setup_monitoring import setup_monitoring
from registry.register_model import register_model
from training.train_launcher import launch_training


def run_pipeline():
    mode = "LIVE (real GCP calls)" if config.USE_VERTEX_AI else "OFFLINE (local simulation)"
    print(f"=== IIoT Vertex AI pipeline demo — mode: {mode} ===\n")

    print("--- Stage 1: Data ---")
    generate_and_save()

    print("\n--- Stage 2: Training ---")
    launch_training()

    print("\n--- Stage 3: Evaluation ---")
    eval_result = evaluate()

    if not eval_result["passed"]:
        print("\n[pipeline] evaluation gate FAILED -> stopping before registry/deployment/monitoring")
        return

    print("\n--- Stage 4: Model Registry ---")
    model = register_model()

    print("\n--- Stage 5: Deployment ---")
    endpoint = deploy_model(model)

    print("\n--- Stage 6: Monitoring ---")
    setup_monitoring(endpoint)

    print("\n=== Pipeline complete ===")
    print(f"MAE={eval_result['mae']:.3f}  RMSE={eval_result['rmse']:.3f}  R2={eval_result['r2']:.3f}")
    print("(Stage 7 - Retraining: run `python retraining/retrain.py` to re-trigger the loop)")


if __name__ == "__main__":
    run_pipeline()
