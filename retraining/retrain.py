"""Stage 7 - Retraining: manual trigger that closes the MLOps loop.

Regenerates data (optionally with simulated equipment drift), re-trains,
re-evaluates, and only re-registers/redeploys/re-monitors if the new model
passes the evaluation gate.
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


def retrain(drift: float = 0.5):
    print(f"[retraining] triggered manually (simulated drift={drift})")
    generate_and_save(drift=drift)
    launch_training()
    result = evaluate()

    if not result["passed"]:
        print("[retraining] new model FAILED evaluation gate -> not re-registering/redeploying")
        return result

    print("[retraining] new model PASSED evaluation gate -> re-registering/redeploying/re-monitoring")
    model = register_model()
    endpoint = deploy_model(model)
    setup_monitoring(endpoint)
    return result


if __name__ == "__main__":
    retrain()
