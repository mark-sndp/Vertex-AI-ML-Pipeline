"""Stage 2 - Training launcher: submits training as a Vertex AI CustomJob.

When USE_VERTEX_AI is False (default), it just runs training/task.py locally
in-process so the whole pipeline can be exercised without any GCP calls.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training import task as training_task


def run_local(train_data_path: str, output_dir: str) -> dict:
    print("[train_launcher] USE_VERTEX_AI=False -> running training/task.py locally")
    return training_task.train(
        train_data_path, output_dir, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE
    )


def run_on_vertex(train_data_path: str, output_dir: str) -> dict:
    from google.cloud import aiplatform

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION, staging_bucket=f"gs://{config.GCS_BUCKET}")

    job = aiplatform.CustomJob.from_local_script(
        display_name="iiot-temperature-training",
        script_path=os.path.join(os.path.dirname(__file__), "task.py"),
        container_uri=config.TRAINING_CONTAINER_IMAGE,
        args=[
            f"--train-data={train_data_path}",
            f"--output-dir={output_dir}",
            f"--epochs={config.EPOCHS}",
            f"--learning-rate={config.LEARNING_RATE}",
            f"--batch-size={config.BATCH_SIZE}",
        ],
        requirements=["pandas", "numpy"],
    )
    print(f"[train_launcher] submitting CustomJob to Vertex AI (project={config.PROJECT_ID}, region={config.REGION})")
    job.run(sync=True)
    print(f"[train_launcher] CustomJob finished: {job.resource_name}")

    # Metrics are written by task.py directly to the shared artifacts path.
    import json

    with open(config.METRICS_PATH) as f:
        return json.load(f)


def launch_training(train_data_path: str = None, output_dir: str = config.MODEL_DIR) -> dict:
    train_data_path = train_data_path or os.path.join(config.DATA_DIR, "train.csv")
    if config.USE_VERTEX_AI:
        return run_on_vertex(train_data_path, output_dir)
    return run_local(train_data_path, output_dir)


if __name__ == "__main__":
    launch_training()
