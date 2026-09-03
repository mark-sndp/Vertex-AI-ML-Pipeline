"""Central configuration for the IIoT Vertex AI pipeline POC.

All values can be overridden via environment variables so the same code can
run fully offline (default) or against a real GCP project.
"""
import os

# --- GCP / Vertex AI ---
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")
REGION = os.environ.get("REGION", "us-central1")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "your-gcs-bucket-name")
GCS_PREFIX = "iiot-demo"

# Master switch: when False, every stage logs the Vertex AI call it *would*
# make and falls back to local execution/storage instead. Flip to True (and
# fill in the values above) to run against a real GCP project.
USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "False").lower() == "true"

# --- Dataset ---
FEATURE_COLUMNS = ["vibration_mm_s", "pressure_bar", "rpm", "load_pct"]
TARGET_COLUMN = "temperature_c"
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "2000"))
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# --- Training ---
EPOCHS = int(os.environ.get("EPOCHS", "150"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
TRAINING_CONTAINER_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-16.py310:latest"
)
SERVING_CONTAINER_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-16:latest"
)

# --- Evaluation gate ---
MAX_ACCEPTABLE_MAE = float(os.environ.get("MAX_ACCEPTABLE_MAE", "3.0"))

# --- Registry / Deployment ---
MODEL_DISPLAY_NAME = "iiot-temperature-regressor"
ENDPOINT_DISPLAY_NAME = "iiot-temperature-endpoint"
DEPLOY_MACHINE_TYPE = "n1-standard-2"

# --- Monitoring ---
MONITORING_JOB_DISPLAY_NAME = "iiot-temperature-monitoring"
MONITORING_SAMPLE_RATE = 0.8
MONITORING_ALERT_EMAIL = os.environ.get("MONITORING_ALERT_EMAIL", "alerts@example.com")
DRIFT_THRESHOLD = 0.3

# --- Local artifact paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
EVAL_RESULT_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_result.json")


def gcs_uri(*parts: str) -> str:
    """Build a gs:// URI under the demo bucket/prefix."""
    return f"gs://{GCS_BUCKET}/{GCS_PREFIX}/" + "/".join(parts)
