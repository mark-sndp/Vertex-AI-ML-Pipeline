"""Stage 4 - Model registry: uploads the trained model to Vertex AI Model Registry.

When USE_VERTEX_AI is False, logs the call that would be made and returns a
stub model reference instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def register_model(model_dir: str = config.MODEL_DIR):
    artifact_uri = model_dir if config.USE_VERTEX_AI else os.path.join(model_dir, "1")

    if not config.USE_VERTEX_AI:
        print(
            "[registry] USE_VERTEX_AI=False -> would call aiplatform.Model.upload("
            f"display_name='{config.MODEL_DISPLAY_NAME}', artifact_uri='{artifact_uri}', "
            f"serving_container_image_uri='{config.SERVING_CONTAINER_IMAGE}')"
        )
        return {"resource_name": "local-stub-model", "display_name": config.MODEL_DISPLAY_NAME}

    from google.cloud import aiplatform

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
    model = aiplatform.Model.upload(
        display_name=config.MODEL_DISPLAY_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri=config.SERVING_CONTAINER_IMAGE,
        labels={"stage": "poc", "dataset": "iiot-temperature"},
    )
    print(f"[registry] registered model: {model.resource_name}")
    return model


if __name__ == "__main__":
    register_model()
