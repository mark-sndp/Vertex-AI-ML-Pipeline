"""Stage 5 - Deployment: creates a Vertex AI Endpoint and deploys the model.

When USE_VERTEX_AI is False, logs the calls that would be made instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def deploy_model(model):
    if not config.USE_VERTEX_AI:
        print(
            f"[deployment] USE_VERTEX_AI=False -> would call aiplatform.Endpoint.create("
            f"display_name='{config.ENDPOINT_DISPLAY_NAME}') and "
            f"model.deploy(machine_type='{config.DEPLOY_MACHINE_TYPE}', min_replica_count=1)"
        )
        return {"resource_name": "local-stub-endpoint", "display_name": config.ENDPOINT_DISPLAY_NAME}

    from google.cloud import aiplatform

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
    endpoint = aiplatform.Endpoint.create(display_name=config.ENDPOINT_DISPLAY_NAME)
    model.deploy(
        endpoint=endpoint,
        machine_type=config.DEPLOY_MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
    )
    print(f"[deployment] deployed model to endpoint: {endpoint.resource_name}")
    return endpoint


if __name__ == "__main__":
    from registry.register_model import register_model

    deploy_model(register_model())
