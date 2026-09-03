"""Stage 6 - Monitoring: configures skew/drift detection on the deployed endpoint.

When USE_VERTEX_AI is False, logs the call that would be made instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_monitoring(endpoint):
    if not config.USE_VERTEX_AI:
        print(
            "[monitoring] USE_VERTEX_AI=False -> would call "
            "aiplatform.ModelDeploymentMonitoringJob.create("
            f"display_name='{config.MONITORING_JOB_DISPLAY_NAME}', endpoint=<endpoint>, "
            f"features={config.FEATURE_COLUMNS}, drift_threshold={config.DRIFT_THRESHOLD}, "
            f"sample_rate={config.MONITORING_SAMPLE_RATE}, "
            f"alert_email='{config.MONITORING_ALERT_EMAIL}')"
        )
        return {"resource_name": "local-stub-monitoring-job"}

    from google.cloud.aiplatform import model_monitoring

    alert_config = model_monitoring.EmailAlertConfig(user_emails=[config.MONITORING_ALERT_EMAIL])
    drift_config = model_monitoring.DriftDetectionConfig(
        drift_thresholds={feature: config.DRIFT_THRESHOLD for feature in config.FEATURE_COLUMNS}
    )
    objective_config = model_monitoring.ObjectiveConfig(drift_detection_config=drift_config)

    from google.cloud import aiplatform

    job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=config.MONITORING_JOB_DISPLAY_NAME,
        endpoint=endpoint,
        logging_sampling_strategy=aiplatform.model_monitoring.RandomSampleConfig(
            sample_rate=config.MONITORING_SAMPLE_RATE
        ),
        objective_configs=objective_config,
        alert_config=alert_config,
    )
    print(f"[monitoring] created monitoring job: {job.resource_name}")
    return job


if __name__ == "__main__":
    from deployment.deploy_endpoint import deploy_model
    from registry.register_model import register_model

    setup_monitoring(deploy_model(register_model()))
