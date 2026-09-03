"""Stage 1 - Data: simulate an IIoT sensor dataset and split train/test.

Simulates readings from a rotating machine (e.g. a pump or motor):
vibration (mm/s), pressure (bar), rpm, and load (%). The target,
temperature (C), is a noisy function of those features.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def simulate_iiot_data(num_samples: int, seed: int = config.RANDOM_SEED, drift: float = 0.0) -> pd.DataFrame:
    """Generate a synthetic IIoT sensor dataset.

    `drift` shifts the load/vibration distribution upward to simulate
    equipment wear, used by the retraining stage to justify a re-run.
    """
    rng = np.random.default_rng(seed)

    vibration = rng.normal(2.5 + drift, 0.6, num_samples).clip(0.1, None)
    pressure = rng.normal(5.0, 0.8, num_samples).clip(0.5, None)
    rpm = rng.normal(1500, 120, num_samples).clip(500, None)
    load_pct = rng.normal(60 + drift * 10, 15, num_samples).clip(0, 100)

    # Ground-truth relationship + noise: higher vibration/load/rpm -> hotter.
    temperature_c = (
        35
        + 4.0 * vibration
        + 0.6 * pressure
        + 0.01 * rpm
        + 0.25 * load_pct
        + rng.normal(0, 1.5, num_samples)
    )

    return pd.DataFrame(
        {
            "vibration_mm_s": vibration,
            "pressure_bar": pressure,
            "rpm": rpm,
            "load_pct": load_pct,
            "temperature_c": temperature_c,
        }
    )


def generate_and_save(output_dir: str = config.DATA_DIR, drift: float = 0.0) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    df = simulate_iiot_data(config.NUM_SAMPLES, drift=drift)
    test_size = int(len(df) * config.TEST_SPLIT)
    test_df = df.iloc[:test_size]
    train_df = df.iloc[test_size:]

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[data] generated {len(df)} rows ({len(train_df)} train / {len(test_df)} test)")
    print(f"[data] saved locally: {train_path}, {test_path}")

    if config.USE_VERTEX_AI:
        from google.cloud import storage

        client = storage.Client(project=config.PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET)
        for local_path, name in [(train_path, "train.csv"), (test_path, "test.csv")]:
            blob = bucket.blob(f"{config.GCS_PREFIX}/data/{name}")
            blob.upload_from_filename(local_path)
        train_path = config.gcs_uri("data", "train.csv")
        test_path = config.gcs_uri("data", "test.csv")
        print(f"[data] uploaded to {train_path}, {test_path}")
    else:
        print(f"[data] USE_VERTEX_AI=False -> skipping upload; would upload to {config.gcs_uri('data', 'train.csv')}")

    return train_path, test_path


if __name__ == "__main__":
    generate_and_save()
