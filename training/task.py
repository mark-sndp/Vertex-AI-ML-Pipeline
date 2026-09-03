"""Stage 2 - Training task: trains a small Keras regression model.

Runnable standalone (python training/task.py --train-data ... --output-dir ...)
or as the entrypoint of a Vertex AI CustomJob training container.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_model(num_features: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def load_dataset(path: str) -> pd.DataFrame:
    # tf.io.gfile transparently handles both local paths and gs:// URIs.
    with tf.io.gfile.GFile(path, "r") as f:
        return pd.read_csv(f)


def train(train_data_path: str, output_dir: str, epochs: int, learning_rate: float, batch_size: int) -> dict:
    df = load_dataset(train_data_path)
    x = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[config.TARGET_COLUMN].to_numpy(dtype=np.float32)

    # Normalize features using train-set statistics only.
    mean, std = x.mean(axis=0), x.std(axis=0)
    x_norm = (x - mean) / std

    model = build_model(num_features=x.shape[1], learning_rate=learning_rate)
    history = model.fit(x_norm, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

    tf.io.gfile.makedirs(output_dir)
    model.save(os.path.join(output_dir, "saved_model.keras"))
    # Also export SavedModel format for Vertex AI prediction container compatibility.
    model.export(os.path.join(output_dir, "1"))

    norm_stats = {"mean": mean.tolist(), "std": std.tolist(), "features": config.FEATURE_COLUMNS}
    with tf.io.gfile.GFile(os.path.join(output_dir, "normalization.json"), "w") as f:
        json.dump(norm_stats, f)

    metrics = {
        "final_train_mae": float(history.history["mae"][-1]),
        "final_val_mae": float(history.history["val_mae"][-1]),
        "epochs": epochs,
    }
    with tf.io.gfile.GFile(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    # Also mirror to the well-known artifacts path used by later stages.
    os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    print(f"[training] done. metrics={metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=os.path.join(config.DATA_DIR, "train.csv"))
    parser.add_argument("--output-dir", default=config.MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    train(args.train_data, args.output_dir, args.epochs, args.learning_rate, args.batch_size)


if __name__ == "__main__":
    main()
