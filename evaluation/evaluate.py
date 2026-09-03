"""Stage 3 - Evaluation: scores the trained model on the holdout test set
and gates progression to registry/deployment based on a MAE threshold.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate(model_dir: str = config.MODEL_DIR, test_data_path: str = None) -> dict:
    test_data_path = test_data_path or os.path.join(config.DATA_DIR, "test.csv")

    with tf.io.gfile.GFile(test_data_path, "r") as f:
        df = pd.read_csv(f)
    with tf.io.gfile.GFile(os.path.join(model_dir, "normalization.json"), "r") as f:
        norm = json.load(f)

    x = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_true = df[config.TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_norm = (x - np.array(norm["mean"])) / np.array(norm["std"])

    model = tf.keras.models.load_model(os.path.join(model_dir, "saved_model.keras"))
    y_pred = model.predict(x_norm, verbose=0).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    passed = mae <= config.MAX_ACCEPTABLE_MAE

    result = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "threshold_mae": config.MAX_ACCEPTABLE_MAE,
        "passed": passed,
    }

    os.makedirs(os.path.dirname(config.EVAL_RESULT_PATH), exist_ok=True)
    with open(config.EVAL_RESULT_PATH, "w") as f:
        json.dump(result, f)

    verdict = "PASSED" if passed else "FAILED"
    print(f"[evaluation] mae={mae:.3f} rmse={rmse:.3f} r2={r2:.3f} -> {verdict} (threshold={config.MAX_ACCEPTABLE_MAE})")
    return result


if __name__ == "__main__":
    evaluate()
