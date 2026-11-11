"""Track VisionGuard experiments locally or on DagsHub with MLflow.

Required environment variables for a remote DagsHub run:

    MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME=<DagsHub username>
    MLFLOW_TRACKING_PASSWORD=<DagsHub access token>

The values are deliberately not stored in this file. ``--tracking-uri`` may
also be used for a local SQLite backend.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

from train_models import (
    IMAGE_SIZE,
    build_baseline,
    build_transfer,
    callbacks,
    compile_baseline,
    make_dataset,
    make_splits,
    set_seed,
)


EXPERIMENT_NAME = "visionguard"


def git_value(arguments: list[str], fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(arguments, text=True).strip() or fallback
    except (OSError, subprocess.CalledProcessError):
        return fallback


def configure_tracking(tracking_uri: str | None) -> str:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    if uri.startswith("http") and not os.getenv("MLFLOW_TRACKING_USERNAME"):
        raise RuntimeError(
            "Remote MLflow requires MLFLOW_TRACKING_USERNAME and "
            "MLFLOW_TRACKING_PASSWORD in the environment."
        )
    if uri.startswith("http") and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError(
            "MLFLOW_TRACKING_PASSWORD is empty. Set it with "
            "`read -r -s MLFLOW_TRACKING_PASSWORD`, then run "
            "`export MLFLOW_TRACKING_PASSWORD`."
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


class ManualHistoryLogger(tf.keras.callbacks.Callback):
    """Log every Keras epoch explicitly, preserving the manual approach."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        metrics = {
            key: float(value)
            for key, value in (logs or {}).items()
            if value is not None
        }
        mlflow.log_metrics(metrics, step=epoch + 1)


def log_common_params(config: dict[str, Any], run_style: str) -> None:
    mlflow.log_params(
        {
            **config,
            "image_height": IMAGE_SIZE[0],
            "image_width": IMAGE_SIZE[1],
            "seed": config["seed"],
            "run_style": run_style,
            "git_commit": git_value(["git", "rev-parse", "HEAD"]),
            "data_version": git_value(
                ["git", "log", "-1", "--format=%H", "--", "archive"],
                "not_configured",
            ),
        }
    )
    mlflow.set_tags(
        {
            "project": "VisionGuard",
            "git_commit": git_value(["git", "rev-parse", "HEAD"]),
            "dvc_data_version": git_value(
                ["git", "log", "-1", "--format=%H", "--", "archive"],
                "not_configured",
            ),
            "seed": str(config["seed"]),
            "run_style": run_style,
        }
    )


def log_artifacts(output_dir: Path, model: tf.keras.Model, run_name: str) -> None:
    """Attach the required report files and the trained Keras model."""
    for filename in (
        "baseline_cnn_confusion_matrix.png",
        "transfer_mobilenetv2_confusion_matrix.png",
        "roc_curves.png",
        "train_split.csv",
        "validation_split.csv",
        "test_split.csv",
        "requirements",
        "tracking_definitions.txt",
    ):
        path = Path(filename)
        if not path.exists():
            path = output_dir / filename
        if not path.exists():
            path = output_dir.parent / "model_output_smoke" / filename
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="evidence")

    model_path = output_dir / f"{run_name}.keras"
    model.save(model_path)
    mlflow.log_artifact(str(model_path), artifact_path="models")
    summary_path = output_dir / f"{run_name}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))
    mlflow.log_artifact(str(summary_path), artifact_path="evidence")


def evaluate_and_log(
    model: tf.keras.Model,
    validation: tf.data.Dataset,
    test: tf.data.Dataset,
    test_frame: pd.DataFrame,
) -> None:
    validation_labels = np.array(
        [label.numpy() for label in validation.map(lambda images, labels: labels).unbatch()],
        dtype="int32",
    )
    validation_probabilities = model.predict(validation, verbose=0).ravel()
    threshold = 0.5
    candidates = np.linspace(0.1, 0.9, 81)
    # Select the validation threshold with the highest binary F1.
    from sklearn.metrics import f1_score

    threshold = float(
        candidates[
            int(
                np.argmax(
                    [
                        f1_score(
                            validation_labels,
                            validation_probabilities >= candidate,
                            zero_division=0,
                        )
                        for candidate in candidates
                    ]
                )
            )
        ]
    )
    labels = test_frame["label"].to_numpy()
    probabilities = model.predict(test, verbose=0).ravel()
    predictions = probabilities >= threshold
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    mlflow.log_metrics(
        {
            "test_accuracy": accuracy_score(labels, predictions),
            "test_precision": precision_score(labels, predictions, zero_division=0),
            "test_recall": recall_score(labels, predictions, zero_division=0),
            "test_f1": f1_score(labels, predictions, zero_division=0),
            "test_roc_auc": roc_auc_score(labels, probabilities),
            "decision_threshold": threshold,
        }
    )


def train_manual(
    config: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    train = make_dataset(
        splits["train"],
        config["batch_size"],
        training=config["augmentation"],
    )
    validation = make_dataset(splits["validation"], config["batch_size"])
    test = make_dataset(splits["test"], config["batch_size"])
    model = build_baseline()
    compile_baseline(model, config["learning_rate"])
    with mlflow.start_run(run_name=config["name"]):
        log_common_params(config, "manual_callback")
        model.fit(
            train,
            validation_data=validation,
            epochs=config["epochs"],
            callbacks=callbacks(output_dir, f"{config['name']}_checkpoint", 2)
            + [ManualHistoryLogger()],
            verbose=1,
        )
        evaluate_and_log(model, validation, test, splits["test"])
        log_artifacts(output_dir, model, config["name"])


def train_autolog(
    config: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    train = make_dataset(
        splits["train"],
        config["batch_size"],
        training=config["augmentation"],
    )
    validation = make_dataset(splits["validation"], config["batch_size"])
    test = make_dataset(splits["test"], config["batch_size"])
    model = build_baseline()
    compile_baseline(model, config["learning_rate"])
    with mlflow.start_run(run_name=config["name"]):
        mlflow.tensorflow.autolog(
            log_models=False,
            log_input_examples=False,
            disable=False,
        )
        log_common_params(config, "tensorflow_autolog")
        model.fit(
            train,
            validation_data=validation,
            epochs=config["epochs"],
            callbacks=callbacks(output_dir, f"{config['name']}_checkpoint", 2),
            verbose=1,
        )
        evaluate_and_log(model, validation, test, splits["test"])
        log_artifacts(output_dir, model, config["name"])
        mlflow.tensorflow.autolog(disable=True)


def comparison_configs(epochs: int, seed: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "lr_1e-3_augmented",
            "learning_rate": 1e-3,
            "augmentation": True,
            "unfrozen_layers": 0,
            "backbone": "baseline_cnn",
            "epochs": epochs,
            "batch_size": 32,
            "seed": seed,
        },
        {
            "name": "lr_3e-4_augmented",
            "learning_rate": 3e-4,
            "augmentation": True,
            "unfrozen_layers": 0,
            "backbone": "baseline_cnn",
            "epochs": epochs,
            "batch_size": 32,
            "seed": seed,
        },
        {
            "name": "lr_1e-4_augmented",
            "learning_rate": 1e-4,
            "augmentation": True,
            "unfrozen_layers": 0,
            "backbone": "baseline_cnn",
            "epochs": epochs,
            "batch_size": 32,
            "seed": seed,
        },
        {
            "name": "lr_1e-3_no_augmentation",
            "learning_rate": 1e-3,
            "augmentation": False,
            "unfrozen_layers": 0,
            "backbone": "baseline_cnn",
            "epochs": epochs,
            "batch_size": 32,
            "seed": seed,
        },
        {
            "name": "lr_1e-3_batch_16",
            "learning_rate": 1e-3,
            "augmentation": True,
            "unfrozen_layers": 0,
            "backbone": "baseline_cnn",
            "epochs": epochs,
            "batch_size": 16,
            "seed": seed,
        },
        {
            "name": "lr_1e-3_batch_64",
            "learning_rate": 1e-3,
            "augmentation": True,
            "unfrozen_layers": 0,
            "backbone": "baseline_cnn",
            "epochs": epochs,
            "batch_size": 64,
            "seed": seed,
        },
    ]


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=root / "archive")
    parser.add_argument("--output-dir", type=Path, default=root / "model_output")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("manual", "autolog", "comparison"), default="manual")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    uri = configure_tracking(args.tracking_uri)
    (args.output_dir / "tracking_definitions.txt").write_text(
        "Experiment: a named group of related MLflow runs.\n"
        "Run: one execution of a training configuration.\n"
        "Parameter: an input setting chosen before training, such as learning rate.\n"
        "Metric: a measured result, such as validation accuracy or test F1.\n"
        "Artifact: a saved output such as a plot, split CSV, package list or model.\n"
        "Tag: searchable metadata describing a run, such as the seed or Git commit.\n\n"
        "Manual callback approach: logs the metrics and metadata explicitly, so the "
        "logged schema is controlled and easy to explain; anything not explicitly "
        "logged is missed.\n"
        "TensorFlow autologging approach: captures standard Keras parameters, epoch "
        "metrics and model information automatically; custom product metrics, "
        "decision thresholds and business tags still need explicit logging.\n",
        encoding="utf-8",
    )
    set_seed(args.seed)
    splits = make_splits(args.dataset_dir, args.output_dir, args.seed)
    if args.mode == "comparison":
        for config in comparison_configs(args.epochs, args.seed):
            set_seed(args.seed)
            train_manual(config, splits, args.output_dir)
    else:
        config = comparison_configs(args.epochs, args.seed)[0]
        config["name"] = f"{args.mode}_visionguard"
        if args.mode == "manual":
            train_manual(config, splits, args.output_dir)
        else:
            train_autolog(config, splits, args.output_dir)
    print(f"Completed {args.mode} run(s) using {uri}")


if __name__ == "__main__":
    main()
