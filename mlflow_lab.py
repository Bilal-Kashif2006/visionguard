"""Local MLflow lab: compare two learning rates on Fashion-MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf


def train_run(learning_rate: float, output_dir: Path) -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:10000].astype("float32") / 255.0
    y_train = y_train[:10000]
    x_test = x_test[:2000].astype("float32") / 255.0
    y_test = y_test[:2000]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_toy",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    with mlflow.start_run(run_name=f"lr_{learning_rate:g}"):
        mlflow.set_tag("purpose", "Task 3.1 local MLflow tool lab")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", 3)
        mlflow.log_param("training_examples", len(x_train))
        history = model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=3,
            batch_size=128,
            verbose=0,
        )
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        for epoch, (loss, accuracy, val_loss, val_accuracy) in enumerate(
            zip(
                history.history["loss"],
                history.history["accuracy"],
                history.history["val_loss"],
                history.history["val_accuracy"],
            ),
            start=1,
        ):
            mlflow.log_metrics(
                {
                    "loss": loss,
                    "accuracy": accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(history.history["val_accuracy"], label="validation accuracy")
        axis.set(xlabel="epoch", ylabel="accuracy", title=f"Learning rate={learning_rate:g}")
        axis.legend()
        figure.tight_layout()
        plot_path = output_dir / f"learning_curve_{learning_rate:g}.png"
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        mlflow.tensorflow.log_model(model, "model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("mlflow_output"))
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="Local MLflow tracking backend (SQLite is the current supported default).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fashion-mnist-learning-rate-comparison")
    for learning_rate in (1e-3, 1e-4):
        train_run(learning_rate, args.output_dir)
    print(
        "Runs logged. Start the comparison UI with: "
        "mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )


if __name__ == "__main__":
    main()
