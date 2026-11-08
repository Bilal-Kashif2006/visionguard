"""Train, tune and evaluate the two image classifiers required by the assignment.

The default dataset is ``./archive`` and all report evidence is written to
``./model_output``.  A short CPU smoke run is available with ``--epochs 1``;
the default settings are intended for the final experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data_audit import audit_dataset


SEED = 42
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def make_splits(dataset_dir: Path, output_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    """Audit, remove exact duplicate copies, and make stratified 70/15/15 splits."""
    audit_path = output_dir / "image_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
    else:
        audit, corrupt = audit_dataset(dataset_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)
        corrupt.to_csv(output_dir / "corrupt_files.csv", index=False)

    # Keep one representative for exact duplicates to avoid train/test leakage.
    audit = audit.sort_values("filepath").drop_duplicates("sha256", keep="first")
    labels = sorted(audit["class"].unique())
    if len(labels) != 2:
        raise ValueError(f"Expected two classes, found {labels}")
    class_to_id = {name: index for index, name in enumerate(labels)}
    audit = audit.assign(label=audit["class"].map(class_to_id).astype("int32"))

    train, remainder = train_test_split(
        audit, test_size=0.30, stratify=audit["label"], random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=remainder["label"], random_state=seed
    )
    splits = {"train": train, "validation": validation, "test": test}
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}_split.csv", index=False)
    (output_dir / "class_mapping.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8"
    )
    return splits


def decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    return tf.cast(image, tf.float32), label


def make_dataset(
    split: pd.DataFrame,
    batch_size: int,
    training: bool = False,
    preprocess: callable | None = None,
) -> tf.data.Dataset:
    """Create read -> decode -> resize -> optional augment -> batch -> prefetch."""
    paths = split["filepath"].astype(str).to_numpy()
    labels = split["label"].astype("float32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(split), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_resize, num_parallel_calls=AUTOTUNE)
    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.15),
                tf.keras.layers.RandomContrast(0.10),
            ],
            name="training_augmentation",
        )
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=AUTOTUNE,
        )
    if preprocess is not None:
        dataset = dataset.map(
            lambda image, label: (preprocess(image), label),
            num_parallel_calls=AUTOTUNE,
        )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def save_augmented_batch(dataset: tf.data.Dataset, output_path: Path) -> None:
    images, labels = next(iter(dataset))
    images = tf.clip_by_value(images / 255.0, 0, 1)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, image, label in zip(axes.flat, images[:8], labels[:8]):
        axis.imshow(image)
        axis.set_title(f"label={int(label.numpy())}")
        axis.axis("off")
    figure.suptitle("Augmented training batch (224 x 224 RGB)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_baseline() -> tf.keras.Model:
    inputs = tf.keras.Input((*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1 / 255.0)(inputs)
    for filters, dropout in [(32, 0.10), (64, 0.15), (128, 0.25)]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(
    learning_rate: float = 1e-3,
    unfrozen_layers: int = 0,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
    except Exception as error:
        raise RuntimeError(
            "MobileNetV2 ImageNet weights could not be loaded. "
            "Connect to the internet once or use the provided --allow-random-backbone "
            "only for a smoke test."
        ) from error
    base.trainable = unfrozen_layers > 0
    for layer in base.layers:
        layer.trainable = False
    if unfrozen_layers:
        for layer in base.layers[-unfrozen_layers:]:
            layer.trainable = True
    inputs = tf.keras.Input((*IMAGE_SIZE, 3), name="image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30, name="transfer_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model, base


def callbacks(output_dir: Path, name: str, patience: int) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / f"{name}.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.25, patience=max(1, patience // 2), min_lr=1e-7
        ),
    ]


def compile_baseline(model: tf.keras.Model, learning_rate: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )


def save_history(history: tf.keras.callbacks.History, path: Path, title: str) -> int:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(path.with_suffix(".csv"), index=False)
    best_epoch = int(history_df["val_loss"].idxmin()) + 1
    epochs = np.arange(1, len(history_df) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history_df["loss"], marker="o", label="train")
    axes[0].plot(epochs, history_df["val_loss"], marker="o", label="validation")
    axes[0].axvline(
        best_epoch, color="red", linestyle="--", label=f"best epoch={best_epoch}"
    )
    axes[0].set_title(f"{title}: loss")
    axes[1].plot(epochs, history_df["accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history_df["val_accuracy"], marker="o", label="validation")
    axes[1].axvline(best_epoch, color="red", linestyle="--", label=f"best epoch={best_epoch}")
    axes[1].set_title(f"{title}: accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(figure)
    return best_epoch


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.10, 0.90, 81):
        predictions = probabilities >= threshold
        rows.append(
            {
                "threshold": threshold,
                "f1": f1_score(labels, predictions, zero_division=0),
                "accuracy": accuracy_score(labels, predictions),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def evaluate_model(
    model: tf.keras.Model,
    validation: tf.data.Dataset,
    test: tf.data.Dataset,
    test_frame: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> dict[str, float]:
    validation_labels = validation.map(lambda images, labels: labels).unbatch()
    val_labels = np.array([label.numpy() for label in validation_labels], dtype="int32")
    val_probabilities = model.predict(validation, verbose=0).ravel()
    threshold, threshold_table = choose_threshold(val_labels, val_probabilities)
    threshold_table.to_csv(output_dir / f"{name}_validation_thresholds.csv", index=False)

    test_labels = test_frame["label"].to_numpy()
    probabilities = model.predict(test, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype("int32")
    metrics = {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, zero_division=0),
        "recall": recall_score(test_labels, predictions, zero_division=0),
        "f1": f1_score(test_labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, probabilities),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / f"{name}_metrics.csv", index=False)
    np.save(output_dir / f"{name}_test_probabilities.npy", probabilities)
    np.save(output_dir / f"{name}_test_predictions.npy", predictions)

    matrix = confusion_matrix(test_labels, predictions)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["real", "AI-generated"],
        yticklabels=["real", "AI-generated"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{name} confusion matrix (threshold={threshold:.2f})",
    )
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, value, ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(figure)
    return metrics


def save_roc_curve(results: list[dict[str, float]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for result in results:
        probabilities = np.load(output_dir / f"{result['model']}_test_probabilities.npy")
        labels = pd.read_csv(output_dir / "test_split.csv")["label"].to_numpy()
        fpr, tpr, _ = roc_curve(labels, probabilities)
        axis.plot(fpr, tpr, label=f"{result['model']} AUC={result['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], "k--", label="chance")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close(figure)


def save_error_grid(
    model: tf.keras.Model,
    test: tf.data.Dataset,
    test_frame: pd.DataFrame,
    threshold: float,
    output_path: Path,
    name: str,
) -> None:
    probabilities = model.predict(test, verbose=0).ravel()
    labels = test_frame["label"].to_numpy()
    wrong = np.flatnonzero((probabilities >= threshold).astype("int32") != labels)[:10]
    figure, axes = plt.subplots(2, 5, figsize=(15, 7))
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, wrong):
        try:
            with Image.open(test_frame.iloc[index]["filepath"]) as image:
                axis.imshow(image.convert("RGB"))
            predicted = "AI-generated" if probabilities[index] >= threshold else "real"
            actual = "AI-generated" if labels[index] else "real"
            axis.set_title(f"actual={actual}\npredicted={predicted}\n"
                           f"reason to inspect: visual ambiguity")
        except (OSError, UnidentifiedImageError):
            continue
        axis.axis("off")
    figure.suptitle(f"{name}: first 10 misclassified test images")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_tuner(
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    output_dir: Path,
    trials: int,
    seed: int,
) -> None:
    """Run the required >=8-trial VisionGuard hyperparameter search."""
    try:
        import keras_tuner as kt
    except ImportError as error:
        raise RuntimeError(
            "Keras Tuner is required. Install it with: pip install keras-tuner"
        ) from error

    def build_tuned(hp: kt.HyperParameters) -> tf.keras.Model:
        learning_rate = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        dropout = hp.Float("dropout", 0.10, 0.50, step=0.10)
        unfrozen = hp.Int("unfrozen_layers", 0, 30, step=10)
        model, _ = build_transfer(learning_rate, unfrozen)
        model.get_layer("transfer_dropout").rate = dropout
        return model

    tuner = kt.RandomSearch(
        build_tuned,
        objective="val_accuracy",
        max_trials=max(8, trials),
        directory=output_dir / "keras_tuner",
        project_name="visionguard",
        overwrite=True,
        seed=seed,
    )
    tuner.search(train, validation_data=validation, epochs=2, verbose=1)
    tuner.results_summary(num_trials=max(8, trials))
    best = tuner.get_best_hyperparameters(1)[0]
    (output_dir / "visionguard_best_hyperparameters.json").write_text(
        json.dumps(best.values, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "trial": trial.trial_id,
                **trial.hyperparameters.values,
                "best_val_accuracy": trial.metrics.get_best_value("val_accuracy"),
            }
            for trial in tuner.oracle.trials.values()
        ]
    ).to_csv(output_dir / "visionguard_tuner_trials.csv", index=False)


def run_toy_tuner(output_dir: Path, trials: int, seed: int) -> None:
    """Tune learning rate and units on a small Fashion-MNIST toy problem."""
    try:
        import keras_tuner as kt
    except ImportError as error:
        raise RuntimeError(
            "Keras Tuner is required. Install it with: pip install keras-tuner"
        ) from error

    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:10000].astype("float32") / 255.0
    y_train = y_train[:10000]

    def build_toy(hp: kt.HyperParameters) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    hp.Int("units", 16, 96, step=16), activation="relu"
                ),
                tf.keras.layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="fashion_mnist_tuned_toy",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.RandomSearch(
        build_toy,
        objective="val_accuracy",
        max_trials=max(8, trials),
        directory=output_dir / "keras_tuner",
        project_name="fashion_mnist_toy",
        overwrite=True,
        seed=seed,
    )
    tuner.search(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=2,
        batch_size=128,
        verbose=1,
    )
    best = tuner.get_best_hyperparameters(1)[0]
    (output_dir / "toy_tuner_best_hyperparameters.json").write_text(
        json.dumps(best.values, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "trial": trial.trial_id,
                **trial.hyperparameters.values,
                "best_val_accuracy": trial.metrics.get_best_value("val_accuracy"),
            }
            for trial in tuner.oracle.trials.values()
        ]
    ).to_csv(output_dir / "toy_tuner_trials.csv", index=False)


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits(Path(args.dataset_dir), output_dir, args.seed)
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    baseline_train = make_dataset(splits["train"], args.batch_size, training=True)
    baseline_validation = make_dataset(splits["validation"], args.batch_size)
    baseline_test = make_dataset(splits["test"], args.batch_size)
    transfer_train = make_dataset(
        splits["train"], args.batch_size, training=True, preprocess=preprocess
    )
    transfer_validation = make_dataset(
        splits["validation"], args.batch_size, preprocess=preprocess
    )
    transfer_test = make_dataset(splits["test"], args.batch_size, preprocess=preprocess)
    save_augmented_batch(baseline_train, output_dir / "augmented_training_batch.png")

    baseline = build_baseline()
    compile_baseline(baseline)
    baseline_history = baseline.fit(
        baseline_train,
        validation_data=baseline_validation,
        epochs=args.epochs,
        callbacks=callbacks(output_dir, "baseline_cnn_best", args.patience),
        verbose=1,
    )
    baseline_best_epoch = save_history(
        baseline_history, output_dir / "baseline_cnn_curves", "Baseline CNN"
    )
    baseline_metrics = evaluate_model(
        baseline,
        baseline_validation,
        baseline_test,
        splits["test"],
        output_dir,
        "baseline_cnn",
    )
    baseline.save(output_dir / "baseline_cnn_final.keras")
    with (output_dir / "baseline_cnn_summary.txt").open("w", encoding="utf-8") as file:
        baseline.summary(print_fn=lambda line: file.write(line + "\n"))

    transfer, base = build_transfer(1e-3, 0)
    stage1_trainable_layers = [layer.name for layer in transfer.layers if layer.trainable]
    transfer_stage1 = transfer.fit(
        transfer_train,
        validation_data=transfer_validation,
        epochs=args.epochs,
        callbacks=callbacks(output_dir, "transfer_stage1_best", args.patience),
        verbose=1,
    )
    stage1_best_epoch = save_history(
        transfer_stage1, output_dir / "transfer_stage1_curves", "Transfer learning (frozen)"
    )
    base.trainable = True
    for layer in base.layers[:-args.unfrozen_layers]:
        layer.trainable = False
    transfer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    transfer_stage2 = transfer.fit(
        transfer_train,
        validation_data=transfer_validation,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks(output_dir, "transfer_stage2_best", args.patience),
        verbose=1,
    )
    stage2_best_epoch = save_history(
        transfer_stage2, output_dir / "transfer_stage2_curves", "Transfer learning (fine-tuned)"
    )
    transfer_metrics = evaluate_model(
        transfer,
        transfer_validation,
        transfer_test,
        splits["test"],
        output_dir,
        "transfer_mobilenetv2",
    )
    transfer.save(output_dir / "transfer_mobilenetv2_final.keras")
    trainable_layers = pd.DataFrame(
        {
            "stage": ["stage1_frozen", "stage2_last_block_unfrozen"],
            "trainable_layers": [
            ", ".join(stage1_trainable_layers) or "none",
            ", ".join(layer.name for layer in transfer.layers if layer.trainable),
            ],
        }
    )
    trainable_layers.to_csv(output_dir / "transfer_trainable_layers.csv", index=False)
    results = [baseline_metrics, transfer_metrics]
    pd.DataFrame(results).to_csv(output_dir / "model_metrics.csv", index=False)
    save_roc_curve(results, output_dir)
    save_error_grid(
        transfer,
        transfer_test,
        splits["test"],
        transfer_metrics["threshold"],
        output_dir / "transfer_error_analysis_grid.png",
        "MobileNetV2",
    )
    pd.DataFrame(
        [
            {"model": "baseline_cnn", "best_epoch": baseline_best_epoch},
            {"model": "transfer_stage1", "best_epoch": stage1_best_epoch},
            {"model": "transfer_stage2", "best_epoch": stage2_best_epoch},
        ]
    ).to_csv(output_dir / "best_epochs.csv", index=False)
    (output_dir / "pipeline_description.txt").write_text(
        "Images are read from split CSV paths, decoded to RGB, resized to 224x224 "
        "with aspect ratio not preserved (images are stretched to the fixed model input), "
        "augmented only for training with horizontal flips, small rotations, zoom and contrast, "
        "batched, and prefetched with tf.data.AUTOTUNE. MobileNetV2 additionally uses "
        "keras.applications.mobilenet_v2.preprocess_input; the baseline uses Rescaling(1/255). "
        "A threshold is selected on validation F1 and then applied once to the test set. "
        "For a content moderation product, calling a real photo AI-generated is the worse "
        "error because it can wrongly reject authentic content; precision and confusion "
        "matrices should therefore be considered alongside recall.",
        encoding="utf-8",
    )
    if not args.skip_tuner:
        run_toy_tuner(output_dir, args.tuner_trials, args.seed)
        run_tuner(transfer_train, transfer_validation, output_dir, args.tuner_trials, args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--dataset-dir", type=Path, default=root / "archive")
    parser.add_argument("--output-dir", type=Path, default=root / "model_output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfrozen-layers", type=int, default=30)
    parser.add_argument("--tuner-trials", type=int, default=8)
    parser.add_argument("--skip-tuner", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
