"""Register two VisionGuard Keras models and classify one image via champion alias."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from PIL import Image

from train_models import IMAGE_SIZE


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--model-name", default="VisionGuard")
    parser.add_argument(
        "--champion-model",
        type=Path,
        default=root / "model_output" / "transfer_mobilenetv2_final.keras",
    )
    parser.add_argument(
        "--second-model",
        type=Path,
        default=root / "model_output_smoke" / "transfer_mobilenetv2_final.keras",
    )
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    for path in (args.second_model, args.champion_model):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = tf.keras.models.load_model(path)
        with mlflow.start_run(run_name=f"register_{path.stem}") as run:
            model_info = mlflow.tensorflow.log_model(
                model,
                name="model",
                registered_model_name=args.model_name,
            )
            print(f"registered {path} as {model_info.registered_model_version}")

    versions = client.search_model_versions(f"name='{args.model_name}'")
    newest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(args.model_name, "champion", newest.version)
    print(f"{args.model_name}@champion -> version {newest.version}")

    image_path = args.image
    if image_path is None:
        image_path = next((root / "archive").rglob("*"), None)
    if image_path is None:
        raise FileNotFoundError("No image supplied and no image was found under archive/")
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    batch = np.asarray(image, dtype="float32")[None, ...]
    champion = mlflow.tensorflow.load_model(f"models:/{args.model_name}@champion")
    probability = float(champion.predict(batch, verbose=0).ravel()[0])
    # The training mapping is AiArtData=0 and RealArt=1, so the model output
    # is the probability of a real image, not the probability of AI generation.
    threshold = float(os.getenv("MODEL_THRESHOLD", "0.23"))
    label = "real" if probability >= threshold else "AI-generated"
    print(f"image={image_path}")
    print(f"prediction={label} probability_ai_generated={probability:.4f}")


if __name__ == "__main__":
    main()
