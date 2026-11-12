"""Model loading and inference for the VisionGuard web service."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = (224, 224)
DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "model_output_smoke"
    / "transfer_mobilenetv2_final.keras"
)


class VisionGuardPredictor:
    """Load a Keras model once and expose deterministic image predictions."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float | None = None,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL))
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.threshold = float(
            threshold if threshold is not None else os.getenv("MODEL_THRESHOLD", "0.23")
        )
        if not 0 < self.threshold < 1:
            raise ValueError("MODEL_THRESHOLD must be between 0 and 1.")
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, image_bytes: bytes) -> dict[str, float | str]:
        """Return the product verdict and confidence for an uploaded image."""
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            batch = np.asarray(image, dtype="float32")[None, ...]
        # Use a direct eager forward pass instead of model.predict(), whose
        # input pipeline and callback machinery add unnecessary overhead here.
        probability_real = float(
            np.asarray(self.model(batch, training=False)).ravel()[0]
        )
        is_real = probability_real >= self.threshold
        confidence = probability_real if is_real else 1.0 - probability_real
        return {
            "label": "real" if is_real else "AI-generated",
            "confidence": confidence,
            "probability_real": probability_real,
        }
