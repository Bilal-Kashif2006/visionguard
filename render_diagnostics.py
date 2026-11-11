"""Staged Render startup diagnostics for the TensorFlow serving process."""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


def stage(name: str) -> None:
    print(f"[visionguard-startup] {name}", flush=True)


stage(f"python={sys.version}")
stage(f"model_path={os.getenv('MODEL_PATH', '<default>')}")
stage("importing tensorflow")
import tensorflow as tf  # noqa: E402

stage(f"tensorflow={tf.__version__}")
stage(f"visible_gpus={tf.config.list_physical_devices('GPU')}")
stage("importing application")
from app import create_app  # noqa: E402

stage("creating Flask application and loading model")
try:
    application = create_app()
except BaseException:
    traceback.print_exc()
    raise
stage("application ready")

