"""Flask web service for VisionGuard image classification."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from database import (
    create_database_engine,
    initialize_database,
    recent_predictions,
    save_prediction,
)
from inference import VisionGuardPredictor

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def create_app(
    predictor: VisionGuardPredictor | None = None,
    engine=None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES))
    )
    app.config["MODEL_NAME"] = os.getenv("MODEL_NAME", "VisionGuard")
    predictor = predictor or VisionGuardPredictor()
    engine = engine or create_database_engine()
    initialize_database(engine)
    app.extensions["visionguard_predictor"] = predictor
    app.extensions["visionguard_engine"] = engine

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return render_template("error.html", message="Choose an image to upload."), 400
        filename = secure_filename(upload.filename)
        suffix = Path(filename).suffix.lower().lstrip(".")
        if not filename or suffix not in ALLOWED_EXTENSIONS:
            return render_template("error.html", message="Use a JPG, PNG, or WEBP image."), 400
        image_bytes = upload.read()
        try:
            result = predictor.predict(image_bytes)
        except (OSError, ValueError):
            return render_template("error.html", message="The uploaded file is not a readable image."), 400
        save_prediction(engine, filename, result, app.config["MODEL_NAME"])
        return render_template("result.html", filename=filename, result=result)

    @app.get("/history")
    def history():
        return render_template("history.html", predictions=recent_predictions(engine))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": str(predictor.model_path)})

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return render_template("error.html", message="The image must be smaller than 5 MB."), 413

    @app.errorhandler(500)
    def internal_error(_error):
        return render_template("error.html", message="The service could not complete that request."), 500

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
