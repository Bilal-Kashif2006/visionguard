# VisionGuard

VisionGuard classifies an uploaded image as **AI-generated** or **real** using
the trained MobileNetV2 model. The model output is the probability of the
`RealArt` class; the validation-selected threshold is configurable through
`MODEL_THRESHOLD`.

## Run locally

Use the existing `.assgn2` environment, install the base requirements and web
requirements, then start the development server with debug mode disabled:

```bash
./.assgn2/bin/pip install -r requirements -r requirements-web.txt
MODEL_PATH=model_output_smoke/transfer_mobilenetv2_final.keras \
  ./.assgn2/bin/python app.py
```

Open `http://localhost:5000`. `GET /health` reports service and model
readiness. Uploads are validated in memory and are not written to disk.
`GET /history` shows the database trail.

## Docker Compose

The model is deliberately mounted read-only instead of copied into the image:
trained artifacts are too large for normal Git source distribution and can be
replaced without rebuilding the application image.

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD in .env
docker compose up --build
```

The web container waits for PostgreSQL readiness. PostgreSQL has no published
host port; the web service reaches it by the Compose service name `database`.

## Existing ML work

`data_audit.py` and `visual_eda.py` implement dataset auditing and visual EDA.
`train_models.py` implements deterministic splits, augmentation, a baseline
CNN, MobileNetV2 transfer learning, validation threshold selection, evaluation,
error analysis, and Keras Tuner searches. `mlflow_lab.py`,
`mlflow_visionguard.py`, and `register_champion.py` cover local/remote
experiment tracking and model registration. Remote MLflow credentials are
read only from environment variables.

For Render, use `requirements-render.txt` instead of the full training
requirements. It contains only the Flask, database, inference, and Gunicorn
runtime dependencies.

If Render exits with a native TensorFlow error such as `free(): invalid
pointer`, temporarily use `render_diagnostics:application` as the start
target. It prints markers before TensorFlow import, application import, and
model loading so the failing phase is visible in Render logs.
