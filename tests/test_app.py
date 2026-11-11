from dataclasses import dataclass

from app import create_app
from sqlalchemy import create_engine


@dataclass
class FakePredictor:
    model_path: str = "fake-model.keras"

    def predict(self, image_bytes: bytes):
        assert image_bytes.startswith(b"\x89PNG")
        return {"label": "AI-generated", "confidence": 0.8, "probability_real": 0.2}


def test_health_and_prediction_history(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'visionguard.db'}")
    app = create_app(FakePredictor(), engine=engine)
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.get("/health").status_code == 200
    response = client.post(
        "/predict",
        data={"image": ( __import__("io").BytesIO(b"\x89PNG\r\n\x1a\n"), "unsafe name.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"AI-generated" in response.data
    assert b"unsafe_name.png" in client.get("/history").data


def test_rejects_bad_extension():
    app = create_app(FakePredictor())
    response = app.test_client().post(
        "/predict",
        data={"image": ( __import__("io").BytesIO(b"not an image"), "file.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
