import os

from database import database_url


def test_render_postgres_url_uses_psycopg3(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://user:password@host.example.com/database",
    )
    assert database_url() == (
        "postgresql+psycopg2://user:password@host.example.com/database"
    )


def test_sqlite_url_is_unchanged(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///visionguard.db")
    assert database_url() == "sqlite:///visionguard.db"
