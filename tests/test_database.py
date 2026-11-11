from database import database_url


def test_render_postgres_url_uses_pg8000(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://dbhost.example.com/database")
    assert database_url() == "postgresql+pg8000://dbhost.example.com/database"


def test_sqlite_url_is_unchanged(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///visionguard.db")
    assert database_url() == "sqlite:///visionguard.db"
