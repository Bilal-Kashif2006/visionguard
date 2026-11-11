"""Prediction history persistence for SQLite and PostgreSQL deployments."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probability_real: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    # Render supplies PostgreSQL URLs without a SQLAlchemy driver name.
    # The deployment uses psycopg 3, so select its dialect explicitly.
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def create_database_engine():
    return create_engine(database_url(), pool_pre_ping=True)


def initialize_database(engine) -> None:
    Base.metadata.create_all(engine)


def save_prediction(engine, filename: str, result: dict[str, float | str], model_name: str) -> None:
    with Session(engine) as session:
        session.add(
            Prediction(
                filename=filename,
                label=str(result["label"]),
                confidence=float(result["confidence"]),
                probability_real=float(result["probability_real"]),
                model_name=model_name,
            )
        )
        session.commit()


def recent_predictions(engine, limit: int = 50) -> list[Prediction]:
    with Session(engine) as session:
        return list(
            session.scalars(
                select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
            )
        )
