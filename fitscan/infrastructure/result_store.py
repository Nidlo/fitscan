"""Avatar persistence layer. SQLAlchemy models for avatars and API keys."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base

from fitscan.domain.models import AvatarExport, BodyMeasurements

Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fitscan.db")

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=_connect_args)


class AvatarRecord(Base):
    """Persistent avatar record. One row per avatar."""

    __tablename__ = "avatars"

    avatar_id = Column(String(36), primary_key=True)
    gender_profile = Column(String(20), nullable=False)
    height_cm = Column(Float, nullable=True)
    measurements = Column(Text, nullable=False)
    betas = Column(Text, nullable=False)
    thetas = Column(Text, nullable=False)
    glb_path = Column(String(512), nullable=False)
    obj_path = Column(String(512), nullable=True)
    thumbnail_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))


class APIKeyRecord(Base):
    """SaaS API key record."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_hash = Column(String(64), nullable=False, unique=True)
    owner_name = Column(String(200), nullable=False)
    owner_email = Column(String(200), nullable=False)
    tier = Column(String(20), nullable=False, default="free")
    monthly_limit = Column(Integer, nullable=False, default=50)
    usage_this_month = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime, nullable=True)


# Create tables on import (for SQLite dev mode)
Base.metadata.create_all(engine)


class AvatarStore:
    """Infrastructure: CRUD for AvatarRecord."""

    def save_avatar(self, avatar: AvatarExport) -> str:
        with Session(engine) as session:
            record = AvatarRecord(
                avatar_id=avatar.avatar_id,
                gender_profile=avatar.gender_profile,
                height_cm=avatar.height_cm,
                measurements=json.dumps(avatar.measurements.to_dict()),
                betas=json.dumps(avatar.betas),
                thetas=json.dumps(avatar.thetas),
                glb_path=avatar.glb_path,
                obj_path=avatar.obj_path,
                thumbnail_path=avatar.thumbnail_path,
            )
            session.merge(record)
            session.commit()
        return avatar.avatar_id

    def get_avatar(self, avatar_id: str) -> AvatarExport | None:
        with Session(engine) as session:
            record = session.get(AvatarRecord, avatar_id)
            if not record:
                return None
            return self._record_to_domain(record)

    def update_avatar(self, avatar_id: str, new_avatar: AvatarExport) -> None:
        with Session(engine) as session:
            record = session.get(AvatarRecord, avatar_id)
            if not record:
                raise KeyError(f"Avatar {avatar_id} not found")
            record.betas = json.dumps(new_avatar.betas)
            record.glb_path = new_avatar.glb_path
            record.measurements = json.dumps(new_avatar.measurements.to_dict())
            session.commit()

    def list_avatars(self, limit: int = 50) -> list[str]:
        with Session(engine) as session:
            records = session.query(AvatarRecord.avatar_id).limit(limit).all()
            return [r[0] for r in records]

    def delete(self, avatar_id: str) -> bool:
        """Hard-delete an avatar: DB row + parent directory on disk.

        Returns True on success, False if the avatar didn't exist. PII —
        body scans + 3D meshes are sensitive — is removed irreversibly. The
        right-to-be-forgotten path the platform exposes upstream calls this.
        """
        with Session(engine) as session:
            record = session.get(AvatarRecord, avatar_id)
            if record is None:
                return False
            paths_to_clean = {
                Path(record.glb_path).parent if record.glb_path else None,
                Path(record.obj_path).parent if record.obj_path else None,
                Path(record.thumbnail_path).parent if record.thumbnail_path else None,
            }
            session.delete(record)
            session.commit()

        for parent in paths_to_clean:
            if parent is None or not parent.exists():
                continue
            # Only remove the avatar's own directory — guard against the
            # caller storing into a shared parent by accident.
            if avatar_id in parent.parts:
                shutil.rmtree(parent, ignore_errors=True)
        return True

    def delete_older_than(self, days: int) -> int:
        """Delete every avatar whose created_at is older than `days`.

        Used by the retention task to enforce the 30-day default policy on
        body scans. Returns how many avatars were removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with Session(engine) as session:
            stale_ids = [
                row[0]
                for row in session.query(AvatarRecord.avatar_id)
                .filter(AvatarRecord.created_at < cutoff)
                .all()
            ]

        removed = 0
        for avatar_id in stale_ids:
            if self.delete(avatar_id):
                removed += 1
        if removed:
            logger.info(f"retention: deleted {removed} avatars older than {days}d")
        return removed

    def _record_to_domain(self, record: AvatarRecord) -> AvatarExport:
        return AvatarExport(
            avatar_id=record.avatar_id,
            glb_path=record.glb_path,
            obj_path=record.obj_path or "",
            thumbnail_path=record.thumbnail_path or "",
            betas=json.loads(record.betas),
            thetas=json.loads(record.thetas),
            measurements=BodyMeasurements(source="stored"),
            gender_profile=record.gender_profile,
            height_cm=record.height_cm or 0.0,
            created_at=str(record.created_at),
        )
