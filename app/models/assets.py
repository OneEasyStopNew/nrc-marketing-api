# Stored image bytes (metadata only; the bytes live in object storage).

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, DateTime, ForeignKeyConstraint, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Asset(Base):
    """Immutable: the app role may insert and read but never update, so any change is a new row.

    The database stores the bucket, key and version id, never a URL. One asset per generation job.
    """

    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("job_id"),
        ForeignKeyConstraint(["tenant_id", "creative_id"], ["creatives.tenant_id", "creatives.id"]),
        ForeignKeyConstraint(["tenant_id", "job_id"], ["jobs.tenant_id", "jobs.id"]),
        ForeignKeyConstraint(
            ["job_id", "tenant_id", "creative_id", "creative_version"],
            ["jobs.id", "jobs.tenant_id", "jobs.creative_id", "jobs.creative_version"],
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    creative_id: Mapped[UUID]
    creative_version: Mapped[int]
    job_id: Mapped[UUID]
    # An AssetOrigin value (app.schemas.assets).
    origin: Mapped[str] = mapped_column(String(16))
    bucket: Mapped[str] = mapped_column(String(63))
    s3_key: Mapped[str] = mapped_column(String(1024))
    s3_version_id: Mapped[str | None] = mapped_column(String(1024))
    mime_type: Mapped[str] = mapped_column(String(64))
    width_px: Mapped[int]
    height_px: Mapped[int]
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(100))
    provider_request_id: Mapped[str | None] = mapped_column(String(200))
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
