# Stored image bytes (metadata only; the bytes live in object storage).

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Asset(Base):
    """Immutable: the app role may insert and read but never update, so any change is a new row.

    The database stores the bucket, key and version id, never a URL. A generated asset belongs to
    exactly one generation job and its creative version. An uploaded asset is added only when its
    upload passed inspection (READY), with the upload's id as its own; it has a study and may
    have no creative.
    """

    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("job_id"),
        UniqueConstraint("upload_id"),
        ForeignKeyConstraint(["tenant_id", "creative_id"], ["creatives.tenant_id", "creatives.id"]),
        ForeignKeyConstraint(["tenant_id", "job_id"], ["jobs.tenant_id", "jobs.id"]),
        ForeignKeyConstraint(
            ["job_id", "tenant_id", "creative_id", "creative_version"],
            ["jobs.id", "jobs.tenant_id", "jobs.creative_id", "jobs.creative_version"],
        ),
        ForeignKeyConstraint(["tenant_id", "study_id"], ["studies.tenant_id", "studies.id"]),
        ForeignKeyConstraint(
            ["tenant_id", "upload_id"], ["media_uploads.tenant_id", "media_uploads.id"]
        ),
        CheckConstraint(
            "(origin = 'generated' AND creative_id IS NOT NULL AND creative_version IS NOT NULL"
            " AND job_id IS NOT NULL AND upload_id IS NULL AND study_id IS NULL)"
            " OR (origin = 'uploaded' AND upload_id IS NOT NULL AND upload_id = id"
            " AND study_id IS NOT NULL"
            " AND job_id IS NULL)",
            name="lineage_matches_origin",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    # Required for generated assets (CHECK lineage_matches_origin); optional for uploads.
    creative_id: Mapped[UUID | None]
    creative_version: Mapped[int | None]
    job_id: Mapped[UUID | None]
    # Uploads only. Generated assets take their study from their creative (readers COALESCE the
    # two in app.services.assets); setting it would need the worker to read creatives.
    study_id: Mapped[UUID | None]
    upload_id: Mapped[UUID | None]
    # Display name (an upload's file name); never in object keys, tags or logs.
    name: Mapped[str | None] = mapped_column(String(255))
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
