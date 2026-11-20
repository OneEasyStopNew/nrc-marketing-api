# Client uploads while they are checked. A READY upload is copied into an immutable assets row.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import IdentifiablePerson, UploadPurpose, UploadState, enum_column

# media_uploads.reason_message column size; writers truncate to it.
UPLOAD_REASON_MAX_CHARS = 500


class MediaUpload(Base):
    """One client upload, from the issued URL to READY or REJECTED.

    Mutable on purpose (state, lease, reasons), unlike assets. The media worker leases rows due
    for inspection only through the lease_media_uploads() database function, which returns ids;
    everything else it reads runs under the upload's tenant (RLS). The bytes never change after
    completion: the S3 version id is pinned when the upload completes.
    """

    __tablename__ = "media_uploads"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("s3_key"),
        ForeignKeyConstraint(["tenant_id", "study_id"], ["studies.tenant_id", "studies.id"]),
        ForeignKeyConstraint(["tenant_id", "creative_id"], ["creatives.tenant_id", "creatives.id"]),
        # The lease scans only uploads waiting for inspection.
        Index(
            "ix_media_uploads_due",
            "available_at",
            postgresql_where=text("state IN ('QUARANTINED', 'SCANNING')"),
        ),
        Index("ix_media_uploads_tenant_id_created_at", "tenant_id", "created_at"),
        # Policy-source uploads (NRC-4) belong to a tenant, not a study.
        CheckConstraint(
            "purpose <> 'creative_media' OR study_id IS NOT NULL", name="creative_media_has_study"
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    # The study scopes who may see the upload; the creative is optional (library uploads). Only
    # policy-source uploads have no study (CHECK creative_media_has_study).
    study_id: Mapped[UUID | None]
    creative_id: Mapped[UUID | None]
    purpose: Mapped[UploadPurpose] = mapped_column(enum_column(UploadPurpose))
    state: Mapped[UploadState] = mapped_column(enum_column(UploadState))
    # The client's file name, for display only: never in object keys, tags or logs.
    name: Mapped[str] = mapped_column(String(255))
    # What the client declared when asking for the URL; the upload URL is signed to match.
    declared_mime_type: Mapped[str] = mapped_column(String(64))
    declared_size_bytes: Mapped[int] = mapped_column(BigInteger)
    declared_checksum_sha256: Mapped[str] = mapped_column(String(64))
    bucket: Mapped[str] = mapped_column(String(63))
    s3_key: Mapped[str] = mapped_column(String(1024))
    # Set when the upload completes; inspection and the asset read exactly this version.
    s3_version_id: Mapped[str | None] = mapped_column(String(1024))
    upload_url_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Found by inspection (the decoded file, not the declaration).
    mime_type: Mapped[str | None] = mapped_column(String(64))
    width_px: Mapped[int | None]
    height_px: Mapped[int | None]
    scan_engine: Mapped[str | None] = mapped_column(String(64))
    scan_simulated: Mapped[bool | None]
    scanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Why it was rejected, or the last retryable problem while it waits. Safe to show.
    reason_code: Mapped[str | None] = mapped_column(String(64))
    reason_message: Mapped[str | None] = mapped_column(String(UPLOAD_REASON_MAX_CHARS))
    # Media authorization, read by the approval and publication gates (NRC-12, NRC-14).
    identifiable_person: Mapped[IdentifiablePerson] = mapped_column(enum_column(IdentifiablePerson))
    usage_authorization_ref: Mapped[str | None] = mapped_column(String(200))
    # Channel values (app.models.enums.Channel).
    permitted_channels: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), server_default=text("'{}'")
    )
    authorization_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Inspection lease, the same scheme as outbox_events: set by lease_media_uploads().
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    leased_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    attempts: Mapped[int] = mapped_column(server_default=text("0"))
    failed_attempts: Mapped[int] = mapped_column(server_default=text("0"))
    created_by: Mapped[str] = mapped_column(String(128))
    completed_by: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
