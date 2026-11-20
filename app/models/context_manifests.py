# Context manifests (NRC-5): the stored, immutable record of what one retrieval decided.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKeyConstraint, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ContextManifestRecord(Base):
    # One row per distinct set of inputs: cache_key hashes every dependency (bundle, snapshot,
    # scope, retrieval settings), so the same request reuses the row and any change gets a new
    # one. The API role may insert and read rows but never change them.

    __tablename__ = "context_manifests"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "cache_key"),
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"], ["policy_bundles.tenant_id", "policy_bundles.id"]
        ),
        ForeignKeyConstraint(
            ["tenant_id", "study_snapshot_id"], ["study_snapshots.tenant_id", "study_snapshots.id"]
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    cache_key: Mapped[str] = mapped_column(String(64))
    policy_bundle_id: Mapped[UUID]
    policy_bundle_version: Mapped[int]
    study_snapshot_id: Mapped[UUID]
    study_snapshot_version: Mapped[int]
    retrieval_config_version: Mapped[int]
    # Validated by app.services.policy_context.ContextManifest on write and on read.
    manifest: Mapped[dict[str, object]] = mapped_column(JSONB)
    # sha256 of the manifest's canonical JSON; NULL on rows stored before it was recorded.
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
