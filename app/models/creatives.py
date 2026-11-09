# Creatives and their immutable versions.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKeyConstraint, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import Channel, Language, PostFormat, enum_column


class Creative(Base):
    # A marketing draft. Content lives in CreativeVersion rows; current_version is the latest.

    __tablename__ = "creatives"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(["tenant_id", "study_id"], ["studies.tenant_id", "studies.id"]),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    study_id: Mapped[UUID]
    original_prompt: Mapped[str] = mapped_column(Text)
    current_version: Mapped[int]
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CreativeVersion(Base):
    # Immutable content of one creative version. The app role may insert but never update it.

    __tablename__ = "creative_versions"
    __table_args__ = (
        UniqueConstraint("creative_id", "version"),
        ForeignKeyConstraint(["tenant_id", "creative_id"], ["creatives.tenant_id", "creatives.id"]),
        ForeignKeyConstraint(
            ["tenant_id", "study_snapshot_id"],
            ["study_snapshots.tenant_id", "study_snapshots.id"],
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    creative_id: Mapped[UUID]
    version: Mapped[int]
    prompt: Mapped[str] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    study_snapshot_id: Mapped[UUID]
    language: Mapped[Language] = mapped_column(enum_column(Language))
    channel: Mapped[Channel] = mapped_column(enum_column(Channel))
    post_format: Mapped[PostFormat] = mapped_column(enum_column(PostFormat))
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
