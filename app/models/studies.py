# Studies and their approved, marketing-safe snapshots.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import Language, RecruitmentStatus, enum_column


class Study(Base):
    __tablename__ = "studies"
    __table_args__ = (UniqueConstraint("tenant_id", "id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    title: Mapped[str] = mapped_column(String(300))
    external_ref: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StudySnapshot(Base):
    # One immutable, approved version of a study's marketing facts. Never patient data.

    __tablename__ = "study_snapshots"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("study_id", "version"),
        ForeignKeyConstraint(["tenant_id", "study_id"], ["studies.tenant_id", "studies.id"]),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    study_id: Mapped[UUID]
    version: Mapped[int]
    recruitment_status: Mapped[RecruitmentStatus] = mapped_column(enum_column(RecruitmentStatus))
    language: Mapped[Language] = mapped_column(enum_column(Language))
    facts: Mapped[dict[str, object]] = mapped_column(JSONB)
    source_ref: Mapped[str] = mapped_column(String(300))
    approved_by: Mapped[str] = mapped_column(String(200))
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_synthetic: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
