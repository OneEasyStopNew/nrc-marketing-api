# Validation runs: one stored, immutable result per check of a creative version.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKeyConstraint, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import ExecutionState, Outcome, ValidationInputKind, enum_column


class ValidationRun(Base):
    # Inputs are pinned (creative version, policy version, snapshot version, input hash), so a
    # later edit or a new policy makes the run stale instead of changing it. The app role may
    # insert runs but never update them.

    __tablename__ = "validation_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(["tenant_id", "creative_id"], ["creatives.tenant_id", "creatives.id"]),
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"], ["policy_bundles.tenant_id", "policy_bundles.id"]
        ),
        # Latest run of one version and kind: the generation gate and the UI read it.
        Index(
            "ix_validation_runs_latest",
            "tenant_id",
            "creative_id",
            "creative_version",
            "input_kind",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    creative_id: Mapped[UUID]
    creative_version: Mapped[int]
    input_kind: Mapped[ValidationInputKind] = mapped_column(enum_column(ValidationInputKind))
    state: Mapped[ExecutionState] = mapped_column(enum_column(ExecutionState))
    outcome: Mapped[Outcome | None] = mapped_column(enum_column(Outcome))
    input_hash: Mapped[str] = mapped_column(String(64))
    policy_bundle_id: Mapped[UUID]
    policy_bundle_version: Mapped[int]
    study_snapshot_version: Mapped[int]
    # gates, rule_results, findings and coverage_gaps; validated by the API schemas on read.
    result: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
