# Durable background jobs, the outbox that drives the worker, and per-job credit reservations.

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import ExecutionState, JobStage, enum_column

# jobs.failure_message column size; writers truncate to it.
FAILURE_MESSAGE_MAX_CHARS = 500


class ReservationState(StrEnum):
    RESERVED = "RESERVED"
    # Provider outcome unknown: the credits stay counted until an operator or reconcile settles it.
    HELD = "HELD"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"


class SpendState(StrEnum):
    # This job's share of the global provider budget (provider_spend).
    RESERVED = "RESERVED"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"


class Job(Base):
    """One unit of background work. Generic, so validation runs can reuse it with their own kind.

    State changes go through app.services.jobs.transition (a guarded UPDATE), so a terminal
    state never regresses and a late or duplicate event is ignored.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        # Referenced by assets, so an asset records exactly the version its job generated.
        UniqueConstraint("id", "tenant_id", "creative_id", "creative_version"),
        ForeignKeyConstraint(["tenant_id", "creative_id"], ["creatives.tenant_id", "creatives.id"]),
        ForeignKeyConstraint(
            ["creative_id", "creative_version"],
            ["creative_versions.creative_id", "creative_versions.version"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "study_snapshot_id"],
            ["study_snapshots.tenant_id", "study_snapshots.id"],
        ),
        # use_alter: jobs and assets reference each other, so this one is added after both exist.
        ForeignKeyConstraint(
            ["tenant_id", "result_asset_id"],
            ["assets.tenant_id", "assets.id"],
            use_alter=True,
        ),
        Index("ix_jobs_tenant_id_state", "tenant_id", "state"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    # A JobKind value (app.schemas.jobs); plain text so a new kind needs no migration.
    kind: Mapped[str] = mapped_column(String(32))
    creative_id: Mapped[UUID]
    creative_version: Mapped[int]
    study_snapshot_id: Mapped[UUID]
    study_snapshot_version: Mapped[int]
    policy_bundle_version: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[ExecutionState] = mapped_column(enum_column(ExecutionState))
    stage: Mapped[JobStage | None] = mapped_column(enum_column(JobStage))
    attempt_count: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int]
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(100))
    # "nrc-job-<job id>", set at accept and resent unchanged on every submit, so the provider
    # never bills twice. Older jobs hold a random UUID in text form.
    provider_idempotency_key: Mapped[str | None] = mapped_column(String(255))
    provider_request_id: Mapped[str | None] = mapped_column(String(200))
    # Provider's price estimate taken before the first submit, in micro-USD (integer money).
    provider_cost_estimate_usd_micros: Mapped[int | None] = mapped_column(BigInteger)
    # The provider's reported cost, when its status response states one (micro-USD).
    provider_cost_actual_usd_micros: Mapped[int | None] = mapped_column(BigInteger)
    # Set with the reservation in provider_spend and changed only by app.services.spend.
    provider_spend_state: Mapped[SpendState | None] = mapped_column(enum_column(SpendState))
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_message: Mapped[str | None] = mapped_column(String(FAILURE_MESSAGE_MAX_CHARS))
    result_asset_id: Mapped[UUID | None]
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OutboxEvent(Base):
    """Work for the connector worker, written in the same transaction as its job.

    The worker leases due rows across tenants only through the lease_outbox_events() database
    function, which returns ids; everything else it reads runs under the job's tenant (RLS).
    """

    __tablename__ = "outbox_events"
    __table_args__ = (
        ForeignKeyConstraint(["tenant_id", "job_id"], ["jobs.tenant_id", "jobs.id"]),
        # The lease filters by event type first, so unhandled types are never scanned.
        Index(
            "ix_outbox_events_due",
            "event_type",
            "available_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    job_id: Mapped[UUID]
    event_type: Mapped[str] = mapped_column(String(64))
    # Ids only: never prompts, URLs or provider responses.
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    leased_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    # Every lease, polls included (drives the poll backoff).
    attempts: Mapped[int] = mapped_column(default=0)
    # Failures only: a lease that expired unfinished, or no handler. Capped by
    # GENERATION_MAX_ATTEMPTS; polls and advisory-lock misses never count.
    failed_attempts: Mapped[int] = mapped_column(server_default=text("0"))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CreditReservation(Base):
    # Generation credits held for one job, in integer units (separate from money).

    __tablename__ = "credit_reservations"
    __table_args__ = (
        UniqueConstraint("job_id"),
        ForeignKeyConstraint(["tenant_id", "job_id"], ["jobs.tenant_id", "jobs.id"]),
        Index("ix_credit_reservations_tenant_id_created_at", "tenant_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    job_id: Mapped[UUID]
    units_reserved: Mapped[int]
    units_committed: Mapped[int] = mapped_column(default=0)
    state: Mapped[ReservationState] = mapped_column(enum_column(ReservationState))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProviderSpend(Base):
    """Running spend per provider across all tenants, checked against GENERATION_GLOBAL_BUDGET_USD.

    No tenant data, so no tenant_id and no RLS (an approved exception, see its migration). Only
    marketing_worker may read or update it, through app.services.spend, inside the transaction of
    a guarded job transition.
    """

    __tablename__ = "provider_spend"
    __table_args__ = (
        CheckConstraint("reserved_usd_micros >= 0", name="reserved_not_negative"),
        CheckConstraint("committed_usd_micros >= 0", name="committed_not_negative"),
    )

    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    # Estimates of jobs that were submitted (or may have been) and are not settled yet.
    reserved_usd_micros: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))
    # Completed generations: the actual cost when the provider reported one, else the estimate.
    committed_usd_micros: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
