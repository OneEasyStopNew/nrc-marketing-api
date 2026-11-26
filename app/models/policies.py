# Versioned policy bundles: the rules that validation checks creatives against.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import BundleLifecycleState, enum_column


class PolicyBundle(Base):
    # One immutable, approved version of a tenant's rules. The active policy is the highest
    # version whose lifecycle_state is ACTIVE.

    __tablename__ = "policy_bundles"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "version"),
        ForeignKeyConstraint(
            ["tenant_id", "based_on_bundle_id"], ["policy_bundles.tenant_id", "policy_bundles.id"]
        ),
        CheckConstraint("revision >= 1", name="revision_positive"),
        # A draft is not approved yet; every state from APPROVED on names who approved and when.
        CheckConstraint(
            "lifecycle_state NOT IN ('APPROVED', 'BUILDING', 'READY', 'ACTIVE', 'RETIRED')"
            " OR (approved_by IS NOT NULL AND approved_at IS NOT NULL)",
            name="approved_states_have_approver",
        ),
        # At most one ACTIVE bundle per tenant: activation retires the previous one first.
        Index(
            "uq_policy_bundles_one_active",
            "tenant_id",
            unique=True,
            postgresql_where=text("lifecycle_state = 'ACTIVE'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    version: Mapped[int]
    name: Mapped[str] = mapped_column(String(200))
    source_ref: Mapped[str] = mapped_column(String(300))
    # Validated against app.schemas.policies.PolicyRule on write (seed) and on read.
    rules: Mapped[list[dict[str, object]]] = mapped_column(JSONB)
    is_synthetic: Mapped[bool]
    # Empty while the bundle is a DRAFT or in REVIEW (CHECK approved_states_have_approver).
    approved_by: Mapped[str | None] = mapped_column(String(200))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # New bundles start as DRAFT, so one inserted without a state never becomes the active policy
    # (the migration set existing rows to ACTIVE). NRC-4 owns the transitions; enforcing a single
    # ACTIVE bundle per tenant belongs to its activation step.
    lifecycle_state: Mapped[BundleLifecycleState] = mapped_column(
        enum_column(BundleLifecycleState), server_default=BundleLifecycleState.DRAFT.value
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    activated_by: Mapped[str | None] = mapped_column(String(200))
    # NRC-4: the version the editor sends in If-Match; every edit of the bundle's graph bumps it.
    revision: Mapped[int] = mapped_column(server_default=text("1"))
    # The bundle this draft was copied from, if any.
    based_on_bundle_id: Mapped[UUID | None]
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Build queue, the same scheme as media_uploads: set by lease_policy_builds().
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    leased_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    attempts: Mapped[int] = mapped_column(server_default=text("0"))
    failed_attempts: Mapped[int] = mapped_column(server_default=text("0"))
    last_error_code: Mapped[str | None] = mapped_column(String(64))
