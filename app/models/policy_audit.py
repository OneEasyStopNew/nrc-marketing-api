# Insert-only audit trail of a policy bundle's lifecycle (NRC-4). Ids only, never policy text.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKeyConstraint, Index, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import BundleLifecycleState, PolicyAuditAction, enum_column


class PolicyAuditEvent(Base):
    """One thing that happened to a bundle: who, what, and the state change it caused.

    The API role may insert and read but never update or delete, so history cannot be rewritten;
    the worker role may insert (build events). `details` holds ids and counts only.
    """

    __tablename__ = "policy_audit_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"], ["policy_bundles.tenant_id", "policy_bundles.id"]
        ),
        Index("ix_policy_audit_events_bundle", "tenant_id", "policy_bundle_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    policy_bundle_id: Mapped[UUID]
    action: Mapped[PolicyAuditAction] = mapped_column(enum_column(PolicyAuditAction))
    from_state: Mapped[BundleLifecycleState | None] = mapped_column(
        enum_column(BundleLifecycleState)
    )
    to_state: Mapped[BundleLifecycleState | None] = mapped_column(enum_column(BundleLifecycleState))
    # The Cognito subject, or a system:* name for workers and scripts.
    actor: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str | None] = mapped_column(String(1000))
    details: Mapped[dict[str, object]] = mapped_column(JSONB, server_default=text("'{}'"))
    request_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
