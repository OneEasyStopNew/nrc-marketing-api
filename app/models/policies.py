# Versioned policy bundles: the rules that validation checks creatives against.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PolicyBundle(Base):
    # One immutable, approved version of a tenant's rules. The highest version is the active one.

    __tablename__ = "policy_bundles"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "version"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    version: Mapped[int]
    name: Mapped[str] = mapped_column(String(200))
    source_ref: Mapped[str] = mapped_column(String(300))
    # Validated against app.schemas.policies.PolicyRule on write (seed) and on read.
    rules: Mapped[list[dict[str, object]]] = mapped_column(JSONB)
    is_synthetic: Mapped[bool]
    approved_by: Mapped[str] = mapped_column(String(200))
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
