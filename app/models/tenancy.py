# Tenants and the memberships that grant users access to them.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ARRAY, DateTime, ForeignKey, ForeignKeyConstraint, String, func, true
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Membership(Base):
    # A user's current access to one tenant. It is checked on every request, not cached.

    __tablename__ = "memberships"

    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"), primary_key=True)
    user_sub: Mapped[str] = mapped_column(String(128), primary_key=True, index=True)
    roles: Mapped[list[str]] = mapped_column(ARRAY(String(32)))
    # Shown instead of the Cognito sub wherever the UI names who did something.
    display_name: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MembershipStudy(Base):
    # Grants a member access to one study inside the tenant.

    __tablename__ = "membership_studies"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "user_sub"],
            ["memberships.tenant_id", "memberships.user_sub"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(["tenant_id", "study_id"], ["studies.tenant_id", "studies.id"]),
    )

    tenant_id: Mapped[UUID] = mapped_column(primary_key=True)
    user_sub: Mapped[str] = mapped_column(String(128), primary_key=True)
    study_id: Mapped[UUID] = mapped_column(primary_key=True)
