# Roles, the capabilities they grant, and the per-request tenant context.

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from uuid import UUID

from app.auth.tokens import Principal

logger = logging.getLogger(__name__)


class Role(StrEnum):
    CREATOR = "creator"
    REVIEWER = "reviewer"
    POLICY_ADMIN = "policy_admin"
    PUBLISHER = "publisher"
    ADMIN = "admin"


class Capability(StrEnum):
    CREATIVE_READ = "creative:read"
    CREATIVE_WRITE = "creative:write"
    VALIDATION_RUN = "validation:run"
    GENERATION_RUN = "generation:run"
    APPROVAL_DECIDE = "approval:decide"
    POLICY_MANAGE = "policy:manage"
    PUBLICATION_MANAGE = "publication:manage"
    MEMBERSHIP_MANAGE = "membership:manage"


# Admin is not a super-role: it manages memberships but cannot approve, publish or edit rules.
ROLE_CAPABILITIES: Final[Mapping[Role, frozenset[Capability]]] = MappingProxyType(
    {
        Role.CREATOR: frozenset(
            {
                Capability.CREATIVE_READ,
                Capability.CREATIVE_WRITE,
                Capability.VALIDATION_RUN,
                Capability.GENERATION_RUN,
            }
        ),
        Role.REVIEWER: frozenset(
            {Capability.CREATIVE_READ, Capability.VALIDATION_RUN, Capability.APPROVAL_DECIDE}
        ),
        Role.POLICY_ADMIN: frozenset({Capability.CREATIVE_READ, Capability.POLICY_MANAGE}),
        Role.PUBLISHER: frozenset({Capability.CREATIVE_READ, Capability.PUBLICATION_MANAGE}),
        Role.ADMIN: frozenset({Capability.CREATIVE_READ, Capability.MEMBERSHIP_MANAGE}),
    }
)


def parse_roles(names: Sequence[str]) -> frozenset[Role]:
    # Map stored role names to roles. Unknown names grant nothing (fail closed).
    unknown = [name for name in names if name not in Role]
    if unknown:
        logger.warning("ignoring unknown membership roles: %r", unknown)
    return frozenset(Role(name) for name in names if name in Role)


def capabilities_for(roles: Iterable[Role]) -> frozenset[Capability]:
    return frozenset(capability for role in roles for capability in ROLE_CAPABILITIES[role])


@dataclass(frozen=True, slots=True)
class TenantContext:
    # Who is calling, in which tenant, and what they may touch there.

    tenant_id: UUID
    principal: Principal
    roles: frozenset[Role]
    capabilities: frozenset[Capability]
    study_ids: frozenset[UUID]
