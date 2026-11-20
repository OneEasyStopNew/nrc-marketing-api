"""Which roles hold which capabilities (app.auth.access)."""

import pytest

from app.auth.access import ROLE_CAPABILITIES, Capability, Role, capabilities_for, parse_roles
from scripts.seed_demo import parse_member

COGNITO_SUB = "4c1d8f2e-9a7b-4e3c-b5d6-0f1e2a3b4c5d"
POLICY_CAPABILITIES = {
    Capability.POLICY_MANAGE,
    Capability.POLICY_APPROVE,
    Capability.POLICY_ACTIVATE,
}


@pytest.mark.parametrize(
    ("role", "policy_capabilities"),
    [
        (Role.CREATOR, set()),
        (Role.REVIEWER, set()),
        (Role.POLICY_ADMIN, {Capability.POLICY_MANAGE}),
        (Role.POLICY_APPROVER, {Capability.POLICY_APPROVE, Capability.POLICY_ACTIVATE}),
        (Role.PUBLISHER, set()),
        (Role.ADMIN, set()),
    ],
)
def test_policy_capabilities_per_role(role: Role, policy_capabilities: set[Capability]) -> None:
    assert ROLE_CAPABILITIES[role] & POLICY_CAPABILITIES == policy_capabilities


def test_every_role_is_mapped() -> None:
    assert set(ROLE_CAPABILITIES) == set(Role)


def test_policy_admin_cannot_approve_or_activate() -> None:
    capabilities = capabilities_for(parse_roles(["policy_admin"]))

    assert Capability.POLICY_APPROVE not in capabilities
    assert Capability.POLICY_ACTIVATE not in capabilities


def test_policy_approver_cannot_edit_rules() -> None:
    capabilities = capabilities_for(parse_roles(["policy_approver"]))

    assert capabilities == {
        Capability.CREATIVE_READ,
        Capability.POLICY_APPROVE,
        Capability.POLICY_ACTIVATE,
    }


def test_both_policy_roles_together_hold_all_policy_capabilities() -> None:
    # No four-eyes rule yet (Q5): one membership may hold both roles.
    capabilities = capabilities_for(parse_roles(["policy_admin", "policy_approver"]))

    assert capabilities >= POLICY_CAPABILITIES


def test_grant_script_accepts_the_policy_approver_role() -> None:
    assert parse_member(f"{COGNITO_SUB}=policy_approver") == (COGNITO_SUB, ["policy_approver"])
