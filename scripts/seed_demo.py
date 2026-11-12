"""Create a synthetic demo tenant, study and policy, and grant Cognito users access to them.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.seed_demo --tenant-name "NRC Demo" \
        --member <cognito-sub>=creator --member <cognito-sub>=reviewer

Everything created is labelled SYNTHETIC. Replace the study with the approved demo study, and
the policy with the approved rules (IPB source still to be confirmed), before the client demo.
"""

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.auth.access import Role
from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession, bind_tenant
from app.models import Membership, MembershipStudy, PolicyBundle, Study, StudySnapshot, Tenant
from app.models.enums import Language, RecruitmentStatus
from app.schemas.policies import PolicyRule

_SUB = re.compile(r"[A-Za-z0-9-]{1,128}")
SYNTHETIC_FACTS = {
    "condition": "high blood pressure",
    "eligibility_summary": "Adults aged 18 to 75 who have been told they have high blood pressure.",
    "min_age": 18,
    "max_age": 75,
    "sites": ["Synthetic Demo Site"],
    "approved_claims": ["The study is testing an investigational treatment for blood pressure."],
    "required_disclosures": ["This is a research study. Taking part is voluntary."],
    "compensation_statement": None,
    "contact_url": "https://example.org/synthetic-demo-study",
}


# Demo rules paraphrased from public FDA guidance on recruitment advertising ("Recruiting Study
# Subjects", FDA information sheet). They are not the client's approved policy.
_FDA_SOURCE = "fda:information-sheet:recruiting-study-subjects"
DEMO_POLICY_NAME = "[SYNTHETIC] Demo recruitment advertising rules"
DEMO_POLICY_SOURCE_REF = "synthetic:demo-policy-v1 (paraphrased public FDA guidance)"
DEMO_POLICY_RULES: tuple[dict[str, object], ...] = (
    {
        "rule_id": "REC-01",
        "rule_version": 1,
        "title": "No safety or effectiveness claims",
        "requirement": "Do not state or imply that the investigational treatment is safe or "
        "effective for the condition being studied.",
        "gate": "policy",
        "severity": "critical",
        "mandatory": True,
        "applies_to": ["brief", "text", "image"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Claims of safety or effectiveness"},
    },
    {
        "rule_id": "REC-02",
        "rule_version": 1,
        "title": "No superiority claims",
        "requirement": "Do not claim the investigational treatment is equivalent or superior to "
        "any other treatment.",
        "gate": "policy",
        "severity": "critical",
        "mandatory": True,
        "applies_to": ["brief", "text"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Equivalence or superiority claims"},
    },
    {
        "rule_id": "REC-03",
        "rule_version": 1,
        "title": "Say it is investigational",
        "requirement": "Do not use terms such as 'new treatment', 'new medication' or 'new drug' "
        "without explaining that the treatment is investigational.",
        "gate": "policy",
        "severity": "major",
        "mandatory": True,
        "applies_to": ["brief", "text"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Use of the word new"},
    },
    {
        "rule_id": "REC-04",
        "rule_version": 1,
        "title": "No promised benefit",
        "requirement": "Do not state or imply a certain favourable outcome or any benefit beyond "
        "what the consent document and protocol describe.",
        "gate": "policy",
        "severity": "critical",
        "mandatory": True,
        "applies_to": ["brief", "text", "image"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Certainty of favourable outcome"},
    },
    {
        "rule_id": "REC-05",
        "rule_version": 1,
        "title": "No free treatment promise",
        "requirement": "Do not promise 'free treatment'. Say instead that taking part costs "
        "nothing, if that is true.",
        "gate": "policy",
        "severity": "major",
        "mandatory": True,
        "applies_to": ["brief", "text"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Free medical treatment"},
    },
    {
        "rule_id": "REC-06",
        "rule_version": 1,
        "title": "Payment not emphasised",
        "requirement": "Payment may be mentioned but must not be emphasised, for example by "
        "large or bold type or by leading with the amount.",
        "gate": "policy",
        "severity": "major",
        "mandatory": False,
        "applies_to": ["text", "image"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Emphasis on payment"},
    },
    {
        "rule_id": "REC-07",
        "rule_version": 1,
        "title": "Research and voluntary disclosure",
        "requirement": "Say that this is a research study and that taking part is voluntary, "
        "using the study's required disclosures.",
        "gate": "completeness",
        "severity": "critical",
        "mandatory": True,
        "applies_to": ["text"],
        "source": {"source_id": "study-snapshot", "clause": "required_disclosures"},
    },
    {
        "rule_id": "REC-08",
        "rule_version": 1,
        "title": "Approved study facts only",
        "requirement": "Statements about the condition, eligibility, sites and compensation must "
        "match the approved study snapshot.",
        "gate": "study_facts",
        "severity": "critical",
        "mandatory": True,
        "applies_to": ["brief", "text"],
        "source": {"source_id": "study-snapshot", "clause": "facts"},
    },
    {
        "rule_id": "REC-09",
        "rule_version": 1,
        "title": "No misleading medical imagery",
        "requirement": "Images must not show a cure, a before-and-after result or a clinician "
        "endorsing the treatment.",
        "gate": "media",
        "severity": "major",
        "mandatory": True,
        "applies_to": ["image"],
        "source": {"source_id": _FDA_SOURCE, "clause": "Certainty of favourable outcome"},
    },
)


def demo_policy_rules() -> list[dict[str, object]]:
    # Validated through the API schema so the stored JSON always matches what the API reads.
    return [PolicyRule.model_validate(rule).model_dump(mode="json") for rule in DEMO_POLICY_RULES]


def parse_member(value: str) -> tuple[str, list[str]]:
    sub, _, roles = value.partition("=")
    role_names = [role.strip() for role in roles.split(",") if role.strip()]
    if not _SUB.fullmatch(sub) or not role_names:
        raise argparse.ArgumentTypeError(f"expected <cognito-sub>=<role>[,<role>], got {value!r}")
    unknown = [name for name in role_names if name not in Role]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown roles {unknown}; use {[r.value for r in Role]}")
    return sub, role_names


async def seed(tenant_name: str, members: list[tuple[str, list[str]]]) -> dict[str, object]:
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    tenant_id, study_id, snapshot_id, policy_id = uuid4(), uuid4(), uuid4(), uuid4()
    now = datetime.now(tz=UTC)
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant_id)
            # No ORM relationships, so each parent level is flushed before its children.
            db.add(Tenant(id=tenant_id, name=tenant_name))
            await db.flush()
            db.add_all(Membership(tenant_id=tenant_id, user_sub=s, roles=r) for s, r in members)
            db.add(Study(id=study_id, tenant_id=tenant_id, title="[SYNTHETIC] Demo study"))
            db.add(
                PolicyBundle(
                    id=policy_id,
                    tenant_id=tenant_id,
                    version=1,
                    name=DEMO_POLICY_NAME,
                    source_ref=DEMO_POLICY_SOURCE_REF,
                    rules=demo_policy_rules(),
                    is_synthetic=True,
                    approved_by="synthetic-fixture",
                    approved_at=now,
                )
            )
            await db.flush()
            db.add(
                StudySnapshot(
                    id=snapshot_id,
                    tenant_id=tenant_id,
                    study_id=study_id,
                    version=1,
                    recruitment_status=RecruitmentStatus.RECRUITING,
                    language=Language.EN,
                    facts=SYNTHETIC_FACTS,
                    source_ref="synthetic:demo-seed-v1",
                    approved_by="synthetic-fixture",
                    approved_at=now,
                    is_synthetic=True,
                )
            )
            db.add_all(
                MembershipStudy(tenant_id=tenant_id, user_sub=sub, study_id=study_id)
                for sub, _ in members
            )
            await db.commit()
    finally:
        await engine.dispose()
    return {
        "tenant_id": str(tenant_id),
        "study_id": str(study_id),
        "snapshot_id": str(snapshot_id),
        "policy_bundle_id": str(policy_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-name", required=True)
    parser.add_argument("--member", type=parse_member, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(seed(args.tenant_name, args.member)), indent=2))


if __name__ == "__main__":
    main()
