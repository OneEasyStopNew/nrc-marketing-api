"""Create a synthetic demo tenant and study, and grant Cognito users access to them.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.seed_demo --tenant-name "NRC Demo" \
        --member <cognito-sub>=creator --member <cognito-sub>=reviewer

Everything created is labelled SYNTHETIC. Replace the study with the approved demo study
before the client demonstration.
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
from app.models import Membership, MembershipStudy, Study, StudySnapshot, Tenant
from app.models.enums import Language, RecruitmentStatus

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


def _member(value: str) -> tuple[str, list[str]]:
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
    tenant_id, study_id, snapshot_id = uuid4(), uuid4(), uuid4()
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            await bind_tenant(db, tenant_id)
            # No ORM relationships, so each parent level is flushed before its children.
            db.add(Tenant(id=tenant_id, name=tenant_name))
            await db.flush()
            db.add_all(Membership(tenant_id=tenant_id, user_sub=s, roles=r) for s, r in members)
            db.add(Study(id=study_id, tenant_id=tenant_id, title="[SYNTHETIC] Demo study"))
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
                    approved_at=datetime.now(tz=UTC),
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
    return {"tenant_id": str(tenant_id), "study_id": str(study_id), "snapshot_id": str(snapshot_id)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-name", required=True)
    parser.add_argument("--member", type=_member, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(seed(args.tenant_name, args.member)), indent=2))


if __name__ == "__main__":
    main()
