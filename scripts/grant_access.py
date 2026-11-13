"""Grant Cognito users roles in an existing tenant, with access to all of its studies.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.grant_access --tenant-id <tenant-uuid> \
        --member <cognito-sub>=creator --member <cognito-sub>=policy_admin

Safe to re-run: an existing membership gets the given roles (replacing the old ones) and is
reactivated. Roles live only in the database. The API reads them on every request and never
takes them from the token's cognito:groups claim, because groups apply to the whole user pool,
not to one tenant.
"""

import argparse
import asyncio
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Final
from uuid import UUID

from sqlalchemy import func, literal, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings
from app.db.session import TenantSession, bind_tenant
from app.models import Membership, MembershipStudy, Study, Tenant
from scripts.seed_demo import parse_member

MAX_DISPLAY_NAME: Final = 200


class UnknownTenantError(LookupError):
    pass


def parse_display_name(value: str) -> tuple[str, str]:
    sub, _, raw_name = value.partition("=")
    # NFC for display text (PY-120); collapse runs of whitespace so names render on one line.
    name = " ".join(unicodedata.normalize("NFC", raw_name).split())
    if not sub or not name or len(name) > MAX_DISPLAY_NAME:
        raise argparse.ArgumentTypeError(
            f"expected <cognito-sub>=<name of 1-{MAX_DISPLAY_NAME} characters>, got {value!r}"
        )
    return sub, name


async def grant_access(
    db: AsyncSession,
    tenant_id: UUID,
    members: Sequence[tuple[str, list[str]]],
    display_names: Mapping[str, str] | None = None,
) -> list[UUID]:
    """Upsert each member's roles (and display name, when given) and give them every study.

    Returns:
        The study IDs the members can now use.

    Raises:
        UnknownTenantError: no tenant has this ID.
    """
    if await db.get(Tenant, tenant_id) is None:
        raise UnknownTenantError(f"no tenant with id {tenant_id}")
    await bind_tenant(db, tenant_id)
    names = display_names or {}
    for sub, roles in members:
        upsert = insert(Membership).values(
            tenant_id=tenant_id, user_sub=sub, roles=roles, display_name=names.get(sub)
        )
        await db.execute(
            upsert.on_conflict_do_update(
                index_elements=[Membership.tenant_id, Membership.user_sub],
                set_={
                    "roles": upsert.excluded.roles,
                    "is_active": True,
                    "display_name": func.coalesce(
                        upsert.excluded.display_name, Membership.display_name
                    ),
                },
            )
        )
        await db.execute(
            insert(MembershipStudy)
            .from_select(
                ["tenant_id", "user_sub", "study_id"],
                select(Study.tenant_id, literal(sub), Study.id).where(Study.tenant_id == tenant_id),
            )
            .on_conflict_do_nothing()
        )
    study_ids = await db.scalars(select(Study.id).where(Study.tenant_id == tenant_id))
    return sorted(study_ids)


async def _run(
    tenant_id: UUID, members: Sequence[tuple[str, list[str]]], display_names: Mapping[str, str]
) -> list[UUID]:
    engine = create_async_engine(
        OwnerDatabaseSettings().migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            study_ids = await grant_access(db, tenant_id, members, display_names)
            await db.commit()
    finally:
        await engine.dispose()
    return study_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--member", type=parse_member, action="append", required=True)
    parser.add_argument(
        "--display-name", type=parse_display_name, action="append", default=[], metavar="SUB=NAME"
    )
    args = parser.parse_args()
    display_names = dict(args.display_name)
    unknown = sorted(display_names.keys() - {sub for sub, _ in args.member})
    if unknown:
        parser.error(f"--display-name given for subs without --member: {unknown}")
    try:
        study_ids = asyncio.run(_run(args.tenant_id, args.member, display_names))
    except UnknownTenantError as err:
        parser.error(str(err))
    result = {
        "tenant_id": str(args.tenant_id),
        "members": {sub: roles for sub, roles in args.member},
        "study_ids": [str(study_id) for study_id in study_ids],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
