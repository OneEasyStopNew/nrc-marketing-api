"""The connector worker gets its own database role, marketing_worker, with exactly what it uses.

Prerequisite: the login role marketing_worker exists (NOSUPERUSER NOCREATEDB NOCREATEROLE
NOBYPASSRLS, not a member of the owner role), like marketing_app; docker/postgres-init.sql
creates it locally. Deploy order: the role must exist and every worker must already connect
through WORKER_DATABASE_URL BEFORE this runs, because it revokes the lease functions and
provider_spend from marketing_app. Row-level security applies to marketing_worker as to
marketing_app: every table below keeps its tenant_isolation policy, and outbox_lease stays
scoped to the function owner only.

The worker reads creative_versions (prompt and format), updates jobs and credit reservations,
writes outbox events (asset.generated), stores assets, keeps provider_spend and leases events.
marketing_app (the API) loses the lease functions and provider_spend, which only the worker
uses.

Revision ID: 3fb9adf5dfbe
Revises: 56b7fa46620e
Create Date: 2026-11-09 23:30:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "3fb9adf5dfbe"
down_revision: str | None = "56b7fa46620e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
WORKER_ROLE = "marketing_worker"
WORKER_GRANTS = {
    "creative_versions": "SELECT",
    # The API creates jobs and reservations; the worker only moves them on.
    "jobs": "SELECT, UPDATE",
    "outbox_events": "SELECT, INSERT, UPDATE",
    "credit_reservations": "SELECT, UPDATE",
    "assets": "SELECT, INSERT",
    "provider_spend": "SELECT, UPDATE",
}
# Granted to the API by earlier revisions; only the worker uses them.
WORKER_ONLY_TABLE = "provider_spend"
WORKER_ONLY_TABLE_PRIVILEGES = "SELECT, UPDATE"
LEASE_FUNCTIONS = (
    "lease_outbox_events(text, integer, integer, text[])",
    # The compatibility wrapper (da90399919d2), until it is dropped.
    "lease_outbox_events(text, integer, integer)",
)


def upgrade() -> None:
    # Names and privileges are the constants above, never user input.
    for table, privileges in WORKER_GRANTS.items():
        op.execute(f"GRANT {privileges} ON {table} TO {WORKER_ROLE}")
    op.execute(f"REVOKE {WORKER_ONLY_TABLE_PRIVILEGES} ON {WORKER_ONLY_TABLE} FROM {APP_ROLE}")
    for function in LEASE_FUNCTIONS:
        op.execute(f"REVOKE ALL ON FUNCTION {function} FROM PUBLIC")
        op.execute(f"REVOKE EXECUTE ON FUNCTION {function} FROM {APP_ROLE}")
        op.execute(f"GRANT EXECUTE ON FUNCTION {function} TO {WORKER_ROLE}")


def downgrade() -> None:
    for function in LEASE_FUNCTIONS:
        op.execute(f"REVOKE EXECUTE ON FUNCTION {function} FROM {WORKER_ROLE}")
        op.execute(f"GRANT EXECUTE ON FUNCTION {function} TO {APP_ROLE}")
    op.execute(f"GRANT {WORKER_ONLY_TABLE_PRIVILEGES} ON {WORKER_ONLY_TABLE} TO {APP_ROLE}")
    for table in WORKER_GRANTS:
        op.execute(f"REVOKE ALL ON {table} FROM {WORKER_ROLE}")
