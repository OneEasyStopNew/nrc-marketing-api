"""Provider concurrency cap (NRC-6): provider_in_flight() counts in-flight jobs across tenants.

The connector worker calls it, under a per-provider advisory lock, in the transaction that moves
a job to its submitting stage (app.workers.limits.submit_slot_free), so the cap
GENERATION_MAX_IN_FLIGHT_PER_PROVIDER holds across all tenants and workers. It is a SECURITY
DEFINER function returning a count only: no ids, no tenant data. In flight means the provider
has the request and is still working on it: submitting, or waiting at the provider (including a
job whose cancel is waiting for that outcome). UNKNOWN jobs are not counted: the outcome is not
known, and counting them could block every new job until an operator settles them.

The count runs under a lock every submit waits on, so a partial index on the in-flight states
keeps it off a full scan of all jobs ever run.

Deployment requirements, the same as for lease_outbox_events() (20261109_ffd6ebe6b868):
- The function's owner (the role running this migration) must own jobs. FORCE RLS applies to
  the owner too, so the provider_in_flight policy lets only that role read every tenant's jobs.
- Neither marketing_app nor marketing_worker may be a member of the owner role, or they would
  inherit that policy. Only marketing_worker gets EXECUTE.
- Consequence: the owner role (MIGRATION_DATABASE_URL: migrations, operator listing scripts)
  sees every tenant's jobs whatever app.tenant_id says, as it already does for outbox_events.
  Owner-run code that reads jobs must filter by tenant_id explicitly (awaiting_review_details
  does).

Revision ID: 4b7d2e9f1a63
Revises: 9e4c2b7a5d18
Create Date: 2026-11-19 22:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4b7d2e9f1a63"
down_revision: str | None = "9e4c2b7a5d18"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

WORKER_ROLE = "marketing_worker"
FUNCTION = "provider_in_flight(text)"
POLICY = "provider_in_flight"
INDEX = "ix_jobs_provider_in_flight"
IN_FLIGHT_STATES = "state IN ('RUNNING', 'WAITING_PROVIDER', 'CANCEL_REQUESTED')"

# Stage values are JobStage values (app/models/enums.py); states are ExecutionState values.
CREATE_FUNCTION = """
CREATE FUNCTION provider_in_flight(p_provider text)
RETURNS bigint
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    SELECT count(*) FROM public.jobs AS j
    WHERE j.provider = p_provider
      AND (
        (j.state IN ('RUNNING', 'CANCEL_REQUESTED') AND j.stage = 'submitting')
        OR (j.state IN ('WAITING_PROVIDER', 'CANCEL_REQUESTED') AND j.stage = 'waiting_provider')
      )
$$
"""


def upgrade() -> None:
    op.create_index(
        INDEX, "jobs", ["provider"], unique=False, postgresql_where=sa.text(IN_FLIGHT_STATES)
    )
    # Read-only and only for the migration owner, which owns the function.
    op.execute(f"CREATE POLICY {POLICY} ON jobs FOR SELECT TO CURRENT_USER USING (true)")
    op.execute(CREATE_FUNCTION)
    op.execute(f"REVOKE ALL ON FUNCTION {FUNCTION} FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {FUNCTION} TO {WORKER_ROLE}")


def downgrade() -> None:
    op.execute(f"DROP FUNCTION {FUNCTION}")
    op.execute(f"DROP POLICY {POLICY} ON jobs")
    op.drop_index(INDEX, table_name="jobs", if_exists=True)
