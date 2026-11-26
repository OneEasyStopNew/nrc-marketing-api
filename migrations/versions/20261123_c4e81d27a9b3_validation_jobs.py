"""Text validation as a worker job (NRC-7): judge and reuse columns, and the worker's grants.

Adds nullable columns to validation_runs that record how a result was produced (the judge model,
its prompt version, the model calls made, the evaluation version) and a reuse key, with an index
on (tenant_id, reuse_key). Expand-only: no existing column changes. validation_runs stays
insert-only: nobody gets UPDATE.

marketing_worker reads what a check needs (the creative, the study snapshot and the approved
policy graph) and writes the policy context manifest and the validation run. All of these tables
already force row-level security with the tenant_isolation policy, which applies to the worker
as to marketing_app, so it sees and writes only the tenant its session binds.

Revision ID: c4e81d27a9b3
Revises: a1a439290791
Create Date: 2026-11-23 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e81d27a9b3"
down_revision: str | None = "a1a439290791"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

WORKER_ROLE = "marketing_worker"
WORKER_GRANTS = {
    "creatives": "SELECT",
    "study_snapshots": "SELECT",
    "policy_bundles": "SELECT",
    "policy_sources": "SELECT",
    "policy_source_clauses": "SELECT",
    "policy_nodes": "SELECT",
    "policy_rules": "SELECT",
    "policy_edges": "SELECT",
    "policy_node_embeddings": "SELECT",
    "policy_index_builds": "SELECT",
    # Insert-only, like for marketing_app.
    "context_manifests": "SELECT, INSERT",
    "validation_runs": "SELECT, INSERT",
}
NEW_COLUMNS = (
    ("judge_model_id", sa.String(length=100)),
    ("judge_prompt_version", sa.String(length=40)),
    ("judge_attempts", sa.Integer()),
    ("evaluation_version", sa.String(length=40)),
    ("reuse_key", sa.String(length=64)),
)


def upgrade() -> None:
    for name, type_ in NEW_COLUMNS:
        op.add_column("validation_runs", sa.Column(name, type_, nullable=True))
    op.create_index(
        "ix_validation_runs_reuse",
        "validation_runs",
        ["tenant_id", "reuse_key"],
        postgresql_where=sa.text("reuse_key IS NOT NULL"),
    )
    # Names and privileges are the constants above, never user input.
    for table, privileges in WORKER_GRANTS.items():
        op.execute(f"GRANT {privileges} ON {table} TO {WORKER_ROLE}")


def downgrade() -> None:
    for table in WORKER_GRANTS:
        op.execute(f"REVOKE ALL ON {table} FROM {WORKER_ROLE}")
    op.drop_index(
        "ix_validation_runs_reuse",
        table_name="validation_runs",
        postgresql_where=sa.text("reuse_key IS NOT NULL"),
    )
    for name, _ in reversed(NEW_COLUMNS):
        op.drop_column("validation_runs", name)
