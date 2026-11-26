"""Policy graph (NRC-5): sources, clauses, nodes, rules, edges and the retrieval index.

Adds the lifecycle columns to policy_bundles (existing rows become ACTIVE, so the active policy
does not change; new rows default to DRAFT) and the tables that retrieval reads. Writes go
through owner scripts until the NRC-4 admin APIs exist, so marketing_app only gets SELECT.
marketing_worker gets nothing yet: it holds exactly what it uses, and retrieval in workers
arrives in a later NRC-5 change. Every link inside the graph is a composite foreign key with
tenant_id and policy_bundle_id, so a row cannot point at another tenant's or another bundle's
row.

Prerequisite: pgvector is installed on the server, and the role running migrations may create it.
pgvector is not a trusted extension, so CREATE EXTENSION needs a superuser locally and a member of
rds_superuser on Aurora. Staging: Prabhu confirms the migration role before deploy (follow-up F29).

Revision ID: 7d2f9a41c6e8
Revises: deb7a7225553
Create Date: 2026-11-17 10:00:00.000000
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "7d2f9a41c6e8"
down_revision: str | None = "deb7a7225553"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
# Dropped in reverse order on downgrade, so children go before the tables they reference.
NEW_TABLES = (
    "policy_sources",
    "policy_source_clauses",
    "policy_nodes",
    "policy_rules",
    "policy_edges",
    "policy_node_embeddings",
    "policy_index_builds",
)
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
EMBEDDING_DIMENSIONS = 1024
RULE_ID_PATTERN = r"^[A-Z][A-Z0-9-]{1,39}$"
EMPTY_ARRAY = sa.text("'{}'")
NODE_COLUMNS = ["policy_nodes.tenant_id", "policy_nodes.policy_bundle_id", "policy_nodes.id"]
CLAUSE_COLUMNS = ["policy_source_clauses.tenant_id", "policy_source_clauses.id"]
BUNDLE_COLUMNS = ["policy_bundles.tenant_id", "policy_bundles.id"]


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def _created_at() -> sa.Column[datetime]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def _string_array(length: int, name: str) -> sa.Column[list[str]]:
    return sa.Column(
        name, sa.ARRAY(sa.String(length=length)), server_default=EMPTY_ARRAY, nullable=False
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "policy_bundles",
        sa.Column(
            "lifecycle_state",
            _enum(
                "bundlelifecyclestate",
                "DRAFT",
                "REVIEW",
                "APPROVED",
                "BUILDING",
                "READY",
                "ACTIVE",
                "REJECTED",
                "RETIRED",
            ),
            server_default="ACTIVE",
            nullable=False,
        ),
    )
    # The ACTIVE default above fills existing rows, so today's policy stays active. New bundles
    # start as DRAFT: one inserted without a state must never become the policy validation uses.
    op.alter_column("policy_bundles", "lifecycle_state", server_default="DRAFT")
    op.add_column(
        "policy_bundles", sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("policy_bundles", sa.Column("activated_by", sa.String(length=200), nullable=True))

    policy_language = ("en", "es")
    op.create_table(
        "policy_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("source_ref", sa.String(length=300), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=True),
        sa.Column("language", _enum("policylanguage", *policy_language), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("created_by", sa.String(length=200), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_policy_sources_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_sources")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_policy_sources_tenant_id_id")),
        # A second import of the same file reuses the row.
        sa.UniqueConstraint(
            "tenant_id",
            "checksum_sha256",
            name=op.f("uq_policy_sources_tenant_id_checksum_sha256"),
        ),
    )
    op.create_table(
        "policy_source_clauses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("clause_ref", sa.String(length=100), nullable=False),
        sa.Column("section", sa.String(length=200), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("language", _enum("policylanguage", *policy_language), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "source_id"],
            ["policy_sources.tenant_id", "policy_sources.id"],
            name=op.f("fk_policy_source_clauses_tenant_id_policy_sources"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_source_clauses")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_policy_source_clauses_tenant_id_id")),
        sa.UniqueConstraint(
            "tenant_id",
            "source_id",
            "clause_ref",
            name=op.f("uq_policy_source_clauses_tenant_id_source_id_clause_ref"),
        ),
    )
    op.create_table(
        "policy_nodes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("node_key", sa.String(length=80), nullable=False),
        sa.Column(
            "node_type",
            _enum(
                "policynodetype",
                "RULE",
                "CLAUSE",
                "DISCLOSURE",
                "EXCEPTION",
                "INTERPRETATION",
                "EXAMPLE",
            ),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", _enum("policylanguage", *policy_language), nullable=False),
        sa.Column("source_clause_id", sa.Uuid(), nullable=True),
        sa.Column("approved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        _string_array(64, "input_kinds"),
        _string_array(64, "channels"),
        _string_array(64, "languages"),
        _string_array(64, "audiences"),
        _string_array(64, "jurisdictions"),
        sa.Column("study_ids", sa.ARRAY(sa.Uuid()), server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("required_study_fact", sa.String(length=64), nullable=True),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector(CASE language WHEN 'es' THEN 'spanish'::regconfig "
                "ELSE 'english'::regconfig END, text)",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.CheckConstraint("language IN ('en', 'es')", name=op.f("ck_policy_nodes_language")),
        sa.CheckConstraint(
            "effective_from IS NULL OR effective_to IS NULL OR effective_from <= effective_to",
            name=op.f("ck_policy_nodes_effective_window"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"],
            BUNDLE_COLUMNS,
            name=op.f("fk_policy_nodes_tenant_id_policy_bundles"),
        ),
        # A composite key with a NULL column is not enforced (MATCH SIMPLE): that is wanted, a
        # node may have no source clause.
        sa.ForeignKeyConstraint(
            ["tenant_id", "source_clause_id"],
            CLAUSE_COLUMNS,
            name=op.f("fk_policy_nodes_tenant_id_policy_source_clauses"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_nodes")),
        # The target of every composite foreign key that points at a node.
        sa.UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "id",
            name=op.f("uq_policy_nodes_tenant_id_policy_bundle_id_id"),
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "node_key",
            name=op.f("uq_policy_nodes_tenant_id_policy_bundle_id_node_key"),
        ),
    )
    op.create_index(
        "ix_policy_nodes_search_vector", "policy_nodes", ["search_vector"], postgresql_using="gin"
    )
    op.create_table(
        "policy_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.Uuid(), nullable=False),
        sa.Column("rule_id", sa.String(length=40), nullable=False),
        sa.Column("rule_version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("requirement", sa.String(length=1000), nullable=False),
        sa.Column(
            "gate",
            _enum("gate", "completeness", "study_facts", "policy", "media"),
            nullable=False,
        ),
        sa.Column("severity", _enum("severity", "critical", "major", "minor"), nullable=False),
        sa.Column("mandatory", sa.Boolean(), nullable=False),
        sa.Column("check_type", _enum("checktype", "deterministic", "semantic"), nullable=False),
        sa.Column("exact_wording", sa.Text(), nullable=True),
        _string_array(200, "prohibited_phrases"),
        # Same pattern as PolicyRule.rule_id; owner scripts write until NRC-4, so the database
        # enforces it too.
        sa.CheckConstraint(
            f"rule_id ~ '{RULE_ID_PATTERN}'", name=op.f("ck_policy_rules_rule_id_pattern")
        ),
        sa.CheckConstraint("rule_version >= 1", name=op.f("ck_policy_rules_rule_version_positive")),
        sa.CheckConstraint(
            "array_position(prohibited_phrases, '') IS NULL",
            name=op.f("ck_policy_rules_no_empty_phrase"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "node_id"],
            NODE_COLUMNS,
            name=op.f("fk_policy_rules_tenant_id_policy_nodes"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_rules")),
        # One rule row per RULE node.
        sa.UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "node_id",
            name=op.f("uq_policy_rules_tenant_id_policy_bundle_id_node_id"),
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "rule_id",
            name=op.f("uq_policy_rules_tenant_id_policy_bundle_id_rule_id"),
        ),
    )
    op.create_table(
        "policy_edges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column(
            "edge_type",
            _enum(
                "policyedgetype",
                "supported_by",
                "requires_disclosure",
                "except_when",
                "supersedes",
                "applies_to",
            ),
            nullable=False,
        ),
        sa.Column("from_node_id", sa.Uuid(), nullable=False),
        sa.Column("to_node_id", sa.Uuid(), nullable=False),
        sa.Column("approved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("provenance_clause_id", sa.Uuid(), nullable=True),
        sa.Column("provenance_note", sa.Text(), nullable=True),
        sa.CheckConstraint("from_node_id <> to_node_id", name=op.f("ck_policy_edges_no_self_edge")),
        # NRC-4 AC: every approved edge links to an approved clause or an interpretation note.
        sa.CheckConstraint(
            "NOT approved OR provenance_clause_id IS NOT NULL"
            " OR (provenance_note IS NOT NULL AND btrim(provenance_note) <> '')",
            name=op.f("ck_policy_edges_approved_has_provenance"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "from_node_id"],
            NODE_COLUMNS,
            name=op.f("fk_policy_edges_from_node_policy_nodes"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "to_node_id"],
            NODE_COLUMNS,
            name=op.f("fk_policy_edges_to_node_policy_nodes"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "provenance_clause_id"],
            CLAUSE_COLUMNS,
            name=op.f("fk_policy_edges_provenance_clause_policy_source_clauses"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_edges")),
        sa.UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "edge_type",
            "from_node_id",
            "to_node_id",
            name=op.f("uq_policy_edges_endpoints"),
        ),
    )
    op.create_index(
        "ix_policy_edges_from_node",
        "policy_edges",
        ["tenant_id", "policy_bundle_id", "from_node_id"],
    )
    op.create_index(
        "ix_policy_edges_to_node", "policy_edges", ["tenant_id", "policy_bundle_id", "to_node_id"]
    )
    op.create_table(
        "policy_node_embeddings",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "node_id"],
            NODE_COLUMNS,
            name=op.f("fk_policy_node_embeddings_tenant_id_policy_nodes"),
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id",
            "policy_bundle_id",
            "node_id",
            "model_id",
            name=op.f("pk_policy_node_embeddings"),
        ),
    )
    op.create_table(
        "policy_index_builds",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("node_count", sa.Integer(), nullable=True),
        sa.Column("content_checksum", sa.String(length=64), nullable=True),
        sa.Column("state", _enum("indexbuildstate", "BUILDING", "READY", "FAILED"), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"],
            BUNDLE_COLUMNS,
            name=op.f("fk_policy_index_builds_tenant_id_policy_bundles"),
        ),
        sa.CheckConstraint(
            "state <> 'READY' OR (node_count IS NOT NULL"
            " AND content_checksum IS NOT NULL AND completed_at IS NOT NULL)",
            name=op.f("ck_policy_index_builds_ready_is_complete"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_index_builds")),
    )
    # A failed or superseded build stays as history; only one may be READY per bundle and model.
    op.create_index(
        "uq_policy_index_builds_ready",
        "policy_index_builds",
        ["tenant_id", "policy_bundle_id", "model_id"],
        unique=True,
        postgresql_where=sa.text("state = 'READY'"),
    )

    # Table names and roles are the constants above, never user input.
    for table in NEW_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
        )
        op.execute(f"GRANT SELECT ON {table} TO {APP_ROLE}")


def downgrade() -> None:
    # Dropping a table also drops its policy and grants.
    for table in reversed(NEW_TABLES):
        op.drop_table(table)
    op.drop_column("policy_bundles", "activated_by")
    op.drop_column("policy_bundles", "activated_at")
    op.drop_column("policy_bundles", "lifecycle_state")
    # The vector extension stays: other objects may use it, and dropping it needs a superuser.
