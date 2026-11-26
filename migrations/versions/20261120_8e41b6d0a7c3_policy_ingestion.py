"""Policy ingestion (NRC-4): review state, bundle lifecycle fields, wordings, audit, build queues.

Expand-only. What it adds:
- policy_nodes and policy_edges: review_state, origin and reviewer columns. Existing rows were
  loaded by the owner import script, so approved ones become APPROVED / IMPORT / system:import
  (backfilled before the CHECKs that tie `approved` to review_state exist).
- policy_bundles: revision, based_on_bundle_id, updated_at, build lease columns, nullable
  approved_by/approved_at (a draft has no approver, CHECK approved_states_have_approver), and
  at most one ACTIVE bundle per tenant (partial unique index). Legacy rows were all ACTIVE and a
  tenant could have several: the data step keeps the highest ACTIVE version and retires the rest.
  This is what validation already used ("highest ACTIVE version"), so behaviour does not change.
- policy_rule_wordings (F40: wording per language) backfilled from policy_rules.exact_wording and
  prohibited_phrases, and policy_audit_events (insert-only for the API: SELECT, INSERT grants).
- policy_graph_frozen(): a trigger that stops the API and worker roles from writing graph rows of
  a bundle that is not DRAFT or REVIEW. The owner (import and seed scripts) is exempt.
- policy_sources: upload_id and extraction queue columns. outbox_events.job_id and
  media_uploads.study_id become nullable, each with a CHECK that says who may omit it.
- lease_policy_sources() and lease_policy_builds(): SECURITY DEFINER, ids only, EXECUTE for the
  worker role only, the same scheme as lease_media_uploads().
- Grants for the API and the worker role. Nothing is ever granted DELETE.

The data steps below run as the table owner. FORCE RLS applies to the owner too, so each one lifts
FORCE on its table for the length of the step (the owner then bypasses RLS) and restores it; this
needs no superuser. Everything stays inside the migration's transaction.

Revision ID: 8e41b6d0a7c3
Revises: 4b7d2e9f1a63
Create Date: 2026-11-20 16:00:00.000000
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import sqlalchemy as sa
from alembic import op

revision: str = "8e41b6d0a7c3"
down_revision: str | None = "4b7d2e9f1a63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
WORKER_ROLE = "marketing_worker"
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
IMPORT_ACTOR = "system:import"
LEASE_SOURCES = "lease_policy_sources(text, integer, integer)"
LEASE_BUILDS = "lease_policy_builds(text, integer, integer)"
TRIGGER_FUNCTION = "policy_graph_frozen()"
SHA256_FUNCTION = "policy_text_sha256(text)"
# convert_to() is only STABLE, which a generated column refuses; the encoding is fixed, so this
# wrapper is safely IMMUTABLE.
CREATE_SHA256_FUNCTION = """
CREATE FUNCTION policy_text_sha256(value text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
SET search_path = pg_catalog
AS $$ SELECT encode(sha256(convert_to(value, 'UTF8')), 'hex') $$
"""
# Tables the frozen-graph trigger guards. Each has tenant_id and policy_bundle_id.
FROZEN_TABLES = ("policy_nodes", "policy_rules", "policy_edges", "policy_rule_wordings")
TRIGGER_NAME = "policy_graph_frozen"
GRAPH_STATES_OPEN = ("DRAFT", "REVIEW")

APPROVER_STATES = "'APPROVED', 'BUILDING', 'READY', 'ACTIVE', 'RETIRED'"
APPROVER_CHECK = (
    f"lifecycle_state NOT IN ({APPROVER_STATES})"
    " OR (approved_by IS NOT NULL AND approved_at IS NOT NULL)"
)
APPROVED_MATCHES_REVIEW_STATE = "approved = (review_state = 'APPROVED')"
REVIEWED_WHEN_DECIDED = (
    "review_state = 'PROPOSED' OR (reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL)"
)
JOB_ID_CHECK = "job_id IS NOT NULL OR left(event_type, 14) = 'policy_bundle.'"
STUDY_CHECK = "purpose <> 'creative_media' OR study_id IS NOT NULL"
EXTRACTION_CHECK = "extraction_state = 'EXTRACTED' OR upload_id IS NOT NULL"

# Keeps the highest ACTIVE version per tenant and retires the other ACTIVE bundles. Exported for
# the tests. The downgrade does not restore the old states: which bundles were ACTIVE is lost.
RETIRE_SUPERSEDED_ACTIVE = """
UPDATE policy_bundles AS b
SET lifecycle_state = 'RETIRED'
WHERE b.lifecycle_state = 'ACTIVE'
  AND EXISTS (
      SELECT 1 FROM policy_bundles AS newer
      WHERE newer.tenant_id = b.tenant_id
        AND newer.lifecycle_state = 'ACTIVE'
        AND newer.version > b.version
  )
"""

# Wording rows for the rules that exist now. A wording is as reviewed as the rule's node (the
# language is the node's). Prohibited phrases are de-duplicated: a rule never repeats one.
BACKFILL_WORDINGS = f"""
INSERT INTO policy_rule_wordings (
    id, tenant_id, policy_bundle_id, policy_rule_id, language, kind, text,
    review_state, origin, reviewed_by, reviewed_at
)
SELECT gen_random_uuid(), w.tenant_id, w.policy_bundle_id, w.policy_rule_id, w.language,
       w.kind, w.text,
       CASE WHEN w.approved THEN 'APPROVED' ELSE 'PROPOSED' END,
       'IMPORT',
       CASE WHEN w.approved THEN '{IMPORT_ACTOR}' END,
       CASE WHEN w.approved THEN now() END
FROM (
    SELECT r.tenant_id, r.policy_bundle_id, r.id AS policy_rule_id, n.language, n.approved,
           'EXACT_WORDING' AS kind, r.exact_wording AS text
    FROM policy_rules AS r
    JOIN policy_nodes AS n
      ON n.tenant_id = r.tenant_id AND n.policy_bundle_id = r.policy_bundle_id
     AND n.id = r.node_id
    WHERE r.exact_wording ~ '[^[:space:]]'
    UNION
    SELECT r.tenant_id, r.policy_bundle_id, r.id, n.language, n.approved,
           'PROHIBITED_PHRASE', btrim(phrase, chr(32) || chr(9) || chr(10) || chr(13))
    FROM policy_rules AS r
    JOIN policy_nodes AS n
      ON n.tenant_id = r.tenant_id AND n.policy_bundle_id = r.policy_bundle_id
     AND n.id = r.node_id
    CROSS JOIN LATERAL unnest(r.prohibited_phrases) AS phrase
    WHERE phrase ~ '[^[:space:]]'
) AS w
"""  # noqa: S608  constants only

# Runs as the invoking role (SECURITY INVOKER), so RLS decides which bundle rows it sees. The
# owner is exempt: import and seed scripts backfill graphs into ACTIVE legacy bundles. FOR SHARE
# on the bundle row makes a concurrent submit or approve wait for this write, so an edit never
# lands in a bundle that has just left DRAFT or REVIEW.
CREATE_TRIGGER_FUNCTION = f"""
CREATE FUNCTION {TRIGGER_FUNCTION}
RETURNS trigger
LANGUAGE plpgsql VOLATILE SECURITY INVOKER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    owner_tenant uuid;
    bundle uuid;
    state text;
BEGIN
    IF current_user NOT IN ('{APP_ROLE}', '{WORKER_ROLE}') THEN
        IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
        RETURN NEW;
    END IF;
    -- An UPDATE checks the bundle the row leaves and the one it joins.
    FOR owner_tenant, bundle IN
        SELECT OLD.tenant_id, OLD.policy_bundle_id WHERE TG_OP IN ('UPDATE', 'DELETE')
        UNION
        SELECT NEW.tenant_id, NEW.policy_bundle_id WHERE TG_OP IN ('INSERT', 'UPDATE')
    LOOP
        SELECT lifecycle_state INTO state FROM public.policy_bundles
        WHERE tenant_id = owner_tenant AND id = bundle
        FOR SHARE;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'policy bundle is not visible to this role' USING ERRCODE = 'P0001';
        END IF;
        IF state NOT IN ({", ".join(f"'{s}'" for s in GRAPH_STATES_OPEN)}) THEN
            RAISE EXCEPTION 'policy graph is frozen: the bundle is %', state
                USING ERRCODE = 'P0001';
        END IF;
    END LOOP;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END
$$
"""  # noqa: S608  constants only

# Runs as the migration owner. FORCE RLS applies to the owner too, so a lease policy lets only
# this role (the function's owner) see every tenant's rows. batch is clamped to 1..50 and
# lease_seconds to 5..600. A finished lease is always cleared (leased_until NULL), so a set one
# here means it expired unfinished: that counts as a failed attempt. The same scheme as
# lease_media_uploads(). `due` holds the states that mean "queued" and "in progress".
_LEASE_TEMPLATE = """
CREATE FUNCTION {name}(owner text, batch integer, lease_seconds integer)
RETURNS TABLE ({id_column} uuid, tenant_id uuid)
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    WITH due AS (
        SELECT t.id FROM public.{table} AS t
        WHERE t.{state_column} IN ({due_states})
          AND t.available_at <= clock_timestamp()
          AND (t.leased_until IS NULL OR t.leased_until <= clock_timestamp())
        ORDER BY t.available_at
        LIMIT LEAST(GREATEST(batch, 1), 50)
        FOR UPDATE SKIP LOCKED
    )
    UPDATE public.{table} AS t
    SET leased_until = clock_timestamp()
            + make_interval(secs => LEAST(GREATEST(lease_seconds, 5), 600)),
        lease_owner = owner,
        attempts = t.attempts + 1,
        failed_attempts = t.failed_attempts
            + CASE WHEN t.leased_until IS NULL THEN 0 ELSE 1 END
    FROM due
    WHERE t.id = due.id
    RETURNING t.id, t.tenant_id
$$
"""
CREATE_LEASE_SOURCES = _LEASE_TEMPLATE.format(
    name="lease_policy_sources",
    id_column="source_id",
    table="policy_sources",
    state_column="extraction_state",
    due_states="'PENDING', 'EXTRACTING'",
)
CREATE_LEASE_BUILDS = _LEASE_TEMPLATE.format(
    name="lease_policy_builds",
    id_column="bundle_id",
    table="policy_bundles",
    state_column="lifecycle_state",
    due_states="'APPROVED', 'BUILDING'",
)

APP_GRANTS = {
    "policy_bundles": "SELECT, INSERT, UPDATE",
    "policy_nodes": "SELECT, INSERT, UPDATE",
    "policy_edges": "SELECT, INSERT, UPDATE",
    "policy_rules": "SELECT, INSERT, UPDATE",
    "policy_rule_wordings": "SELECT, INSERT, UPDATE",
    "policy_sources": "SELECT, INSERT, UPDATE",
    "policy_audit_events": "SELECT, INSERT",
}
# Tables the API could only read before this revision (downgrade restores that).
APP_READ_ONLY_BEFORE = (
    "policy_bundles",
    "policy_nodes",
    "policy_edges",
    "policy_rules",
    "policy_sources",
)
WORKER_GRANTS = {
    "policy_bundles": "SELECT",
    "policy_nodes": "SELECT",
    "policy_edges": "SELECT",
    "policy_rules": "SELECT",
    "policy_rule_wordings": "SELECT",
    "policy_sources": "SELECT",
    "policy_source_clauses": "SELECT, INSERT",
    "policy_node_embeddings": "SELECT, INSERT",
    "policy_index_builds": "SELECT, INSERT, UPDATE",
    "policy_audit_events": "INSERT",
}
# Column-level UPDATE: the worker moves work through its queue and nothing else.
_LEASE_COLUMNS = ("available_at", "leased_until", "lease_owner", "attempts", "failed_attempts")
WORKER_UPDATE_COLUMNS = {
    "policy_bundles": (
        "lifecycle_state",
        "updated_at",
        "last_error_code",
        *_LEASE_COLUMNS,
    ),
    "policy_sources": (
        "extraction_state",
        "extraction_error_code",
        "extracted_at",
        "page_count",
        *_LEASE_COLUMNS,
    ),
}


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=32)


@contextmanager
def _owner_sees_all_rows(*tables: str) -> Iterator[None]:
    # Table names are this file's constants, never input.
    for table in tables:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
    try:
        yield
    finally:
        for table in tables:
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")


def _enable_rls(table: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON {table} "
        f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
    )


def _created_at() -> sa.Column[object]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def _add_review_columns(table: str) -> None:
    # origin starts as IMPORT so existing rows are labelled as imported; new rows default to HUMAN.
    op.add_column(
        table,
        sa.Column(
            "review_state",
            _enum("reviewstate", "PROPOSED", "APPROVED", "REJECTED"),
            server_default="PROPOSED",
            nullable=False,
        ),
    )
    op.add_column(
        table,
        sa.Column(
            "origin",
            _enum("proposalorigin", "HUMAN", "IMPORT", "LLM_SIMULATED", "LLM"),
            server_default="IMPORT",
            nullable=False,
        ),
    )
    op.add_column(table, sa.Column("reviewed_by", sa.String(length=200), nullable=True))
    op.add_column(table, sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(table, sa.Column("review_note", sa.String(length=1000), nullable=True))
    with _owner_sees_all_rows(table):
        op.execute(
            f"UPDATE {table} SET review_state = 'APPROVED', reviewed_by = '{IMPORT_ACTOR}',"  # noqa: S608  constants only
            " reviewed_at = now() WHERE approved"
        )
    op.alter_column(table, "origin", server_default="HUMAN")
    op.create_check_constraint(
        "approved_matches_review_state", table, APPROVED_MATCHES_REVIEW_STATE
    )
    op.create_check_constraint("reviewed_when_decided", table, REVIEWED_WHEN_DECIDED)


def _drop_review_columns(table: str) -> None:
    op.drop_constraint(op.f(f"ck_{table}_reviewed_when_decided"), table, type_="check")
    op.drop_constraint(op.f(f"ck_{table}_approved_matches_review_state"), table, type_="check")
    for column in ("review_note", "reviewed_at", "reviewed_by", "origin", "review_state"):
        op.drop_column(table, column)


def _upgrade_bundles() -> None:
    op.add_column(
        "policy_bundles",
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
    )
    op.add_column("policy_bundles", sa.Column("based_on_bundle_id", sa.Uuid(), nullable=True))
    op.add_column(
        "policy_bundles",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column(
        "policy_bundles", sa.Column("available_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "policy_bundles", sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("policy_bundles", sa.Column("lease_owner", sa.String(length=128), nullable=True))
    op.add_column(
        "policy_bundles",
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.add_column(
        "policy_bundles",
        sa.Column("failed_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.add_column(
        "policy_bundles", sa.Column("last_error_code", sa.String(length=64), nullable=True)
    )
    op.alter_column("policy_bundles", "approved_by", nullable=True)
    op.alter_column("policy_bundles", "approved_at", nullable=True)
    op.create_foreign_key(
        op.f("fk_policy_bundles_tenant_id_policy_bundles"),
        "policy_bundles",
        "policy_bundles",
        ["tenant_id", "based_on_bundle_id"],
        ["tenant_id", "id"],
    )
    op.create_check_constraint("revision_positive", "policy_bundles", "revision >= 1")
    op.create_check_constraint("approved_states_have_approver", "policy_bundles", APPROVER_CHECK)
    with _owner_sees_all_rows("policy_bundles"):
        op.execute(RETIRE_SUPERSEDED_ACTIVE)
    op.create_index(
        "uq_policy_bundles_one_active",
        "policy_bundles",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("lifecycle_state = 'ACTIVE'"),
    )


def _upgrade_sources() -> None:
    op.add_column("policy_sources", sa.Column("upload_id", sa.Uuid(), nullable=True))
    op.add_column(
        "policy_sources",
        sa.Column(
            "extraction_state",
            _enum("sourceextractionstate", "PENDING", "EXTRACTING", "EXTRACTED", "FAILED"),
            server_default="EXTRACTED",
            nullable=False,
        ),
    )
    op.add_column(
        "policy_sources", sa.Column("extraction_error_code", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "policy_sources", sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("policy_sources", sa.Column("page_count", sa.Integer(), nullable=True))
    op.add_column(
        "policy_sources", sa.Column("available_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "policy_sources", sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("policy_sources", sa.Column("lease_owner", sa.String(length=128), nullable=True))
    op.add_column(
        "policy_sources",
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.add_column(
        "policy_sources",
        sa.Column("failed_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.create_foreign_key(
        op.f("fk_policy_sources_tenant_id_media_uploads"),
        "policy_sources",
        "media_uploads",
        ["tenant_id", "upload_id"],
        ["tenant_id", "id"],
    )
    op.create_unique_constraint(
        op.f("uq_policy_sources_tenant_id_upload_id"),
        "policy_sources",
        ["tenant_id", "upload_id"],
    )
    op.create_check_constraint("extraction_has_upload", "policy_sources", EXTRACTION_CHECK)


def _create_wordings() -> None:
    op.execute(CREATE_SHA256_FUNCTION)
    op.create_table(
        "policy_rule_wordings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("policy_rule_id", sa.Uuid(), nullable=False),
        sa.Column("language", _enum("policylanguage", "en", "es"), nullable=False),
        sa.Column(
            "kind", _enum("wordingkind", "EXACT_WORDING", "PROHIBITED_PHRASE"), nullable=False
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "text_sha256",
            sa.String(length=64),
            sa.Computed("policy_text_sha256(text)", persisted=True),
            nullable=False,
        ),
        sa.Column("source_clause_id", sa.Uuid(), nullable=True),
        sa.Column("translation_of_id", sa.Uuid(), nullable=True),
        _created_at(),
        sa.Column(
            "review_state",
            _enum("reviewstate", "PROPOSED", "APPROVED", "REJECTED"),
            server_default="PROPOSED",
            nullable=False,
        ),
        sa.Column(
            "origin",
            _enum("proposalorigin", "HUMAN", "IMPORT", "LLM_SIMULATED", "LLM"),
            server_default="HUMAN",
            nullable=False,
        ),
        sa.Column("reviewed_by", sa.String(length=200), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_note", sa.String(length=1000), nullable=True),
        sa.CheckConstraint(
            "language IN ('en', 'es')", name=op.f("ck_policy_rule_wordings_language")
        ),
        sa.CheckConstraint(
            "text ~ '[^[:space:]]'", name=op.f("ck_policy_rule_wordings_text_not_blank")
        ),
        sa.CheckConstraint(
            REVIEWED_WHEN_DECIDED, name=op.f("ck_policy_rule_wordings_reviewed_when_decided")
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "policy_rule_id"],
            ["policy_rules.tenant_id", "policy_rules.policy_bundle_id", "policy_rules.id"],
            name=op.f("fk_policy_rule_wordings_tenant_id_policy_rules"),
        ),
        # A composite key with a NULL column is not enforced (MATCH SIMPLE): wanted, a wording
        # may have no source clause.
        sa.ForeignKeyConstraint(
            ["tenant_id", "source_clause_id"],
            ["policy_source_clauses.tenant_id", "policy_source_clauses.id"],
            name=op.f("fk_policy_rule_wordings_tenant_id_policy_source_clauses"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "translation_of_id"],
            [
                "policy_rule_wordings.tenant_id",
                "policy_rule_wordings.policy_bundle_id",
                "policy_rule_wordings.id",
            ],
            name=op.f("fk_policy_rule_wordings_tenant_id_policy_rule_wordings"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_rule_wordings")),
        sa.UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "id",
            name=op.f("uq_policy_rule_wordings_tenant_id_policy_bundle_id_id"),
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "policy_rule_id",
            "language",
            "kind",
            "text_sha256",
            name="uq_policy_rule_wordings_rule_language_kind_text",
        ),
    )
    op.create_index(
        "uq_policy_rule_wordings_one_exact",
        "policy_rule_wordings",
        ["tenant_id", "policy_rule_id", "language"],
        unique=True,
        postgresql_where=sa.text("kind = 'EXACT_WORDING' AND review_state <> 'REJECTED'"),
    )
    with _owner_sees_all_rows("policy_rules", "policy_nodes"):
        # The new table has RLS off until _enable_rls, so the owner inserts across tenants.
        op.execute(BACKFILL_WORDINGS)


def _create_audit() -> None:
    op.create_table(
        "policy_audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("policy_bundle_id", sa.Uuid(), nullable=False),
        sa.Column(
            "action",
            _enum(
                "policyauditaction",
                "CREATE",
                "EDIT",
                "SUBMIT",
                "REVIEW",
                "APPROVE",
                "REJECT",
                "RETURN_TO_DRAFT",
                "BUILD_STARTED",
                "BUILD_READY",
                "BUILD_FAILED",
                "ACTIVATE",
                "ROLLBACK",
                "RETIRE",
            ),
            nullable=False,
        ),
        sa.Column(
            "from_state",
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
            nullable=True,
        ),
        sa.Column(
            "to_state",
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
            nullable=True,
        ),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.String(length=1000), nullable=True),
        sa.Column(
            "details",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"],
            ["policy_bundles.tenant_id", "policy_bundles.id"],
            name=op.f("fk_policy_audit_events_tenant_id_policy_bundles"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_audit_events")),
    )
    op.create_index(
        "ix_policy_audit_events_bundle",
        "policy_audit_events",
        ["tenant_id", "policy_bundle_id", "created_at"],
    )


def upgrade() -> None:
    _upgrade_bundles()

    # media_uploads and outbox_events: widening NOT NULL to NULL keeps every row and reader valid.
    op.alter_column("media_uploads", "study_id", nullable=True)
    op.create_check_constraint("creative_media_has_study", "media_uploads", STUDY_CHECK)
    op.alter_column("outbox_events", "job_id", nullable=True)
    op.create_check_constraint("job_id_unless_policy_bundle", "outbox_events", JOB_ID_CHECK)

    for table in ("policy_nodes", "policy_edges"):
        _add_review_columns(table)
    op.create_unique_constraint(
        op.f("uq_policy_rules_tenant_id_policy_bundle_id_id"),
        "policy_rules",
        ["tenant_id", "policy_bundle_id", "id"],
    )
    _upgrade_sources()
    _create_wordings()
    _create_audit()

    for table in ("policy_rule_wordings", "policy_audit_events"):
        _enable_rls(table)

    op.execute(CREATE_TRIGGER_FUNCTION)
    for table in FROZEN_TABLES:
        op.execute(
            f"CREATE TRIGGER {TRIGGER_NAME} BEFORE INSERT OR UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION {TRIGGER_FUNCTION}"
        )

    # The lease policies let the function owner see every tenant's queue rows (see the function).
    op.execute(
        "CREATE POLICY policy_source_lease ON policy_sources TO CURRENT_USER "
        "USING (true) WITH CHECK (true)"
    )
    op.execute(
        "CREATE POLICY policy_build_lease ON policy_bundles TO CURRENT_USER "
        "USING (true) WITH CHECK (true)"
    )
    for create, signature in (
        (CREATE_LEASE_SOURCES, LEASE_SOURCES),
        (CREATE_LEASE_BUILDS, LEASE_BUILDS),
    ):
        op.execute(create)
        op.execute(f"REVOKE ALL ON FUNCTION {signature} FROM PUBLIC")
        op.execute(f"GRANT EXECUTE ON FUNCTION {signature} TO {WORKER_ROLE}")

    # Table names, columns and privileges are the constants above, never input. No DELETE.
    for table, privileges in APP_GRANTS.items():
        op.execute(f"GRANT {privileges} ON {table} TO {APP_ROLE}")
    for table, privileges in WORKER_GRANTS.items():
        op.execute(f"GRANT {privileges} ON {table} TO {WORKER_ROLE}")
    for table, columns in WORKER_UPDATE_COLUMNS.items():
        op.execute(f"GRANT UPDATE ({', '.join(columns)}) ON {table} TO {WORKER_ROLE}")


def downgrade() -> None:
    # Local development only. States are not restored: bundles that the single-ACTIVE data step
    # retired stay RETIRED, and drafts get a placeholder approver below so approved_by can be
    # NOT NULL again. Rows that only exist because of this revision (policy-source uploads,
    # policy_bundle.* outbox events) are deleted so the old NOT NULLs can come back.
    for table in WORKER_GRANTS:
        op.execute(f"REVOKE ALL ON {table} FROM {WORKER_ROLE}")
    for table in APP_READ_ONLY_BEFORE:
        op.execute(f"REVOKE INSERT, UPDATE ON {table} FROM {APP_ROLE}")

    for signature in (LEASE_SOURCES, LEASE_BUILDS):
        op.execute(f"DROP FUNCTION {signature}")
    op.execute("DROP POLICY policy_build_lease ON policy_bundles")
    op.execute("DROP POLICY policy_source_lease ON policy_sources")
    for table in FROZEN_TABLES:
        op.execute(f"DROP TRIGGER {TRIGGER_NAME} ON {table}")
    op.execute(f"DROP FUNCTION {TRIGGER_FUNCTION}")

    # Dropping a table also drops its policy, grants, indexes and constraints.
    op.drop_table("policy_audit_events")
    op.drop_table("policy_rule_wordings")
    op.execute(f"DROP FUNCTION {SHA256_FUNCTION}")

    op.drop_constraint(op.f("ck_policy_sources_extraction_has_upload"), "policy_sources", "check")
    op.drop_constraint(op.f("uq_policy_sources_tenant_id_upload_id"), "policy_sources", "unique")
    op.drop_constraint(
        op.f("fk_policy_sources_tenant_id_media_uploads"), "policy_sources", "foreignkey"
    )
    for column in (
        "failed_attempts",
        "attempts",
        "lease_owner",
        "leased_until",
        "available_at",
        "page_count",
        "extracted_at",
        "extraction_error_code",
        "extraction_state",
        "upload_id",
    ):
        op.drop_column("policy_sources", column)
    op.drop_constraint(
        op.f("uq_policy_rules_tenant_id_policy_bundle_id_id"), "policy_rules", "unique"
    )
    for table in ("policy_edges", "policy_nodes"):
        _drop_review_columns(table)

    op.drop_constraint(
        op.f("ck_outbox_events_job_id_unless_policy_bundle"), "outbox_events", "check"
    )
    with _owner_sees_all_rows("outbox_events", "media_uploads"):
        op.execute("DELETE FROM outbox_events WHERE job_id IS NULL")
        op.execute("DELETE FROM media_uploads WHERE study_id IS NULL")
    op.alter_column("outbox_events", "job_id", nullable=False)
    op.drop_constraint(op.f("ck_media_uploads_creative_media_has_study"), "media_uploads", "check")
    op.alter_column("media_uploads", "study_id", nullable=False)

    op.drop_index("uq_policy_bundles_one_active", table_name="policy_bundles")
    op.drop_constraint(
        op.f("ck_policy_bundles_approved_states_have_approver"), "policy_bundles", "check"
    )
    with _owner_sees_all_rows("policy_bundles"):
        op.execute(
            "UPDATE policy_bundles SET approved_by = 'system:downgrade',"
            " approved_at = created_at WHERE approved_by IS NULL OR approved_at IS NULL"
        )
    op.alter_column("policy_bundles", "approved_at", nullable=False)
    op.alter_column("policy_bundles", "approved_by", nullable=False)
    op.drop_constraint(op.f("ck_policy_bundles_revision_positive"), "policy_bundles", "check")
    op.drop_constraint(
        op.f("fk_policy_bundles_tenant_id_policy_bundles"), "policy_bundles", "foreignkey"
    )
    for column in (
        "last_error_code",
        "failed_attempts",
        "attempts",
        "lease_owner",
        "leased_until",
        "available_at",
        "updated_at",
        "based_on_bundle_id",
        "revision",
    ):
        op.drop_column("policy_bundles", column)
