"""Client uploads (NRC-3): media_uploads, its lease function, and assets that can hold uploads.

media_uploads tracks an upload from the issued URL to READY or REJECTED. It is mutable (state,
lease, reasons), so it is separate from assets, which stays insert-only. When inspection passes,
the media worker inserts an assets row with the upload's id, so later steps only ever see
uploads that passed.

assets (expand only): creative_id, creative_version and job_id become nullable, because an
upload may have no creative and never has a job. New nullable study_id, upload_id and name. The
lineage_matches_origin CHECK keeps generated rows exactly as strict as before (creative, version
and job required) and requires upload_id = id and a study on uploaded rows. Existing rows are all
generated and already satisfy it. study_id is for uploads only: generated rows reach their study
through their creative, so there is nothing to backfill (and no need to lift FORCE RLS here).

The media worker leases due uploads across tenants only through lease_media_uploads(), a
SECURITY DEFINER function returning ids, the same scheme as lease_outbox_events(). It connects
as marketing_worker, which may read and update uploads but not create them.

Revision ID: deb7a7225553
Revises: b3e6bfed9303
Create Date: 2026-11-19 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "deb7a7225553"
down_revision: str | None = "b3e6bfed9303"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "marketing_app"
WORKER_ROLE = "marketing_worker"
TABLE = "media_uploads"
# Unset or reset settings read as '' after a transaction ends; NULLIF makes that match nothing.
CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
LEASE_FUNCTION = "lease_media_uploads(text, integer, integer)"
UPLOAD_STATES = ("PENDING_UPLOAD", "QUARANTINED", "SCANNING", "READY", "REJECTED", "EXPIRED")
DUE = sa.text("state IN ('QUARANTINED', 'SCANNING')")
LINEAGE_CHECK = (
    "(origin = 'generated' AND creative_id IS NOT NULL AND creative_version IS NOT NULL"
    " AND job_id IS NOT NULL AND upload_id IS NULL)"
    " OR (origin = 'uploaded' AND upload_id IS NOT NULL AND upload_id = id"
    " AND study_id IS NOT NULL"
    " AND job_id IS NULL)"
)

# Runs as the migration owner. FORCE RLS applies to the owner too, so the media_upload_lease
# policy lets only this role (the function's owner) see every tenant's uploads. batch is clamped
# to 1..50 and lease_seconds to 5..600, whatever the caller passes. A finished lease is always
# cleared (leased_until NULL), so a set one here means it expired unfinished: that counts as a
# failed attempt. SCANNING rows are leased again only after their lease expired (a crash).
CREATE_LEASE_FUNCTION = """
CREATE FUNCTION lease_media_uploads(owner text, batch integer, lease_seconds integer)
RETURNS TABLE (upload_id uuid, tenant_id uuid)
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    WITH due AS (
        SELECT u.id FROM public.media_uploads AS u
        WHERE u.state IN ('QUARANTINED', 'SCANNING')
          AND u.available_at <= clock_timestamp()
          AND (u.leased_until IS NULL OR u.leased_until <= clock_timestamp())
        ORDER BY u.available_at
        LIMIT LEAST(GREATEST(batch, 1), 50)
        FOR UPDATE SKIP LOCKED
    )
    UPDATE public.media_uploads AS u
    SET leased_until = clock_timestamp()
            + make_interval(secs => LEAST(GREATEST(lease_seconds, 5), 600)),
        lease_owner = owner,
        attempts = u.attempts + 1,
        failed_attempts = u.failed_attempts
            + CASE WHEN u.leased_until IS NULL THEN 0 ELSE 1 END
    FROM due
    WHERE u.id = due.id
    RETURNING u.id, u.tenant_id
$$
"""


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("creative_id", sa.Uuid(), nullable=True),
        sa.Column("purpose", _enum("uploadpurpose", "creative_media"), nullable=False),
        sa.Column("state", _enum("uploadstate", *UPLOAD_STATES), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("declared_mime_type", sa.String(length=64), nullable=False),
        sa.Column("declared_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("declared_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("bucket", sa.String(length=63), nullable=False),
        sa.Column("s3_key", sa.String(length=1024), nullable=False),
        sa.Column("s3_version_id", sa.String(length=1024), nullable=True),
        sa.Column("upload_url_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=True),
        sa.Column("width_px", sa.Integer(), nullable=True),
        sa.Column("height_px", sa.Integer(), nullable=True),
        sa.Column("scan_engine", sa.String(length=64), nullable=True),
        sa.Column("scan_simulated", sa.Boolean(), nullable=True),
        sa.Column("scanned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason_code", sa.String(length=64), nullable=True),
        sa.Column("reason_message", sa.String(length=500), nullable=True),
        sa.Column(
            "identifiable_person",
            _enum("identifiableperson", "yes", "no", "unknown"),
            nullable=False,
        ),
        sa.Column("usage_authorization_ref", sa.String(length=200), nullable=True),
        sa.Column(
            "permitted_channels",
            sa.ARRAY(sa.String(length=32)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("authorization_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("failed_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column("completed_by", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id", "study_id"],
            ["studies.tenant_id", "studies.id"],
            name=op.f("fk_media_uploads_tenant_id_studies"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "creative_id"],
            ["creatives.tenant_id", "creatives.id"],
            name=op.f("fk_media_uploads_tenant_id_creatives"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_media_uploads")),
        sa.UniqueConstraint("s3_key", name=op.f("uq_media_uploads_s3_key")),
        sa.UniqueConstraint("tenant_id", "id", name=op.f("uq_media_uploads_tenant_id_id")),
    )
    op.create_index(
        "ix_media_uploads_due", TABLE, ["available_at"], unique=False, postgresql_where=DUE
    )
    op.create_index(
        "ix_media_uploads_tenant_id_created_at",
        TABLE,
        ["tenant_id", "created_at"],
        unique=False,
    )

    # Table names and privileges are the constants above, never user input.
    op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON {TABLE} "
        f"USING (tenant_id = {CURRENT_TENANT}) WITH CHECK (tenant_id = {CURRENT_TENANT})"
    )
    op.execute(
        f"CREATE POLICY media_upload_lease ON {TABLE} TO CURRENT_USER "
        "USING (true) WITH CHECK (true)"
    )
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON {TABLE} TO {APP_ROLE}")
    # The worker moves uploads through inspection; only the API creates them.
    op.execute(f"GRANT SELECT, UPDATE ON {TABLE} TO {WORKER_ROLE}")
    op.execute(CREATE_LEASE_FUNCTION)
    op.execute(f"REVOKE ALL ON FUNCTION {LEASE_FUNCTION} FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {LEASE_FUNCTION} TO {WORKER_ROLE}")

    # assets: expand only. Widening NOT NULL to NULL keeps every existing row and reader valid.
    for column in ("creative_id", "creative_version", "job_id"):
        op.alter_column("assets", column, nullable=True)
    op.add_column("assets", sa.Column("study_id", sa.Uuid(), nullable=True))
    op.add_column("assets", sa.Column("upload_id", sa.Uuid(), nullable=True))
    op.add_column("assets", sa.Column("name", sa.String(length=255), nullable=True))
    op.create_foreign_key(
        op.f("fk_assets_tenant_id_studies"),
        "assets",
        "studies",
        ["tenant_id", "study_id"],
        ["tenant_id", "id"],
    )
    op.create_foreign_key(
        op.f("fk_assets_tenant_id_media_uploads"),
        "assets",
        TABLE,
        ["tenant_id", "upload_id"],
        ["tenant_id", "id"],
    )
    op.create_unique_constraint(op.f("uq_assets_upload_id"), "assets", ["upload_id"])
    op.create_check_constraint("lineage_matches_origin", "assets", LINEAGE_CHECK)


def downgrade() -> None:
    # Local development only. Uploaded assets cannot satisfy the restored NOT NULLs, so they go
    # first; this runs as the owner, so with FORCE RLS it must run where the owner bypasses RLS
    # (the local superuser), or it fails loudly at the NOT NULL step instead of losing data.
    op.execute("DELETE FROM assets WHERE origin = 'uploaded'")
    op.drop_constraint(op.f("ck_assets_lineage_matches_origin"), "assets", type_="check")
    op.drop_constraint(op.f("uq_assets_upload_id"), "assets", type_="unique")
    op.drop_constraint(op.f("fk_assets_tenant_id_media_uploads"), "assets", type_="foreignkey")
    op.drop_constraint(op.f("fk_assets_tenant_id_studies"), "assets", type_="foreignkey")
    op.drop_column("assets", "name")
    op.drop_column("assets", "upload_id")
    op.drop_column("assets", "study_id")
    for column in ("job_id", "creative_version", "creative_id"):
        op.alter_column("assets", column, nullable=False)
    op.execute(f"DROP FUNCTION {LEASE_FUNCTION}")
    op.drop_index("ix_media_uploads_tenant_id_created_at", table_name=TABLE)
    op.drop_index("ix_media_uploads_due", table_name=TABLE, postgresql_where=DUE)
    # Dropping the table also drops its policies and grants.
    op.drop_table(TABLE)
