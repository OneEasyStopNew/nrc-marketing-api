---
paths:
  - "migrations/**"
  - "app/models/**"
---

# Migrations and models

- Expand-only: new tables, or nullable or defaulted columns. In the same release as the code,
  never drop or rename a column, narrow a type, or add NOT NULL without a default (contract steps
  ship in a later release).
- Every tenant-owned table: `tenant_id`, `ENABLE` + `FORCE ROW LEVEL SECURITY`, the
  `tenant_isolation` policy on `app.tenant_id` (USING and WITH CHECK), and an explicit grant to
  the role that uses it (`marketing_app` for the API; the worker role for worker-only tables). Copy the pattern and constants from
  `migrations/versions/20261104_de579cd5c63a_foundation.py`. Table names in SQL come from
  constants, never input.
- Grants: `SELECT, INSERT, UPDATE` for mutable tables; `SELECT, INSERT` only for immutable ones
  (assets, evidence). No `DELETE` unless agreed.
- Reference tenant-owned rows with composite foreign keys `(tenant_id, <id>)` and add
  `UniqueConstraint("tenant_id", "id")` on the referenced table.
- Models in `app/models/` and migrations must match: `uv run alembic check` passes, and
  `uv run alembic heads` shows exactly one head. File names follow
  `YYYYMMDD_<revision>_<slug>.py`.
- Keep `downgrade()` working for local development.
- Test new tables for cross-tenant isolation (another tenant gets 404 / sees nothing).
