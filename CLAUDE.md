# NRC Marketing API: guide for coding agents

Shared, project-level instructions for anyone working in this repo with an AI coding agent. Keep it
about the project, not about one person or ticket. Personal preferences go in your own
`~/.claude/CLAUDE.md` or a git-ignored `CLAUDE.local.md`.

## What this service is

The backend of NRC's Marketing module (Jira project NRC, epic NRC-29): marketers write a creative
brief for an approved clinical study, validate it against approved study facts and IPB policy
rules, generate an image with an AI provider (Higgsfield), validate the final image, and export a
review draft. Later phases add approvals, scheduling and social publishing.

The journey: brief → validate → generate → validate rendered media → (approve → publish).

Non-negotiables that come from the domain:
- Content is only ever generated from an approved, versioned study snapshot, never patient data.
- Validation outcomes are PASS, FAIL or NEEDS_REVIEW; an unknown or unassessed mandatory check
  never becomes PASS. Job execution state ("it ran") is separate from validation ("it passed").
- Every result is bound to exact versions (creative version, asset checksum, study snapshot,
  policy bundle). An edit makes earlier results stale.
- Generated output is a draft for review, never "approved" or "compliant" by itself.

## Architecture

Target (HLD v1.1): React frontend on S3 + CloudFront → API Gateway (WAF, Cognito) → private EKS
running this FastAPI service as separate deployments (API, connector/generation worker, policy,
media, webhook, event) → Aurora PostgreSQL (+ pgvector), private S3, Step Functions/SQS/EventBridge,
Bedrock for validation models, backend-only provider adapters (Higgsfield, Meta). NRC resources
live in NRC's own AWS account; buckets are named `nrc-<env>-<data>`.

How that maps to this code today:
- One repo, one Postgres database, row-level security per tenant.
- Slow or paid work never runs inside a request. The API commits a job row plus an outbox row in
  one transaction and returns 202; a separate worker process leases outbox rows and does the work.
  The outbox is the seam that SQS/Step Functions replace later.
- Providers sit behind a protocol (`app/services/providers/`); a labelled simulator is the default.
- Object storage is S3. Locally it is SeaweedFS from docker compose (not MinIO).

## Who owns what

| Area | Owner |
|---|---|
| Identity, tenancy, studies, creatives, policy/GraphRAG, validation, approvals, social, platform; backend reviews | Dinesh |
| Durable jobs and worker, generation provider adapters, media storage | Firoz |
| Frontend app (consumes `contracts/openapi.json`) | Praneeth |
| Infra, AWS, buckets, secrets, handbook | Prabhu |
| Cognito user pool | Bhushan |
| Product decisions (limits, credits) | Vishnu |

Changing another area's shapes or behaviour needs that owner's agreement.

## Building context before you change code

1. Read the Jira ticket (acceptance criteria and its "demo scope" section) and the PR it belongs to.
2. Read `README.md` for setup and the conventions table.
3. Read the code you will touch plus its tests; `app/models/` is the schema's source of truth.
4. For an API question, extract the path from `contracts/openapi.json` (don't read it whole; it is
   large): `uv run python -c "import json; d = json.load(open('contracts/openapi.json')); print(json.dumps(d['paths']['/v1/marketing/creatives'], indent=1))"`
5. Area rules in `.claude/rules/` load automatically for matching files; read them before adding
   a new file in that area.
6. Team rules come from the OctavoLab Engineering Handbook (Confluence space ENH). This file
   records where this service agreed an exception.

## Commands (repo root, PowerShell or bash)

| Task | Command |
|---|---|
| Start local services | `docker compose up -d --wait` |
| Install from the lockfile | `uv sync` |
| Apply migrations | `uv run alembic upgrade head` |
| Run the API | `uv run uvicorn app.main:create_app --factory --port 8000` |
| Tests | `uv run pytest -q -x`; one test: `uv run pytest -q tests/test_x.py::test_name` |
| Format and lint | `uv run ruff format .` then `uv run ruff check .` |
| Types | `uv run mypy app tests migrations scripts` |
| Migration checks | `uv run alembic check`, and `uv run alembic heads` prints exactly one head |
| Regenerate the contract | `uv run python -m scripts.export_openapi` |

Before calling a change done, run what CI runs (`.github/workflows/ci.yml`): ruff format --check,
ruff check, mypy, alembic upgrade/check/single head, pytest. CI also runs pip-audit on the locked
runtime dependencies, so add packages with `uv add`, never by editing `uv.lock`.

## Layout

- `app/main.py`: app factory; routers mount under `/v1/marketing` (`API_PREFIX`).
- `app/api/routes/`: thin handlers. Parse, authorize through deps, call one service, return a
  schema. No SQL or business rules here.
- `app/api/deps.py`: Cognito access token, `X-Tenant-Id` tenant context, capability checks
  (`CreativeReader`, `GenerationRunner`, ...), `Idempotency-Key` and `If-Match` headers.
- `app/services/`: business logic and transactions. Provider-specific code only in
  `app/services/providers/<provider>.py`.
- `app/workers/`: background process loops (no request handling).
- `app/models/`: SQLAlchemy models. `app/schemas/`: request and response models (the contract).
- `app/core/`: `Settings`, domain errors, JSON logging, request context.
- `app/db/session.py`: engine, sessions, `bind_tenant`.
- `migrations/versions/`, `scripts/` (one-off CLIs), `contracts/` (openapi.json, examples),
  `tests/` (shared fixtures only in `tests/conftest.py`; test modules never import each other).

## Invariants (ask before a change that needs to break one)

### Tenancy and auth
- Every tenant-owned table has `tenant_id`, forced RLS with the `tenant_isolation` policy and an
  explicit grant to the role that uses it (see `.claude/rules/migrations.md`).
- The API connects as `marketing_app` (no BYPASSRLS); background workers use their own role;
  migrations run as the owner through `MIGRATION_DATABASE_URL`. `bind_tenant` sets the tenant per
  transaction and is re-applied after each commit.
- `X-Tenant-Id` is checked against the caller's current membership on every request; a client
  header alone is never trusted.
- Another tenant's resource returns 404, not 403.
- Never weaken auth, including in tests or locally. Tests mint RS256 tokens against a static JWKS
  (`auth` fixture in `tests/conftest.py`). No bypass flags or seeded bearer tokens.

### Contract
- The frontend generates its TypeScript client from `contracts/openapi.json`. Changes are additive
  only (new optional fields, endpoints, headers). Renaming, removing or retyping a field needs the
  frontend owner's agreement first.
- Operation IDs are the handler function names; renaming a handler changes the client.
- After any route or schema change run `scripts.export_openapi` and commit the result;
  `tests/test_contract.py` fails if it is stale. Add sample responses to `contracts/examples/`.

### Errors
- Raise classes from `app/core/errors.py`; `STATUS_BY_ERROR` in `app/api/errors.py` maps them to
  HTTP statuses. Never build error JSON by hand. The body is always
  `{code, message, request_id, details, retryable}` (agreed with the frontend; it replaces the
  handbook's envelope for this service). 5xx messages stay generic.
- Missing or malformed `Idempotency-Key` gets 422; the same key with a different body gets 409
  `IDEMPOTENCY_KEY_REUSED`; missing `If-Match` gets 428; a stale one gets 412 with
  `details.current_version`.

### Async work, money and providers
- POSTs that start work return 202 only after the job, its outbox row and any reservation commit.
- State changes use guarded updates (`WHERE state IN (...)`); terminal states never regress.
- A provider call that may have been accepted is never blindly repeated: it is reconciled with the
  same provider idempotency key, or held as UNKNOWN for review.
- Spend is bounded before any paid call (estimate, per-tenant and global limits). Money and
  credits are integers (minor units / micro-USD), never floats.
- Details: `.claude/rules/storage-providers.md`.

### Migrations
- Expand-only, one Alembic head, models and migrations match (`.claude/rules/migrations.md`).

### Configuration and secrets
- All configuration goes through `Settings` in `app/core/config.py`. Anything an operator may tune
  per environment (limits, timeouts, TTLs, hosts, model ids, bucket) is a bounded setting; values
  that must match code or SQL, or protocol facts, are module constants with a one-line reason.
  One source of truth per value. Don't read `os.environ` directly.
- A required secret is a `SecretStr` with no default, so startup fails and names the variable.
- New variable: add it to `.env.example` marked `# secret` or `# setting`, and list the name
  (never the value) under `Config changes:` in the PR. Provider keys go in git-ignored
  `.env.local` locally and AWS Secrets Manager on servers.
- Never open, print or copy `.env` or `.env.local`. No secrets in code, logs, fixtures, OpenAPI
  examples or error messages.
- Log ids only (`job=... tenant=... request=...`). Never log prompts, tokens, credentials,
  presigned URLs or provider output URLs.

### Code and tests
- Python 3.14, mypy strict, the ruff rules in `pyproject.toml` (including `S` security, `DTZ`
  timezone-aware datetimes, `ASYNC`). Line length 100.
- `#` comments for single lines; docstrings only for longer explanations. Comment the why.
- Follow existing patterns before adding abstractions. Services own their transactions; there is
  no repository layer.
- Every behaviour change gets a test; a bug fix starts with a failing regression test
  (`.claude/rules/tests.md`). Simulated or mocked provider output is labelled as such and never
  counts as live evidence.

## Git and pull requests (handbook: Branching and Release Standard)

- One Jira ticket, one branch, one PR, squash-merged. Branch `<type>/<KEY>-<slug>` from `main`
  (`feature`, `fix`, `chore`, `refactor`), at most 3 working days, ideally under 1,000 changed
  lines. Stacked PRs are fine when one story builds on another; say so in the description.
- Commit and PR titles start with the key: `NRC-6: ...`.
- PR description: what and why, how it was tested, `Config changes:` (names only), deploy-order
  notes for migrations or roles, Jira link. Backend PRs need 2 approvals.
- Never push to `main`, force-push a shared branch or rewrite published history. GitHub doesn't
  enforce branch protection yet, so this is on us.
- Stage files by path (`git add <paths>`), not `git add -A`: local tool folders such as
  `.claude/worktrees/` live inside the checkout.
