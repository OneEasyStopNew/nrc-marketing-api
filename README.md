# NRC Marketing backend

FastAPI service for the NRC Marketing journey (Jira epic NRC-29). This is the D1 foundation for the
13 Nov demo: login, tenant context, study snapshots and versioned creative drafts. The rest of the
agreed contract is published, and its endpoints return `501 NOT_IMPLEMENTED` until they ship.

Stack: Python 3.14, FastAPI, SQLAlchemy 2.1 (async, asyncpg), Alembic, PostgreSQL 17 with
row-level security, Cognito access tokens.

## Run locally

Needs Docker and [uv](https://docs.astral.sh/uv/).

```powershell
Copy-Item .env.example .env        # then set the two passwords and the URLs that use them
docker compose up -d --wait        # Postgres + pgvector on 127.0.0.1:5432
uv sync
uv run alembic upgrade head
uv run python -m scripts.seed_demo --tenant-name "NRC Demo" --member <cognito-sub>=creator
uv run uvicorn app.main:create_app --factory --port 8000
```

API docs are at http://127.0.0.1:8000/docs when `ENABLE_DOCS=true`. Calls need a real Cognito access
token, so set the `COGNITO_*` values from Bhushan first.

### Local object storage (SeaweedFS)

`docker compose up -d --wait` also starts SeaweedFS, an S3-compatible server on
`http://127.0.0.1:8333`. It replaces MinIO. Generated images are stored there locally and in a private
S3 bucket on servers.

1. Add a profile to `~/.aws/config`:

   ```ini
   [profile nrc-local]
   region = us-east-1
   ```

2. Add its credentials to `~/.aws/credentials`. These values only open the local container (see
   `docker/seaweedfs-s3.json`); they are not AWS keys.

   ```ini
   [nrc-local]
   aws_access_key_id = localdev
   aws_secret_access_key = localdev-password
   ```

3. In `.env` set `AWS_PROFILE=nrc-local`, `AWS_S3_BUCKET=nrc-local-media`,
   `AWS_ENDPOINT_URL_S3=http://127.0.0.1:8333` and `S3_FORCE_PATH_STYLE=true`.
4. Create the bucket (versioning on). The script refuses to run unless `AWS_ENDPOINT_URL_S3` is set,
   so it can never touch real AWS:

   ```powershell
   uv run python -m scripts.create_local_bucket
   ```

The storage tests use the bucket `nrc-test-media` and skip themselves when SeaweedFS is not running.

## Checks

CI (`.github/workflows/ci.yml`) runs the same checks on every push and pull request, plus a
`pip-audit` scan of the locked dependencies.

```powershell
uv run ruff format --check . ; uv run ruff check . ; uv run mypy app tests migrations scripts
uv run alembic check
uv run pytest            # uses the marketing_test database; each test is rolled back
```

## Contract for the frontend

- `contracts/openapi.json` is the spec. Regenerate it with `uv run python -m scripts.export_openapi`.
  A test fails if it is stale. Operation IDs are the handler names (for example `create_creative`).
- `contracts/examples/d1.json` holds sample responses for mocks. They are validated against the
  schemas.
- Generate the TypeScript types with `npx openapi-typescript contracts/openapi.json -o schema.d.ts`.

Conventions:

| Header / rule | Meaning |
|---|---|
| `Authorization: Bearer <access token>` | Cognito access token. ID tokens are rejected. |
| `X-Tenant-Id` | Tenant to act in. It is checked against your current membership on every request. |
| `Idempotency-Key` | Required on POST commands. Resend the same key only when retrying the same body. A different body gets `409`. |
| `If-Match: "<version>"` | Required on PATCH. A missing value gets `428`; a stale one gets `412` with `details.current_version`. |
| Errors | Always `{code, message, request_id, details, retryable}`. |
| Text offsets (D2) | Zero-based Unicode code points, end exclusive. JS strings are UTF-16, so convert before slicing. |
| Image regions (D4) | Fractions 0–1 of the stored image's original width and height. |

## Staging prerequisites

- A Postgres login role `marketing_app` that does not own the tables and has no `BYPASSRLS`. The API's
  `DATABASE_URL` uses it. Migrations run as the owner through `MIGRATION_DATABASE_URL`.
- Cognito: `COGNITO_REGION`, `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_IDS` and a resource-server
  scope matching `REQUIRED_SCOPE`.
- `ENABLE_DOCS=false`, `CORS_ORIGINS` set to the frontend origin, and `FORWARDED_ALLOW_IPS` set to the
  load balancer's subnet range.
- Image: `docker build -t nrc-marketing .`. Run `alembic upgrade head` as a separate release step.

## Not done yet

Validation (D2), Higgsfield generation and assets (D3), image checks (D4), export (D5), and the
staging deployment (D6). Cleanup of old idempotency keys and the study import from the approved
demo study are still to do. The seeded study is synthetic.
