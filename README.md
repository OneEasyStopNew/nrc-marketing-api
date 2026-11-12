# NRC Marketing backend

FastAPI service for the NRC Marketing journey (Jira epic NRC-29). This is the D1 foundation for the
13 Nov demo: login, tenant context, study snapshots, versioned creative drafts, the active policy and
the generation-eligibility check. The rest of the agreed contract is published, and its endpoints
return `501 NOT_IMPLEMENTED` until they ship.

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
token, so set the `COGNITO_*` values for the staging pool first (from Prabhu) and leave
`DEV_JWKS_FILE` empty.

## Test users and roles

Roles come from memberships in the database, which the API reads on every request. They are not
taken from the token's `cognito:groups` claim: groups apply to the whole user pool, not to one
tenant, and the API's database role can only read memberships. The web app reads roles from `/me`
and `/context`.

To give a Cognito user a role in an existing tenant (and every study in it):

```powershell
# The sub is in /me (signed in, before any membership) or in the Cognito console.
uv run python -m scripts.grant_access --tenant-id <tenant-uuid> `
    --member <sub>=creator --member <sub>=reviewer --member <sub>=publisher `
    --member <sub>=policy_admin --member <sub>=admin
```

The Cognito group `policy-admin` corresponds to the role `policy_admin`. Re-running replaces the user's roles and
reactivates the membership. Admin is not a super-role; it only manages memberships.

## Checks

CI (`.github/workflows/ci.yml`) runs the same checks on every push and pull request, plus a
`pip-audit` scan of the locked dependencies and a compile of the generated TypeScript types.

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
- Generate the TypeScript types with `npx openapi-typescript@7.4.4 contracts/openapi.json -o schema.d.ts`.
  CI checks that they compile.

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

Policy and generation:

- `GET /policies/active` returns the active policy (the highest version) and its rules, read-only.
  `/context` carries a summary in `active_policy`. The seeded policy is synthetic demo rules
  paraphrased from public FDA recruitment-advertising guidance, not the approved client policy.
- `GET /creatives/{id}/generation-eligibility[?version=N]` says whether generation may start and why
  not (`reason`). The generation job calls `app.services.eligibility.check_generation_eligibility`
  and raises `GenerationNotEligibleError` (409 `GENERATION_NOT_ELIGIBLE`) unless it is eligible.
  Until brief validation ships, the best possible answer is `NOT_VALIDATED`.

## Demo stubs (frontend integration before D2-D5)

Set `DEMO_STUBS=true` (and `DEMO_PUBLIC_BASE_URL` to the address the frontend uses) and run one
worker. Validations, generation jobs, assets and export then answer with simulated data in the
real contract shapes instead of `501`:

- Validation results come from phrase checks against the tenant's active policy (for example
  "guaranteed" fails REC-04), not from a real validator. Every run says so in `coverage_gaps`.
- A brief run takes 2 s and a generation job 6 s (QUEUED, RUNNING, WAITING_PROVIDER, SUCCEEDED),
  so polling UIs can be exercised. Generation needs a passing brief run for that exact version.
- The image is a placeholder SVG served from `/demo-files/{asset_id}`.
- Nothing is stored in the database; restarting the API forgets every run, job and asset.

Never enable it in staging or production.

## Dev tokens (offline, without Cognito)

For offline work without Cognito, sign tokens locally. The API still checks issuer, client,
scope, expiry and tenant membership; only the signing key changes.

```powershell
uv run python -m scripts.dev_token init                      # once; writes .dev-keys/ (git-ignored)
# add DEV_JWKS_FILE=<printed path> to .env, then restart the API
uv run python -m scripts.seed_demo --tenant-name "NRC Demo" --member dev-praneeth=creator
uv run python -m scripts.dev_token issue --sub dev-praneeth  # prints a token valid for 12 hours
```

Send it as `Authorization: Bearer <token>`. `DEMO_STUBS` and `DEV_JWKS_FILE` work only with
`APP_ENV=local` or `demo`; any other value refuses to start. The private key never leaves the
machine that ran `init`.

## Staging prerequisites

- A Postgres login role `marketing_app` that does not own the tables and has no `BYPASSRLS`. The API's
  `DATABASE_URL` uses it. Migrations run as the owner through `MIGRATION_DATABASE_URL`.
- Cognito: `COGNITO_REGION`, `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_IDS` and a resource-server
  scope matching `REQUIRED_SCOPE`.
- `APP_ENV=staging` (or `production`), `ENABLE_DOCS=false`, `CORS_ORIGINS` set to the frontend origin, and `FORWARDED_ALLOW_IPS` set to the
  load balancer's subnet range.
- Image: `docker build -t nrc-marketing .`. Run `alembic upgrade head` as a separate release step.

## Not done yet

Validation (D2), Higgsfield generation and assets (D3), image checks (D4), export (D5), and the
staging deployment (D6). Cleanup of old idempotency keys, the study import from the approved demo
study and the approved policy rules are still to do. The seeded study and policy are synthetic.
