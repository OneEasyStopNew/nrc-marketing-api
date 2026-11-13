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
uv run python -m scripts.seed_demo --tenant-name "NRC Staging" --member <cognito-sub>=creator
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

Add `--display-name "<sub>=Priya Shah"` (once per member) to show a name instead of the sub in
version history and the media library. Re-running without it keeps the name already set.

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

### Run the generation worker locally

Image generation runs in a separate worker process, never inside a request. With Postgres and
SeaweedFS up (see above), in a second terminal:

```powershell
uv run python -m app.workers.connector
```

`GENERATION_MODE` defaults to `simulator`: no provider account is needed and every image is stamped
SIMULATED. End to end you will see:

1. `POST /creative-jobs` returns `202` with a `job_id` (state `QUEUED`).
2. The worker leases the job, "submits" it to the simulated provider and polls it.
3. It checks the 1080x1080 PNG, stores it in the bucket and adds an `assets` row; the job becomes
   `SUCCEEDED` with a `result_asset_id`.
4. `GET /jobs/{id}` shows that state; `GET /assets/{asset_id}` returns the metadata and a 15-minute
   download URL.

Without `AWS_S3_BUCKET` the simulator worker still runs, but finished jobs stop at stage
`output_ready`. `GENERATION_MODE=live` refuses to start without storage.

The worker connects as `marketing_worker` through `WORKER_DATABASE_URL` and refuses to start
without it. A job it gives up on while the provider may still have it stays `UNKNOWN` with its
credits held; `uv run python -m scripts.list_unknown_jobs` lists those (ids only, read-only) for an
operator to settle.

### Higgsfield (live)

Live mode (`GENERATION_MODE=live`) stays off: the worker refuses to start until the validation-PASS
gate (NRC-7) exists, as well as without a key, `HIGGSFIELD_MODEL`, `HIGGSFIELD_OUTPUT_HOSTS` or
storage.

- **Key.** Locally, put it in `.env.local` (git-ignored) as `HIGGSFIELD_API_KEY=<key-id>:<key-secret>`,
  never in `.env`. On staging the worker reads it at startup from Secrets Manager
  (`HIGGSFIELD_SECRET_ID=nrc/stg/higgsfield`, JSON entry `HIGGSFIELD_API_KEY`)
  through its IAM role and keeps it in memory only.
- **Connectivity check.** `scripts/higgsfield_smoke.py` runs outside the API. `--help` lists the
  candidate model ids and their smallest documented settings, with the doc source.

  ```powershell
  # Free: calls only /estimate and prints model + USD.
  uv run python -m scripts.higgsfield_smoke estimate --models higgsfield-ai/soul/standard,higgsfield-ai/soul/v2/standard
  # Real money, one image: refuses without --confirm-spend, above GENERATION_MAX_COST_USD_PER_IMAGE,
  # or unless AWS_ENDPOINT_URL_S3 points at local SeaweedFS. Prints ids, checksum, size and S3 key.
  uv run python -m scripts.higgsfield_smoke generate --model higgsfield-ai/soul/v2/standard --confirm-spend
  ```

- **Cost guard.** Before the first submit the worker prices the image with `/estimate`. Above
  `GENERATION_MAX_COST_USD_PER_IMAGE` the job fails `COST_ABOVE_LIMIT`. It then reserves the estimate
  against `GENERATION_GLOBAL_BUDGET_USD` (default 40, across all tenants, table `provider_spend`) in
  the same transaction that marks it submitting; past the budget it fails `BUDGET_EXHAUSTED` and
  nothing is sent. Failed, rejected and canceled requests give the amount back, completed ones commit
  it (the actual cost if Higgsfield ever reports one, else the estimate), and `UNKNOWN` jobs keep it
  until reconciled.
- **Output host not allowed.** Images are downloaded only from `HIGGSFIELD_OUTPUT_HOSTS`. If
  Higgsfield answers with another host, the worker logs `provider output host not allowed ...
  host=<host>` at ERROR (host only, never the URL) and keeps the job waiting, retrying with
  backoff. To recover, add that host to `HIGGSFIELD_OUTPUT_HOSTS` and restart the worker before
  the output expires; after expiry the job fails `OUTPUT_EXPIRED` (billed). Without an
  `expires_at` from Higgsfield, expiry is `PROVIDER_OUTPUT_RETENTION_FALLBACK_SECONDS` (7 days)
  after the worker first saw the output completed.

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
- `contracts/examples/` (`d1.json`, `d3.json`) holds sample responses for mocks. They are validated against the
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
  Only the latest stored brief run of that exact version counts, and only if it was checked against
  the current policy version.

Validation (NRC-7):

- `POST /validations` with `input_kind` `brief` (the prompt) or `text` (the caption) runs the rule
  checks in `app/services/rules.py` and stores the run before it answers, so `state` is already
  `SUCCEEDED`. `job_id` equals `validation_run_id` until validation runs on the job framework.
- The checks are phrase and study-fact matches: REC-01 to REC-06 phrases, REC-07 the research and
  voluntary disclosure (captions only), REC-08 ages and payment amounts against the approved study
  facts. A policy rule with no check is `UNABLE_TO_ASSESS` and listed in `coverage_gaps`; if it is
  mandatory, the outcome is `NEEDS_REVIEW`. Every run also says that no review of meaning ran.
- Runs are immutable: the API role can insert them but not change them. `is_current` turns false
  when the creative gets a new version or the policy a new version.
- A tenant without a policy gets 409 `NO_ACTIVE_POLICY`. Image input answers 501 unless
  `DEMO_STUBS=true`.

## Demo stubs (frontend integration before D3-D5)

Set `DEMO_STUBS=true` (and `DEMO_PUBLIC_BASE_URL` to the address the frontend uses) and run one
worker. Image validation, generation jobs, assets, the media list and export then answer with
simulated data in the real contract shapes. Brief and caption validation is always real.

With `DEMO_STUBS=false`, generation jobs are the real durable jobs (NRC-6, run by the connector
worker). Asset reads and `GET /media` use stored images and require configured object storage
(503 if unavailable); image validation and export still answer `501`. The real media list returns
only successful generated images in the caller's tenant and permitted studies, with fresh,
short-lived links. Pass its `next_cursor` unchanged for older images; timestamp ties stay on the
same page, which may exceed the requested `limit`. Both generation paths use the same gate
(`app.services.generation_checks`):

- 412 `VERSION_CONFLICT`: not the current version.
- 409 `GENERATION_NOT_ELIGIBLE`, `details.reason` = `STUDY_CLOSED`, `LANGUAGE_NOT_ALLOWED` or
  `FORMAT_NOT_ALLOWED`.
- 409 `VALIDATION_NOT_PASSED` (not retryable): the latest brief run of this version did not pass
  against the active policy. `details.status` = `NOT_VALIDATED`, `VALIDATION_FAILED`,
  `NEEDS_REVIEW` or `NO_ACTIVE_POLICY`; `details.validation_run_id` is that run (null if none).

- A generation job takes 6 s (QUEUED, RUNNING, WAITING_PROVIDER, SUCCEEDED) and an image check 2 s,
  so polling UIs can be exercised. Generation needs a stored, current, passing brief run.
- The image is a placeholder SVG served from `/demo-files/{asset_id}`. Its check uses the brief,
  since the placeholder has no content: a brief that says "get paid" gets a REC-06 region.
- `GET /media` lists the finished images in the caller's studies, newest first. It is read-only;
  uploads are not supported.
- Export bundles the image with the latest stored validation run of the current version.
- Stub jobs, images and image checks live in memory; restarting the API forgets them.

Never enable it in staging or production.

## Dev tokens (offline, without Cognito)

For offline work without Cognito, sign tokens locally. The API still checks issuer, client,
scope, expiry and tenant membership; only the signing key changes.

```powershell
uv run python -m scripts.dev_token init                      # once; writes .dev-keys/ (git-ignored)
# add DEV_JWKS_FILE=<printed path> to .env, then restart the API
uv run python -m scripts.seed_demo --tenant-name "NRC Staging" --member dev-praneeth=creator
uv run python -m scripts.dev_token issue --sub dev-praneeth  # prints a token valid for 12 hours
```

Send it as `Authorization: Bearer <token>`. `DEMO_STUBS` and `DEV_JWKS_FILE` work only with
`APP_ENV=local` or `demo`; any other value refuses to start. The private key never leaves the
machine that ran `init`.

## Staging prerequisites

- A Postgres login role `marketing_app` that does not own the tables and has no `BYPASSRLS`. The API's
  `DATABASE_URL` uses it. Migrations run as the owner through `MIGRATION_DATABASE_URL`.
- A second login role `marketing_worker`, created the same way (not an owner, no `BYPASSRLS`, not a
  member of the owner role), before `alembic upgrade head`, which grants it the worker's tables. The
  connector worker's `WORKER_DATABASE_URL` uses it and the worker refuses to start without it.
- Deploy order for migration `3fb9adf5dfbe`: create `marketing_worker` and switch every worker to
  `WORKER_DATABASE_URL` BEFORE it runs. It revokes the lease function and `provider_spend` from
  `marketing_app`, so a worker still on `DATABASE_URL` stops leasing.
- Cognito: `COGNITO_REGION`, `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_IDS` and a resource-server
  scope matching `REQUIRED_SCOPE`.
- `APP_ENV=staging` (or `production`), `ENABLE_DOCS=false`, `CORS_ORIGINS` set to the frontend origin, and `FORWARDED_ALLOW_IPS` set to the
  load balancer's subnet range.
- Image: `docker build -t nrc-marketing .`. Run `alembic upgrade head` as a separate release step.

## Not done yet

A review of meaning (AI checks and GraphRAG over the policy sources) for briefs and captions,
image checks (D4), export (D5), and the staging deployment (D6). Live Higgsfield generation
through the app (D3) is wired to the validation gate and needs the live settings. Cleanup of
old idempotency keys, the study import from the approved demo study and the approved policy
rules are still to do. The seeded study and policy are synthetic.
