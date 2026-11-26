# Deploying the backend (NRC-26)

One image, three workloads, one Helm chart: `deploy/helm/nrc-marketing-api`.

| Workload | Runs | Notes |
|---|---|---|
| `<release>-api` | uvicorn (the image's CMD) on :8000 | 2 replicas spread over zones, PDB, probes on `/livez` and `/readyz` |
| `<release>-worker` | `python -m app.workers.connector` | leases outbox events from the DB; no HTTP port |
| `<release>-migrations` | `scripts.ensure_db_roles --set-passwords`, then `alembic upgrade head` | Helm pre-install/pre-upgrade hook, before any new pod |

Pods run as uid 10001, read-only root filesystem, no Linux capabilities, no Kubernetes API token.

## How a deploy runs

GitHub Actions > **Deploy staging** > Run workflow (from `main`):

1. the full CI (lint, types, migrations, tests, chart render) runs first;
2. OIDC into AWS (no stored keys), build the image, push `ECR_REPOSITORY:<commit sha>`; a
   rerun of the same commit reuses the image;
3. `helm upgrade --install --rollback-on-failure --wait`: the migration job runs first; if it
   fails, nothing else changes. If the new pods don't become ready, Helm puts the previous
   release back.

The schema is never rolled back. Every migration must work with the previous release's code
too: add columns and tables first, remove them in a later release.

## What has to exist first (Prabhu / infra)

GitHub `staging` environment variables:

- `AWS_DEPLOY_ROLE_ARN`: the OIDC role for this repository. It needs ECR push to the backend
  repository, `eks:DescribeCluster`, and an EKS access entry with rights in namespace
  `nrc-marketing`.
- `ECR_REPOSITORY`: e.g. `694702802041.dkr.ecr.us-east-1.amazonaws.com/nrc-stg/marketing-api`.

In the cluster (namespace `nrc-marketing`), Secrets with these keys:

| Secret (values-staging.yaml) | Keys |
|---|---|
| `nrc-marketing-api-env` (API and worker) | `DATABASE_URL` (marketing_app via RDS Proxy, ending `?ssl=require`); later `META_APP_SECRET`, `TOKEN_ENCRYPTION_KEY` |
| `nrc-marketing-worker-env` (worker only) | `WORKER_DATABASE_URL` (marketing_worker via RDS Proxy, `?ssl=require`) |
| `nrc-marketing-migrations-env` (migration job only) | `MIGRATION_DATABASE_URL` (nrc_admin, `?ssl=require`), `APP_DB_PASSWORD`, `WORKER_DB_PASSWORD` |

The two passwords must be the ones in `nrc/stg/db-app` and `nrc/stg/db-worker`, because RDS
Proxy logs in with those secrets. The migration job sets them on the roles (as SCRAM verifiers,
so the clear password never reaches the database logs).

Pod Identity (or IRSA) for the service accounts:

- `nrc-marketing-api`: S3 read/write on `nrc-stg-media-694702802041`, KMS for that bucket.
- `nrc-marketing-worker`: the same S3 rights, plus `secretsmanager:GetSecretValue` on
  `nrc/stg/higgsfield`.
- `nrc-marketing-migrations`: nothing in AWS.

Networking: the internal ALB's target group (ARN in `values-staging.yaml`, target type `ip`)
and the ALB subnets in `api.forwardedAllowIps`, so uvicorn trusts forwarded headers only from
the ALB. If the ALB is created by an Ingress instead, set `targetGroupBinding.enabled: false`.

Lines in `values-staging.yaml` marked "from Prabhu" are the values still to confirm.

## Checking a deploy

```powershell
kubectl -n nrc-marketing get deploy,pods,jobs -l app.kubernetes.io/instance=nrc-marketing-api
kubectl -n nrc-marketing logs job/nrc-marketing-api-migrations -c db-roles
kubectl -n nrc-marketing logs job/nrc-marketing-api-migrations -c alembic
helm -n nrc-marketing history nrc-marketing-api
```

Rolling back the pods by hand: `helm -n nrc-marketing rollback nrc-marketing-api <revision>`.

Render the chart locally (Helm 4):

```powershell
helm lint deploy/helm/nrc-marketing-api -f deploy/helm/nrc-marketing-api/values-staging.yaml --set image.tag=local --strict
helm template nrc-marketing-api deploy/helm/nrc-marketing-api -f deploy/helm/nrc-marketing-api/values-staging.yaml --set image.tag=local
```

## Not covered yet (rest of NRC-26)

Separate images per service, image signing and scanning gates, Argo CD promotion, KEDA for the
worker, NetworkPolicies, OpenTelemetry traces, dashboards, alarms and runbooks, and the
backup/restore drill.
