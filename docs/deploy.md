# Deploying the backend (NRC-26)

One image, three workloads, one Helm chart: `deploy/helm/nrc-marketing-api`.

| Workload (staging name) | Runs | Notes |
|---|---|---|
| `marketing-api` | uvicorn (the image's CMD) on :8000 | 2 replicas spread over zones, PDB, probes on `/livez` and `/readyz` |
| `connector-worker` | `python -m app.workers.connector` | leases outbox events from the DB; no HTTP port |
| `nrc-marketing-api-migrations` | `scripts.ensure_db_roles --set-passwords`, then `alembic upgrade head` | Helm pre-install/pre-upgrade hook, before any new pod |

Names default to `<release>-api` / `<release>-worker`; staging fixes them in
`values-staging.yaml` (`api.name`, `worker.name`). Pods run as uid 10001, read-only root filesystem, no Linux capabilities, no Kubernetes API token.

## How a deploy runs

GitHub Actions > **Deploy staging** > Run workflow (from `main`):

1. the full CI (lint, types, migrations, tests, chart render) runs first;
2. OIDC into AWS (role `nrc-stg-gha-api-deploy`, trusted only from this repository's `staging`
   environment), build the image and push `nrc-stg/marketing-api:<commit sha>`. ECR tags are
   immutable, so a rerun of the same commit reuses the image;
3. the three database Secrets are rebuilt from Secrets Manager (below). RDS rotates the nrc_admin
   password, so it is read on every deploy;
4. `helm upgrade --install --rollback-on-failure --wait`: the migration job runs first; if it
   fails, nothing else changes. If the new pods don't become ready, Helm puts the previous
   release back.

The schema is never rolled back. Every migration must work with the previous release's code
too: add columns and tables first, remove them in a later release.

### First Helm deploy

Staging first ran from plain manifests (nrc-platform-infra `k8s/app`). Helm can't adopt those,
and the Deployments' selectors can't change, so the first run deletes them: tick
**remove_old_manifests**. It deletes only `deployment/marketing-api`, `deployment/connector-worker`,
`service/marketing-api` and a TargetGroupBinding on the staging target group, and only when Helm
doesn't manage them. Without the tick the run stops and lists them. The API is down for the few
minutes between that delete and the new pods being ready. Other Deployments are left alone and
listed as a warning.

## What staging provides (infra)

GitHub `staging` environment variables (set by infra): `AWS_REGION`, `AWS_ROLE_ARN`,
`ECR_REGISTRY`, `EKS_CLUSTER`.

Secrets the workflow builds in namespace `nrc-marketing` (each source is a Secrets Manager login
secret with `username` and `password`; every URL goes through RDS Proxy with `?ssl=require`):

| Secret | Keys | Read by | Built from |
|---|---|---|---|
| `marketing-db-app` | `DATABASE_URL` | API | `nrc/stg/db-app` |
| `marketing-db-worker` | `DATABASE_URL`, `WORKER_DATABASE_URL` | connector worker | `nrc/stg/db-worker` (it connects only with `WORKER_DATABASE_URL`) |
| `marketing-db-admin` | `MIGRATION_DATABASE_URL`, `APP_DB_PASSWORD`, `WORKER_DB_PASSWORD` | migration job | the RDS-managed nrc_admin secret, plus the two passwords above |

The migration job sets the two passwords on `marketing_app` and `marketing_worker` (as SCRAM
verifiers, so the clear password never reaches the database logs), because RDS Proxy logs in
with the same secrets.

Pod Identity, on existing ServiceAccounts (the chart doesn't create them):

- `marketing-api` (role `nrc-stg-pod-marketing-api`): S3 media, KMS. No Bedrock, so
  `EMBEDDING_MODE=off` in the API.
- `connector-worker` (role `nrc-stg-pod-connector-worker`): S3 media, KMS, the Higgsfield
  secret (`nrc/stg/higgsfield`, read at startup), queues, Bedrock.
- `nrc-marketing-migrations`: created by the chart, no AWS access.

Networking: the internal ALB's target group `nrc-stg-api` is bound to Service `marketing-api`
port 8000 (target type `ip`, health check `/readyz`) by the chart's TargetGroupBinding.
`api.forwardedAllowIps` is empty for now, so uvicorn ignores forwarded headers; set it to the ALB
subnet CIDRs when the real client IP is needed.

Lines in `values-staging.yaml` marked "to confirm" are still open. The run refuses to deploy
while the file still has a `replace-with` placeholder.

Test users (one per role, NRC Demo tenant) sign in only through the hosted login (code + PKCE);
`USER_PASSWORD_AUTH` is off for the web client. Passwords are shared directly, never in the repo.

## Checking a deploy

```powershell
kubectl -n nrc-marketing get deploy,pods,jobs,targetgroupbindings -l app.kubernetes.io/instance=nrc-marketing-api
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
