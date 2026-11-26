---
paths:
  - "app/services/**"
  - "app/workers/**"
  - "docker/**"
  - "docker-compose.yml"
---

# Storage, providers and workers

- S3: the bucket comes only from `AWS_S3_BUCKET`; S3 clients are created without keys (IAM role
  on servers, the local SeaweedFS profile on laptops via `AWS_ENDPOINT_URL_S3`,
  `S3_FORCE_PATH_STYLE`, `AWS_PROFILE`; never set those three on servers).
- One storage helper is the only caller of `put_object`. Keys:
  `<folder>/<yyyy>/<mm>/<uuid>.<ext>` with the real extension and content type; object tags
  `doc-type=...&retention=...`. No customer or study data in keys, tags or metadata.
- The database stores bucket, key, version id, size and sha256, never a URL. Downloads are
  presigned GET links (`ASSET_URL_TTL_SECONDS`, 15 minutes) issued after a tenant and
  authorization check. Never log presigned URLs.
- boto3 is synchronous: call it through `asyncio.to_thread` from async code.
- Provider (Higgsfield) calls run only in workers, never in a request handler. Store the provider
  idempotency key before submitting. A timeout after possible acceptance makes the job UNKNOWN;
  reconcile by resending the identical request with the same key, never a new key.
- Live mode never falls back to the simulator and refuses to start without keys or while the
  validation gate is a stub. Simulated output is stamped SIMULATED.
- Fetching provider output: https only, host allowlist (`HIGGSFIELD_OUTPUT_HOSTS`), resolved IP must
  be public, redirects off or re-checked, byte cap (`MAX_GENERATED_IMAGE_BYTES`), timeout, and
  the decoded image type must match the content type (Pillow with `MAX_IMAGE_PIXELS`).
- Job state changes use guarded updates (`WHERE state IN (...)`), so terminal states never regress
  and late events are ignored.
- Tests use `httpx.MockTransport` for providers and the local SeaweedFS for storage. CI never calls
  a live provider.
