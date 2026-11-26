# Client uploads: frontend guide (NRC-3, NRC-18, NRC-41)

Examples are in `uploads.json`. All data is synthetic. Images only (PNG, JPEG) for now; video
comes later.

## The flow

1. `GET /v1/marketing/assets/upload-limits`: read the accepted types, max bytes and URL lifetime.
2. Hash the file in the browser with SHA-256 (`crypto.subtle.digest`). Send it as lower-case hex.
3. `POST /v1/marketing/assets/uploads`: send study, name, mime type, size, checksum and the
   authorization block. Keep `asset_id` and `upload`.
4. `PUT` the file to `upload.url`. Send exactly `upload.headers` and nothing else. The browser
   sets `Content-Length` itself, and it must equal `size_bytes`.
5. `POST /v1/marketing/assets/{asset_id}/complete`.
6. Poll `GET /v1/marketing/assets/uploads/{asset_id}` until the status is `READY` or `REJECTED`.

## Errors while uploading

- `409 UPLOAD_NOT_RECEIVED` (retryable): the PUT has not reached storage yet. Wait, then call
  complete again.
- Expired URL: the PUT fails or `upload.expires_at` has passed. Call
  `POST /v1/marketing/assets/{asset_id}/upload-url` for a new target, then PUT again.
- `409 UPLOAD_NOT_PENDING`: the upload already moved on. Read its current status and show that.
- `403 FORBIDDEN`: only the person who started the upload can complete it.
- `422 UPLOAD_NOT_ACCEPTED`: `details.field` names the bad field. Use the limits from step 1.
- `422 VALIDATION_FAILED` on `authorization.usage_authorization_ref`: the reference is a record
  id (letters, digits and `. _ : / -`, starting with a letter or digit). Surrounding spaces are
  removed; blank values and spaces inside are refused.
- `429 UPLOAD_LIMIT_REACHED` (retryable): the tenant has the most uploads in progress it may
  have. Uploads waiting for inspection count, and so do pending ones while their URL is valid.
  Show "too many uploads in progress" and let the user try again later; also possible on
  `POST /assets/{asset_id}/upload-url` for an upload whose URL expired.

## Reasons and polling

- `reason.retryable: true` (for example `SCAN_UNAVAILABLE`, `STORAGE_UNAVAILABLE`): the system
  retries by itself. Keep polling and show a "still checking" state.
- `reason.retryable: false` (for example `MALWARE_DETECTED`, `TYPE_MISMATCH`): the upload is
  final. Show `reason.message` and offer a new file. Do not poll.
- `scan.simulated: true` means no real malware scan ran. Show a visible "simulated scan" label
  next to the status. Never hide it.

## Secrets and URLs

- `upload.url` and every `download_url` are secret. Do not log, store, cache or put them in
  analytics.
- Validate a URL is `https:` before you use it. Reject anything else.

## Authorization

- `authorization.status` is a declaration record. It is not verification of consent.
- `MISSING` and `EXPIRED` are meant to block approval and publishing. Those gates come with
  NRC-12 and NRC-14; until then, show the status so reviewers can see it.
- Do not add "consent on file" checkboxes. They are not proof. Collect
  `usage_authorization_ref` instead (the signed release or licence record ID, not the document).
- `identifiable_person` is the uploader's declaration: `yes`, `no` or `unknown`.

## Nullable fields

- `creative_id` and `creative_version` can be null in `AssetRead` and `MediaItem`. Uploads not
  tied to a draft have no creative. Handle null before showing or linking them.
- `study_id`, `name` and `authorization` are null for generated images.

## Examples in uploads.json

- `upload_limits`
- `upload_created`
- `upload_quarantined`
- `upload_scanning`
- `upload_ready_simulated_scan`
- `upload_rejected_malware`
- `upload_rejected_forged_type`
- `upload_waiting_scanner`
- `upload_authorization_missing`
- `upload_authorization_expired`
- `uploads_page`
- `ready_upload_as_asset`
- `media_page_with_upload`
- `error_upload_not_accepted`
- `error_upload_not_received`
- `error_upload_not_pending`
- `error_forbidden_not_uploader`
- `error_upload_limit_reached`
