# Demo branch: validation flow (NRC-4, NRC-5, NRC-7 together)

`demo/NRC-29-validation-flow` combines three open PRs so the frontend can build against one
contract before they reach main: #26 (NRC-5 retrieval and evidence), #31 (NRC-7 validation as a
worker job) and #30 (NRC-4 policy ingestion and review). It is a throwaway bridge: never open a PR
from it and never merge it. Once the PRs are on main, switch to main; the contract is the same,
because the merges add nothing new.

## The contract

One file: `contracts/openapi.json` on this branch. Sample responses: `contracts/examples/nrc4.json`
and `nrc4.md` (policy admin), `nrc5.json` (evidence), `nrc7.json` (validation),
`policy_uploads.json` (policy file upload).

## Run it locally

Follow "Run locally" in `README.md` (Docker, `uv sync`, `alembic upgrade head`), with dev tokens
("Dev tokens" section) so no Cognito is needed. Then:

```powershell
# Synthetic tenant, approved study, and an ACTIVE synthetic policy (all labelled SYNTHETIC)
uv run python -m scripts.seed_demo --tenant-name "Demo" `
    --member dev-creator=creator --member dev-reviewer=reviewer `
    --member dev-approver=policy_approver --member dev-admin=policy_admin
uv run python -m scripts.dev_token issue --sub dev-creator     # one token per user, 12 hours

uv run uvicorn app.main:create_app --factory --port 8000       # API
uv run python -m app.workers.connector                         # runs every validation job
uv run python -m app.workers.media                             # policy file uploads (optional)
uv run python -m app.workers.policy                            # clause extraction, index builds (optional)
```

Without the connector worker, `POST /validations` stays QUEUED forever.

Settings for a local demo (names only, set them in your local settings file):
- `VALIDATION_MODEL_MODE=simulator`: the model check answers with labelled SIMULATED results and
  passes every meaning-based rule. `off` leaves those rules UNABLE_TO_ASSESS, so most briefs come
  back NEEDS_REVIEW. `live` calls Amazon Bedrock (`BEDROCK_VALIDATION_MODEL_ID`, Mistral Large 3
  for the demo) and needs AWS access.
- `EMBEDDING_MODE=simulator` (or `off` for full-text search only); `live` uses
  `BEDROCK_EMBEDDING_MODEL_ID` (Titan Text Embeddings V2).
- `MALWARE_SCAN_MODE=simulator` and `POLICY_SOURCE_BUCKET` only if you test policy file uploads.

## Screens and endpoints (all under `/v1/marketing`)

| Screen | Endpoints |
|---|---|
| Validate a brief or caption | `POST /validations` returns 202 QUEUED; poll `GET /validations/{id}` until SUCCEEDED or FAILED |
| Evidence on the results page | `GET /validations/{id}/context`, `GET /policy-bundles/{id}/rules` |
| Policy files | `POST /policy-sources/uploads` → `.../upload-url` → `.../complete`, then `POST /policy-sources`; `GET /policy-sources`, `.../{id}`, `.../{id}/clauses` |
| Policy bundles | `POST/GET /policy-bundles`, `GET .../{id}`, `.../rule-records`, `.../nodes`, `.../edges`, `.../checks`, `.../audit`; edits: `POST .../nodes`, `.../edges`, `.../rules`, `.../rules/{rule_id}/wordings`, `PATCH .../nodes/{id}`, `.../rules/{id}` |
| Review and lifecycle | `POST .../review-decisions`, `POST .../transitions` |

## Who sees which buttons

| Action | Role |
|---|---|
| Create and edit bundle items, submit for review, review ordinary items | `policy_admin` |
| Approve or reject a bundle; approve or reject exceptions, disclosures and study-wording pointers | `policy_approver` |
| Build, activate, roll back, retire | `policy_admin` |
| Return to draft | either |
| Validate briefs and captions | creator, reviewer |

A caller without the capability gets 403. The route descriptions in the contract list each one.

## Contract changes since the PR samples

- Rule results can be `NEEDS_REVIEW` (a conflict in the policy context), next to PASS, FAIL,
  NOT_APPLICABLE and UNABLE_TO_ASSESS.
- `POST /validations` can return 429 `VALIDATION_LIMIT_REACHED` (too many validations queued for
  the tenant; retryable).
- Each finding can carry `suggested_correction` (a wording fix, at most 500 characters). There is no
  full rewrite of the brief yet.
- Policy: transitions and item reviews are split by capability (table above); `post_formats` on
  nodes and rules; `channels` and `languages` are enums now, not free text; audit details include
  `actor_roles`; `WordingKind` gains `STUDY_APPROVED_WORDING` (points to the study's approved
  disclosure or phrase instead of holding text).

## Sample briefs (synthetic)

Expected results against the seeded policy. Rules REC-07 (research and voluntary disclosure) and
REC-08 (approved study facts) are checked in code; the others need the model check.

| Brief | Expected |
|---|---|
| Adults aged 18 to 75 with high blood pressure may be able to join a research study of an investigational medicine. This is a research study. Taking part is voluntary. | PASS (simulator or live) |
| Adults aged 18 to 75 with high blood pressure may be able to join a research study of an investigational medicine. | FAIL on REC-07: the required disclosure is missing |
| Adults aged 18 to 90 with high blood pressure may be able to join a research study. This is a research study. Taking part is voluntary. | FAIL on REC-08: the age range isn't the approved one |
| This investigational pill lowers blood pressure quickly. This is a research study. Taking part is voluntary. | FAIL on REC-01 with the live model; the simulator passes it |
| Why settle for ordinary medication when you could try something more advanced? This is a research study. Taking part is voluntary. | FAIL on REC-02 with the live model; NEEDS_REVIEW with the model off |

These expectations come from the rules and the labelled evaluation cases; they haven't been run
end to end on this branch yet.
