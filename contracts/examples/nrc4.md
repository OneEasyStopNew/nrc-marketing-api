# Policy sources and bundles: frontend guide (NRC-4, for Praneeth)

Examples are in `nrc4.json`. All data is synthetic: every source and bundle carries
`is_synthetic`, and it stays `true` until the IPB corpus exists. Show it as a visible label.
All paths are under `/v1/marketing`, tag `policy`.

## The flow

1. Upload the policy file with `POST /policy-sources/uploads` (see `policy_uploads.json`) and
   poll the upload until `READY`.
2. `POST /policy-sources` with `upload_id`, `title`, `source_ref`, `language`, `is_synthetic`.
   `201` = new source, `200` = the same file was registered before (same source returned; if its
   extraction had `FAILED`, it is queued again, which is how to retry). Both carry `Location`. Poll `GET /policy-sources/{id}` until `extraction_state` is `EXTRACTED`
   (or `FAILED`, see `extraction_error_code`), then `GET /policy-sources/{id}/clauses`.
3. `POST /policy-bundles` (`name`, `source_ref`, `is_synthetic`, optional `based_on_bundle_id` to
   copy a bundle's graph and review into a new draft). The bundle starts as `DRAFT`.
4. Edit the draft: `POST .../nodes`, `.../rules`, `.../edges`, `.../rules/{rule_id}/wordings`,
   `PATCH .../nodes/{node_id}`, `PATCH .../rules/{rule_id}`. A rule's `{rule_id}` in paths is
   `PolicyRuleRead.id` (the record id), not the `rule_id` code. Every new or edited item is
   `PROPOSED`.
5. Review: `POST .../review-decisions` with `item_kind` (`node`, `edge`, `wording`), `item_id`,
   `decision` (`APPROVED` or `REJECTED`), `note`. Returns the bundle. An approved edge needs a
   source clause or a note.
6. `GET .../checks` lists what would block approval (`{issues: [{code, subject, message}]}`).
   Empty means ready.
7. `POST .../transitions` with `{command, reason}`: `submit` (DRAFT to REVIEW), then `approve` or
   `reject` or `return_to_draft` (from REVIEW). After approval the worker builds the index
   (`APPROVED`, `BUILDING`, `READY`); poll the bundle. If the build fails the bundle is back to
   `APPROVED` with `last_error_code`; `build` queues it again. Then `activate` (READY to ACTIVE).
   `retire` (ACTIVE to RETIRED) and `rollback` (RETIRED to ACTIVE, only for a bundle that was
   active before) complete the cycle. Activation retires the previously active bundle.
8. `GET .../audit` shows who did what, newest first.

## If-Match and ETag

- Every edit, review decision and transition needs `If-Match: "<revision>"`, the bundle's
  `revision` (also its `ETag`). Every response to a change carries the new `ETag`; send it with
  the next change. `GET /policy-bundles/{id}` returns the current one.
- Missing or malformed header: `428 PRECONDITION_REQUIRED`.
- Stale: `412 VERSION_CONFLICT` with `details.current_version`. Reload, reapply.

## Who sees which actions

| Role | Can do |
|---|---|
| `policy_admin` | Read everything, register sources, create bundles, edit, review items, `submit`, `return_to_draft`, and see proposed rules (`include_proposals=true`) |
| `policy_approver` | Read everything, `approve`, `reject`, `return_to_draft`, `build`, `activate`, `rollback`, `retire` |
| `creator`, `reviewer`, others | No policy routes: `403 FORBIDDEN` |

`policy_admin` cannot approve or activate, and `policy_approver` cannot edit: `403 FORBIDDEN`.
A person may hold both roles (the four-eyes rule is not decided yet). Another tenant's ids give
`404 NOT_FOUND`. Hide actions the role cannot use, but rely on the API's answer.

## Rules view

`GET /policy-bundles/{id}/rule-records` (the editor view; the older `GET .../{id}/rules` for
active or retired bundles is unchanged). It leaves out `PROPOSED` rules and wordings. Managers
pass `include_proposals=true` to see them; anyone else gets `403`.
`GET .../nodes` and `.../edges` list every row with `review_state` and `origin`
(`HUMAN`, `IMPORT`, `LLM_SIMULATED`, `LLM`); label `LLM_SIMULATED` as simulated.

## Errors

- `409 POLICY_BUNDLE_STATE`: `details.state` and `details.allowed`. Refresh and show the state.
- `409 POLICY_BUNDLE_HAS_ISSUES`: `details.issues` (same shape as `checks`). Show them.
  Issue codes: `EMPTY_BUNDLE`, `UNREVIEWED_ITEM`, `MISSING_PROVENANCE`,
  `EDGE_TO_UNAPPROVED_NODE`, `SUPERSEDES_CYCLE`, `SUPERSEDES_CONFLICT`,
  `CONTRADICTORY_WORDING`, `UNRESOLVED_EXCEPTION`, `UNSUPPORTED_LANGUAGE`,
  `DUPLICATE_OBLIGATION`, `EXACT_WORDING_NOT_IN_SOURCE`, and `INDEX_STALE` on activation.
- `409 POLICY_ITEM_INVALID`: the request is well-formed but the item is not allowed (for example
  a missing `reason`, a duplicate edge); read `message`.
- `409 POLICY_SOURCE_UPLOAD_NOT_READY`: the upload is not `READY` yet (`details.status`).
- `reason` is required for `reject`, `activate`, `rollback` and `retire`.

## Paging

`GET /policy-sources` and `.../audit`: `limit` (1 to 50) and `before` = the previous
`next_cursor`; a page may exceed `limit` when rows tie. `GET /policy-bundles`: `limit` and
`before_version` = the previous `next_cursor`, optional `state` filter.

## Examples in nrc4.json

- `policy_source`
- `policy_clauses`
- `policy_bundle_detail`
- `policy_bundle_active`
- `policy_rules`
- `policy_checks_with_issue`
- `policy_audit_page`
- `error_policy_bundle_has_issues`
