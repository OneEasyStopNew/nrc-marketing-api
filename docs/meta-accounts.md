# Connecting Facebook Pages and Instagram accounts (NRC-13)

## Flow

1. A tenant admin clicks Connect. The web app calls `POST /v1/marketing/integrations/meta/connect`
   and sends the browser to the returned `authorization_url` (a top-level navigation, not a popup
   fetch).
2. Meta asks the person to log in and choose Pages, then sends the browser to this API's
   `GET /webhooks/meta/oauth/callback`. API Gateway routes `/webhooks/*` without Cognito, so the
   OAuth `state` is the only credential there: `<tenant>.<secret>`, single use, 10 minutes. The
   database keeps only the secret's SHA-256.
3. The API swaps the code for a long-lived user token, reads the granted permissions and the Pages
   (with their linked Instagram accounts), and redirects the browser to
   `<WEB_APP_URL><return_path>?meta_connection=<id>&meta_connect=select_accounts`. Failures land
   with `meta_connect=failed` (see `error_code` on the connection).
4. The web app reads `GET /integrations/meta/connections/{id}` and shows the candidates. Nothing is
   bound until the same admin picks accounts with `POST .../connections/{id}/accounts`.
5. Only the encrypted Page token is kept for each bound account. The user token is deleted when
   binding completes; unfinished logins keep it encrypted until they expire.

Reconnect (`POST /channel-accounts/{id}/reconnect`) runs the same login and refreshes that one
account directly (`meta_connect=reconnected`). Disconnect keeps the account listed as
`DISCONNECTED`, deletes its token and stops new publishing. Posts that are already live are not
removed.

## Account states

| State | Can publish | Meaning |
|---|---|---|
| CONNECTED | yes | Full grant, Meta app has Advanced Access (`META_APP_MODE=live`) |
| TEST_ONLY | yes, test accounts only | Meta app in development mode; only people with a role on the app |
| PENDING_APP_REVIEW | no | Permissions submitted for App Review (`META_APP_MODE=in_review`) |
| PERMISSION_MISSING | no | A permission was declined, or the person lacks the Page's Create content task |
| EXPIRED / REVOKED | no | Meta no longer accepts the token (`POST /channel-accounts/{id}/check`) |
| DISCONNECTED | no | Disconnected in Marketing |

## Meta app prerequisites (owner: whoever holds the Meta Business portfolio)

- A Meta app of type Business with the Facebook Login for Business product, and the content
  management use case for Pages and Instagram.
- Valid OAuth Redirect URI: `https://<api host>/webhooks/meta/oauth/callback` for each
  environment (staging and production are separate entries).
- Permissions, and nothing broader (no inbox, ads or insights):
  `business_management`, `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`,
  `instagram_basic`, `instagram_content_publish`.
  Meta's docs also show `instagram_content_publishing` for the last one; use the exact name the
  app dashboard lists.
- Instagram accounts must be professional (business or creator) and linked to a Facebook Page the
  person can manage.
- Until App Review and Business Verification pass, the app works only for people with a role on
  it (Standard Access). Add the test Page admins as app testers and keep `META_APP_MODE=development`.
- App id and secret go to Secrets Manager; the API reads them from its environment
  (`META_APP_ID`, `META_APP_SECRET`). `TOKEN_ENCRYPTION_KEY` is a separate secret.

## Capabilities

`capabilities` on each account says what this app can publish there now, with a `version` that
approval and publication records keep. Values Meta does not state are `null`, not guessed.
Instagram image limits (JPEG, up to 8 MB, 4:5 to 1.91:1, 2,200-character captions) come from
Meta's IG User Media reference. Instagram's daily publishing cap is read per account from
`content_publishing_limit` when publishing is built (NRC-15), because Meta's docs give two
different numbers.

Graph API version: `v26.0` (current as of Nov 2026). Bump `META_GRAPH_VERSION` and
`CAPABILITY_VERSION` together after checking Meta's changelog.
