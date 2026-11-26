---
paths:
  - "tests/**"
---

# Tests

- Tests run against real Postgres (`marketing_test`); the `connection` fixture wraps each test in a
  transaction that is rolled back. Don't add SQLite or mock the database.
- Auth: use the `auth` fixture, which mints RS256 tokens against a static JWKS. Never add a bypass,
  a shortcut token or a flag that weakens verification.
- Seed data with the existing helpers in `tests/conftest.py` (`seed_tenant`, the `world`
  fixture). Fixture data is synthetic and labelled as such.
- `filterwarnings = error` and pytest `strict = true`: fix warnings, don't silence them.
- Cover every behaviour change, including the failure paths: wrong tenant (404), missing
  capability (403), stale version (412), idempotency replay and key reuse (409), validation (422).
- Concurrency: use `asyncio.gather` for parallel duplicates.
- Run the narrow test first (`uv run pytest -q tests/test_x.py::test_name`), then the full suite.
- Contract: `tests/test_contract.py` fails when `contracts/openapi.json` is stale; regenerate it
  with `uv run python -m scripts.export_openapi` instead of editing it.
