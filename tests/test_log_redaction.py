# NRC-25: prompts, captions, tokens and rejected input never reach the logs or error bodies.

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final
from uuid import uuid4

import httpx

from app.core.logging import JsonFormatter
from tests.conftest import AuthHeaders, World

API: Final = "/v1/marketing"
PROMPT: Final = "Invite adults with prompt-marker-7f3a to learn about the study."
EDITED_PROMPT: Final = "Invite adults with edit-marker-2b9e to learn about the study."
CAPTION: Final = "Learn more near you, caption-marker-91c2."
PROBE: Final = "log-capture-probe"


@contextmanager
def _app_logs() -> Iterator[io.StringIO]:
    # The production formatter on the root logger, at DEBUG so stricter than the INFO default.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


async def test_sensitive_values_stay_out_of_logs_and_errors(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    headers = auth("creator-a", world.a.tenant_id)
    unsigned_bearer = f"token-marker-{uuid4().hex}"
    draft = {
        "study_id": str(world.a.study_id),
        "prompt": PROMPT,
        "caption": CAPTION,
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
    }
    with _app_logs() as logs:
        created = await client.post(
            f"{API}/creatives", headers={**headers, "Idempotency-Key": uuid4().hex}, json=draft
        )
        creative = created.json()
        validated = await client.post(
            f"{API}/validations",
            headers={**headers, "Idempotency-Key": uuid4().hex},
            json={
                "creative_id": creative["id"],
                "creative_version": creative["version"],
                "input_kind": "brief",
            },
        )
        errors = [
            await client.patch(
                f"{API}/creatives/{creative['id']}",
                headers={**headers, "If-Match": '"7"'},
                json={"current_prompt": EDITED_PROMPT},
            ),
            await client.post(
                f"{API}/creatives",
                headers={**headers, "Idempotency-Key": uuid4().hex},
                json={**draft, "prompt": EDITED_PROMPT, "language": "xx"},
            ),
            await client.get(f"{API}/me", headers={"Authorization": f"Bearer {unsigned_bearer}"}),
        ]
        logging.getLogger("tests.probe").warning(PROBE)

    assert created.status_code == 201, created.text
    assert validated.status_code == 202, validated.text
    assert [response.status_code for response in errors] == [412, 422, 401]
    captured = logs.getvalue()
    # Proves the capture is wired to the app's logging; otherwise the checks below prove nothing.
    assert PROBE in captured
    secrets = (
        "prompt-marker",
        "edit-marker",
        "caption-marker",
        "token-marker",
        headers["Authorization"],
    )
    for secret in secrets:
        assert secret not in captured, secret
        for response in errors:
            assert secret not in response.text, (response.status_code, secret)
