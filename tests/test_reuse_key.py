"""The validation reuse key (NRC-7): a stored result is reused only for an equal key, so every
input that can change the result must change the key, and nothing that can't should be in it.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from app.core.config import Settings, get_settings
from app.models.enums import Channel, Language, PostFormat, ValidationInputKind
from app.schemas.policies import PolicyBundleRead
from app.schemas.tenancy import StudyFacts
from app.services import judging, rules, validations
from app.services.providers.text_judge import NOVA_LITE_MODEL_ID
from scripts.seed_demo import DEMO_POLICY_NAME, SYNTHETIC_FACTS

BUNDLE_ID = UUID("00000000-0000-4000-8000-000000000001")
SNAPSHOT_ID = UUID("00000000-0000-4000-8000-000000000002")
DAY = date(2026, 11, 26)

BASE_TARGET = validations.Target(
    prompt="Join a study for adults with asthma.",
    caption="Learn more about the study.",
    language=Language.EN,
    channel=Channel.FACEBOOK,
    post_format=PostFormat.FEED_IMAGE,
    facts=StudyFacts.model_validate(SYNTHETIC_FACTS),
    snapshot_id=SNAPSHOT_ID,
    snapshot_version=1,
)


def _policy(bundle_id: UUID = BUNDLE_ID, version: int = 1) -> PolicyBundleRead:
    # Only the bundle id and version reach the key.
    return PolicyBundleRead(
        policy_bundle_id=bundle_id,
        version=version,
        name=DEMO_POLICY_NAME,
        source_ref="synthetic:test",
        is_synthetic=True,
        approved_by="synthetic-fixture",
        approved_at=datetime(2026, 11, 1, tzinfo=UTC),
        rules=[],
    )


def _settings(**update: Any) -> Settings:
    return get_settings().model_copy(
        update={
            "validation_model_mode": "live",
            "bedrock_validation_model_id": NOVA_LITE_MODEL_ID,
            **update,
        }
    )


def _key(
    kind: ValidationInputKind = ValidationInputKind.BRIEF,
    target: validations.Target = BASE_TARGET,
    policy: PolicyBundleRead | None = None,
    settings: Settings | None = None,
    day: date = DAY,
) -> str:
    return validations.reuse_key(
        kind,
        target.text(kind),
        target,
        policy or _policy(),
        settings or _settings(),
        day,
    )


# One case per part of the key. Language, channel and format have one member each today, so a
# raw value stands in for the next one.
CHANGES: dict[str, Callable[[], str]] = {
    "kind": lambda: _key(kind=ValidationInputKind.TEXT),
    "text": lambda: _key(target=replace(BASE_TARGET, prompt="Join a study for adults.")),
    "channel": lambda: _key(target=replace(BASE_TARGET, channel="INSTAGRAM")),  # type: ignore[arg-type]  # no second member yet
    "post_format": lambda: _key(target=replace(BASE_TARGET, post_format="STORY")),  # type: ignore[arg-type]  # no second member yet
    "language": lambda: _key(target=replace(BASE_TARGET, language="ES")),  # type: ignore[arg-type]  # no second member yet
    "policy_bundle id": lambda: _key(
        policy=_policy(bundle_id=UUID("00000000-0000-4000-8000-000000000003"))
    ),
    "policy_bundle version": lambda: _key(policy=_policy(version=2)),
    "study_snapshot id": lambda: _key(
        target=replace(BASE_TARGET, snapshot_id=UUID("00000000-0000-4000-8000-000000000004"))
    ),
    "study_snapshot version": lambda: _key(target=replace(BASE_TARGET, snapshot_version=2)),
    "judge model": lambda: _key(
        settings=_settings(bedrock_validation_model_id="mistral.mistral-large-3-675b-instruct")
    ),
    "judge off": lambda: _key(settings=_settings(validation_model_mode="off")),
    "day": lambda: _key(day=DAY + timedelta(days=1)),
    "retrieval max_hops": lambda: _key(settings=_settings(policy_retrieval_max_hops=2)),
    "retrieval max_nodes": lambda: _key(settings=_settings(policy_retrieval_max_nodes=100)),
    "retrieval top_k": lambda: _key(settings=_settings(policy_retrieval_top_k=3)),
    "retrieval max_statements": lambda: _key(
        settings=_settings(policy_retrieval_max_statements=10)
    ),
    # The worker also searches by vector when it has an embedder.
    "embedding mode": lambda: _key(settings=_settings(embedding_mode="live")),
    "embedding model": lambda: _key(
        settings=_settings(
            embedding_mode="live", bedrock_embedding_model_id="amazon.titan-embed-text-v2:0"
        )
    ),
}


@pytest.mark.parametrize("change", list(CHANGES))
def test_each_input_that_can_change_the_result_changes_the_key(change: str) -> None:
    assert CHANGES[change]() != _key()


def test_a_new_evaluation_or_prompt_version_changes_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _key()
    monkeypatch.setattr(rules, "EVALUATION_VERSION", f"{rules.EVALUATION_VERSION}-next")
    evaluation = _key()
    monkeypatch.setattr(
        judging, "PROMPT_TEMPLATE_VERSION", f"{judging.PROMPT_TEMPLATE_VERSION}-next"
    )

    assert len({base, evaluation, _key()}) == 3


def test_every_part_of_the_key_has_a_test_above(monkeypatch: pytest.MonkeyPatch) -> None:
    # A new part added to reuse_key must get its own case in CHANGES (or the version test).
    seen: list[dict[str, Any]] = []
    real_dumps = json.dumps

    def capture(parts: dict[str, Any], **kwargs: Any) -> str:
        seen.append(parts)
        return real_dumps(parts, **kwargs)

    monkeypatch.setattr(json, "dumps", capture)
    _key()

    [parts] = seen
    tested = {name.split()[0] for name in CHANGES} | {
        "evaluation",
        "prompt",
        "judge",
        "retrieval",
        "embedding",
    }
    assert set(parts) <= tested
    assert set(parts["retrieval"]) == {
        name.split()[1] for name in CHANGES if name.startswith("retrieval ")
    }


@pytest.mark.parametrize(
    "unrelated",
    [{"validation_model_timeout_seconds": 10}, {"bedrock_max_retries": 0}],
)
def test_settings_that_cannot_change_the_result_leave_the_key_alone(
    unrelated: dict[str, object],
) -> None:
    # Timeouts and client retries change whether a call succeeds, never what a stored result is.
    assert _key(settings=_settings(**unrelated)) == _key()
