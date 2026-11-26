"""Text judge (NRC-7): the simulator's scenarios, the answer models and the settings checks."""

from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.validations import RuleResultValue
from app.services.providers.base import (
    ProviderConfigError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.services.providers.bedrock_judge import BedrockTextJudge
from app.services.providers.text_judge import (
    CLAUDE_HAIKU_5_5_MODEL_ID,
    FAKE_RULE_ID,
    NOVA_LITE_MODEL_ID,
    SIMULATED_JUDGE_MODEL_ID,
    SUPPORTED_BEDROCK_JUDGE_MODEL_IDS,
    UNKNOWN_EVIDENCE_KEY,
    JudgeAnswer,
    JudgeFinding,
    JudgeRequest,
    SimulatedJudgeScenario,
    SimulatedTextJudge,
    missing_live_judge_settings,
    text_judge_from_settings,
)

REQUIRED: dict[str, Any] = {
    "database_url": "postgresql+asyncpg://u:p@localhost/db",
    "cognito_region": "ap-south-1",
    "cognito_user_pool_id": "ap-south-1_TESTPOOL",
    "cognito_client_ids": ("c",),
    "required_scope": "marketing/api",
}


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def _request(text: str = "Buy now, limited offer") -> JudgeRequest:
    return JudgeRequest(
        system="system",
        user="user",
        rule_ids=("R-1", "R-2", "R-3"),
        text=text,
        max_output_tokens=500,
    )


# Simulator


async def test_default_scenario_passes_every_rule_and_is_labelled_simulated() -> None:
    judge = SimulatedTextJudge()

    reply = await judge.judge(_request())

    answer = JudgeAnswer.model_validate_json(reply.raw)
    assert [r.rule_id for r in answer.results] == ["R-1", "R-2", "R-3"]
    assert all(r.result is RuleResultValue.PASS for r in answer.results)
    assert all(r.explanation.startswith("SIMULATED: ") for r in answer.results)
    assert all(not r.findings and not r.evidence_node_keys for r in answer.results)
    assert reply.model_id == judge.model_id == SIMULATED_JUDGE_MODEL_ID == "simulator:judge-v1"
    assert reply.is_simulated
    assert judge.is_simulated
    assert reply.input_tokens is None
    assert reply.output_tokens is None
    assert reply.latency_ms == 0


async def test_fail_first_fails_the_first_rule_on_the_first_word() -> None:
    reply = await SimulatedTextJudge(SimulatedJudgeScenario.FAIL_FIRST).judge(_request())

    first, *rest = JudgeAnswer.model_validate_json(reply.raw).results
    assert first.rule_id == "R-1"
    assert first.result is RuleResultValue.FAIL
    assert [(f.start, f.end, f.excerpt) for f in first.findings] == [(0, 3, "Buy")]
    assert [r.result for r in rest] == [RuleResultValue.PASS, RuleResultValue.PASS]


async def test_fail_first_offsets_are_code_points_not_bytes() -> None:
    text = "\U0001f600 hello world"
    reply = await SimulatedTextJudge(SimulatedJudgeScenario.FAIL_FIRST).judge(_request(text))

    (finding,) = JudgeAnswer.model_validate_json(reply.raw).results[0].findings
    # The emoji is one code point (4 UTF-8 bytes, 2 UTF-16 units), so "hello" starts at 2.
    assert (finding.start, finding.end, finding.excerpt) == (2, 7, "hello")
    assert text[finding.start : finding.end] == finding.excerpt


async def test_fail_first_without_a_word_fails_without_findings() -> None:
    reply = await SimulatedTextJudge(SimulatedJudgeScenario.FAIL_FIRST).judge(_request("  !? "))

    first = JudgeAnswer.model_validate_json(reply.raw).results[0]
    assert first.result is RuleResultValue.FAIL
    assert first.findings == []


async def test_malformed_returns_a_truncated_reply_that_does_not_parse() -> None:
    reply = await SimulatedTextJudge(SimulatedJudgeScenario.MALFORMED).judge(_request())

    assert reply.stop_reason == "max_tokens"
    assert reply.is_simulated
    with pytest.raises(ValidationError, match="Invalid JSON"):
        JudgeAnswer.model_validate_json(reply.raw)


async def test_fake_citation_cites_unknown_evidence_and_answers_for_an_unasked_rule() -> None:
    reply = await SimulatedTextJudge(SimulatedJudgeScenario.FAKE_CITATION).judge(_request())

    results = JudgeAnswer.model_validate_json(reply.raw).results
    assert [r.rule_id for r in results] == ["R-1", "R-2", "R-3", FAKE_RULE_ID]
    assert all(r.result is RuleResultValue.PASS for r in results)
    assert all(r.evidence_node_keys == [UNKNOWN_EVIDENCE_KEY] for r in results[:3])
    assert UNKNOWN_EVIDENCE_KEY == "NOT-IN-CONTEXT"


async def test_timeout_scenario_is_unavailable_and_throttled_is_rate_limited() -> None:
    with pytest.raises(ProviderUnavailableError, match="SIMULATED judge") as unavailable:
        await SimulatedTextJudge(SimulatedJudgeScenario.TIMEOUT).judge(_request())
    with pytest.raises(ProviderRateLimitedError, match="SIMULATED judge") as limited:
        await SimulatedTextJudge(SimulatedJudgeScenario.THROTTLED).judge(_request())

    assert unavailable.value.retryable
    assert limited.value.retryable


async def test_a_request_without_rules_is_rejected_not_a_crash() -> None:
    request = JudgeRequest(system="s", user="u", rule_ids=(), text="t", max_output_tokens=500)

    with pytest.raises(ProviderRejectedError, match="names no rules"):
        await SimulatedTextJudge().judge(request)


# Answer models


def test_answer_models_refuse_unknown_fields_and_out_of_range_values() -> None:
    good = {"rule_id": "R-1", "result": "PASS", "explanation": "ok"}

    JudgeAnswer.model_validate({"results": [good]})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JudgeAnswer.model_validate({"results": [{**good, "confidence": 1}]})
    with pytest.raises(ValidationError, match="at least 1 item"):
        JudgeAnswer.model_validate({"results": []})
    with pytest.raises(ValidationError, match="Input should be"):
        JudgeAnswer.model_validate({"results": [{**good, "result": "MAYBE"}]})
    with pytest.raises(ValidationError, match="greater than 0"):
        JudgeFinding(start=0, end=0, excerpt="x")
    with pytest.raises(ValidationError, match="at most 500"):
        JudgeFinding(start=0, end=1, excerpt="x" * 501)


# Settings and the factory


def test_judge_is_off_by_default_and_the_factory_gives_none() -> None:
    settings = _settings()

    assert settings.validation_model_mode == "off"
    assert text_judge_from_settings(settings) is None
    assert missing_live_judge_settings(settings) == []


def test_factory_gives_the_simulator_locally() -> None:
    judge = text_judge_from_settings(_settings(app_env="local", validation_model_mode="simulator"))

    assert isinstance(judge, SimulatedTextJudge)
    assert judge.scenario is SimulatedJudgeScenario.PASS_ALL


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_simulator_is_refused_outside_local_and_demo(app_env: str) -> None:
    with pytest.raises(ValidationError, match="VALIDATION_MODEL_MODE=simulator cannot be used"):
        _settings(app_env=app_env, validation_model_mode="simulator")


def test_simulator_is_allowed_in_demo_and_live_in_staging() -> None:
    _settings(app_env="demo", validation_model_mode="simulator")
    _settings(
        app_env="staging",
        validation_model_mode="live",
        bedrock_validation_model_id=NOVA_LITE_MODEL_ID,
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "missing BEDROCK_VALIDATION_MODEL_ID"),
        ({"bedrock_validation_model_id": ""}, "missing BEDROCK_VALIDATION_MODEL_ID"),
        ({"bedrock_validation_model_id": "amazon.nova-micro-v1:0"}, "not a supported model"),
    ],
)
def test_live_mode_lists_what_it_lacks_and_the_factory_refuses(
    overrides: dict[str, str], expected: str
) -> None:
    settings = _settings(validation_model_mode="live", **overrides)

    problems = missing_live_judge_settings(settings)
    assert len(problems) == 1
    assert expected in problems[0]
    with pytest.raises(ProviderConfigError, match=expected):
        text_judge_from_settings(settings)


def test_live_mode_builds_the_bedrock_adapter_for_a_supported_model() -> None:
    settings = _settings(
        validation_model_mode="live", bedrock_validation_model_id=CLAUDE_HAIKU_5_5_MODEL_ID
    )

    judge = text_judge_from_settings(settings)

    assert isinstance(judge, BedrockTextJudge)
    assert judge.model_id == CLAUDE_HAIKU_5_5_MODEL_ID
    assert not judge.is_simulated


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("validation_model_timeout_seconds", 0),
        ("validation_model_timeout_seconds", 121),
        ("validation_model_max_output_tokens", 255),
        ("validation_model_max_output_tokens", 8001),
    ],
)
def test_judge_limits_are_bounded(field: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: bad})


def test_candidate_set_has_the_six_models_to_evaluate() -> None:
    assert len(SUPPORTED_BEDROCK_JUDGE_MODEL_IDS) == 6
