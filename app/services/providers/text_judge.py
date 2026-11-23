"""Contract between text validation (NRC-7) and the LLM judge that checks text against rules.

No boto3 here: the API process imports this module (settings checks, the contract) but never the
Bedrock adapter, which the factory imports only for live mode. This step holds the transport only;
building the prompt and verifying the answer come in later steps.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.schemas.validations import RuleResultValue
from app.services.providers.base import (
    ProviderConfigError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
)

# Candidate models to evaluate, not a choice: Vishnu chooses after the evaluation (NRC-7 task
# file, _context/NRC-7-model-candidates.md). The Claude models are not callable in-region in
# us-east-1, so they use the US inference profile id.
NOVA_LITE_MODEL_ID: Final = "amazon.nova-lite-v1:0"
NOVA_PRO_MODEL_ID: Final = "amazon.nova-pro-v1:0"
CLAUDE_HAIKU_5_5_MODEL_ID: Final = "us.anthropic.claude-haiku-5-5"
CLAUDE_SONNET_5_5_MODEL_ID: Final = "us.anthropic.claude-sonnet-5-5"
MISTRAL_LARGE_3_MODEL_ID: Final = "mistral.mistral-large-3-675b-instruct"
GPT_OSS_120B_MODEL_ID: Final = "openai.gpt-oss-120b-1:0"
SUPPORTED_BEDROCK_JUDGE_MODEL_IDS: Final = frozenset(
    {
        NOVA_LITE_MODEL_ID,
        NOVA_PRO_MODEL_ID,
        CLAUDE_HAIKU_5_5_MODEL_ID,
        CLAUDE_SONNET_5_5_MODEL_ID,
        MISTRAL_LARGE_3_MODEL_ID,
        GPT_OSS_120B_MODEL_ID,
    }
)

SIMULATED_JUDGE_MODEL_ID: Final = "simulator:judge-v1"
FAKE_RULE_ID: Final = "FAKE-01"
UNKNOWN_EVIDENCE_KEY: Final = "NOT-IN-CONTEXT"
_SIMULATED_PREFIX: Final = "SIMULATED: "
_WORD = re.compile(r"\w+")
_MAX_EXCERPT_CHARS: Final = 500


class _AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JudgeFinding(_AnswerModel):
    # Code-point offsets into the text under review, end exclusive.
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    excerpt: str = Field(min_length=1, max_length=_MAX_EXCERPT_CHARS)
    suggested_correction: str | None = Field(default=None, max_length=500)


class JudgeRuleAnswer(_AnswerModel):
    rule_id: str = Field(min_length=1, max_length=40)
    result: RuleResultValue
    explanation: str = Field(min_length=1, max_length=1000)
    exception_node_key: str | None = None
    evidence_node_keys: list[str] = Field(default_factory=list, max_length=20)
    findings: list[JudgeFinding] = Field(default_factory=list, max_length=20)


class JudgeAnswer(_AnswerModel):
    """The JSON the model must answer with, exactly this shape.

    Nothing in it is trusted until a later step verifies it: rule ids against the request,
    evidence keys against the policy context, offsets and excerpts against the text.
    """

    results: list[JudgeRuleAnswer] = Field(min_length=1, max_length=50)


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    system: str
    user: str
    # The rules the answer must cover.
    rule_ids: tuple[str, ...]
    # The text under review (the simulator reads it; the model gets it inside `user`).
    text: str
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class JudgeReply:
    # The model's text exactly as returned, not parsed.
    raw: str
    model_id: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    is_simulated: bool


class TextJudge(Protocol):
    """One judge model. Implementations hold no per-call state."""

    model_id: str
    is_simulated: bool

    async def judge(self, request: JudgeRequest) -> JudgeReply:
        """The model's reply to one request. Raises a ProviderError subclass on failure."""
        ...


class SimulatedJudgeScenario(StrEnum):
    PASS_ALL = "pass_all"  # noqa: S105  scenario name, not a credential
    FAIL_FIRST = "fail_first"
    MALFORMED = "malformed"
    FAKE_CITATION = "fake_citation"
    TIMEOUT = "timeout"
    THROTTLED = "throttled"


class SimulatedTextJudge:
    """SIMULATED judge for local runs and tests. Makes no network calls.

    Answers by scenario, never by reading the policy: it proves the plumbing and the failure
    handling, never judging quality. Every explanation starts with "SIMULATED: ".
    """

    model_id = SIMULATED_JUDGE_MODEL_ID
    is_simulated = True

    def __init__(self, scenario: SimulatedJudgeScenario = SimulatedJudgeScenario.PASS_ALL) -> None:
        self.scenario = scenario

    def __repr__(self) -> str:
        return f"SimulatedTextJudge(scenario={self.scenario.value!r})"

    async def judge(self, request: JudgeRequest) -> JudgeReply:
        if not request.rule_ids:
            raise ProviderRejectedError("SIMULATED judge: the request names no rules")
        match self.scenario:
            case SimulatedJudgeScenario.TIMEOUT:
                raise ProviderUnavailableError("SIMULATED judge: the model could not be reached")
            case SimulatedJudgeScenario.THROTTLED:
                raise ProviderRateLimitedError("SIMULATED judge: the model is throttled")
            case SimulatedJudgeScenario.MALFORMED:
                # A reply cut off by the token limit.
                return self._reply('{"results": [{"rule_id": "', "max_tokens")
            case SimulatedJudgeScenario.FAIL_FIRST:
                answer = _fail_first(request)
            case SimulatedJudgeScenario.FAKE_CITATION:
                answer = _fake_citation(request)
            case SimulatedJudgeScenario.PASS_ALL:
                answer = JudgeAnswer(results=[_pass(rule_id) for rule_id in request.rule_ids])
        return self._reply(answer.model_dump_json(), "end_turn")

    def _reply(self, raw: str, stop_reason: str) -> JudgeReply:
        return JudgeReply(
            raw=raw,
            model_id=self.model_id,
            stop_reason=stop_reason,
            input_tokens=None,
            output_tokens=None,
            latency_ms=0,
            is_simulated=True,
        )


def _pass(rule_id: str, evidence: list[str] | None = None) -> JudgeRuleAnswer:
    return JudgeRuleAnswer(
        rule_id=rule_id,
        result=RuleResultValue.PASS,
        explanation=f"{_SIMULATED_PREFIX}no problem found",
        evidence_node_keys=evidence or [],
    )


def _fail_first(request: JudgeRequest) -> JudgeAnswer:
    first, *rest = request.rule_ids
    word = _WORD.search(request.text)
    findings: list[JudgeFinding] = []
    if word is not None:
        # Offsets are code points (Python str indexes), not UTF-8 bytes or UTF-16 units.
        start, end = word.start(), min(word.end(), word.start() + _MAX_EXCERPT_CHARS)
        findings.append(JudgeFinding(start=start, end=end, excerpt=request.text[start:end]))
    failed = JudgeRuleAnswer(
        rule_id=first,
        result=RuleResultValue.FAIL,
        explanation=f"{_SIMULATED_PREFIX}the first word breaks this rule",
        findings=findings,
    )
    return JudgeAnswer(results=[failed, *(_pass(rule_id) for rule_id in rest)])


def _fake_citation(request: JudgeRequest) -> JudgeAnswer:
    # Cites evidence the policy context does not hold and answers for a rule nobody asked about.
    results = [_pass(rule_id, [UNKNOWN_EVIDENCE_KEY]) for rule_id in request.rule_ids]
    return JudgeAnswer(results=[*results, _pass(FAKE_RULE_ID)])


def missing_live_judge_settings(settings: Settings) -> list[str]:
    # What live mode lacks for the judge; messages name settings, never values. Empty for the
    # other modes.
    if settings.validation_model_mode != "live":
        return []
    model_id = settings.bedrock_validation_model_id
    if not model_id:
        return ["missing BEDROCK_VALIDATION_MODEL_ID"]
    if model_id not in SUPPORTED_BEDROCK_JUDGE_MODEL_IDS:
        supported = ", ".join(sorted(SUPPORTED_BEDROCK_JUDGE_MODEL_IDS))
        return [f"BEDROCK_VALIDATION_MODEL_ID is not a supported model (use one of: {supported})"]
    return []


def text_judge_from_settings(settings: Settings) -> TextJudge | None:
    """The judge for VALIDATION_MODEL_MODE; None when it is off (no model judge).

    Raises:
        ProviderConfigError: live mode lacks a setting. It never falls back to the simulator.
    """
    if settings.validation_model_mode == "off":
        return None
    if settings.validation_model_mode == "simulator":
        return SimulatedTextJudge()
    problems = missing_live_judge_settings(settings)
    if problems:
        raise ProviderConfigError(f"The live text judge is not configured: {'; '.join(problems)}")
    # Imported here so the API process never loads boto3 or the adapter.
    from app.services.providers.bedrock_judge import BedrockTextJudge

    return BedrockTextJudge.from_settings(settings)
