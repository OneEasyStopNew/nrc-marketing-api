"""Compare text-judge models (NRC-7) on labelled synthetic texts: misses, false flags, cost.

Usage (calls Amazon Bedrock with the caller's AWS credentials; no database):
    uv run python -m scripts.eval_judge --models <id>[,<id>...] [--profile <aws profile>]
        [--budget-usd 1.0] [--cases <file>] [--prices <file>] [--dry-run]
        [--out-md <file>] [--out-json <file>]

Every model gets exactly the prompt production sends (judging.build_request) for each case, with
the synthetic graph's semantic text rules and their supporting clauses as evidence, and each
answer goes through production's verification (judging.verify). One call per case and model: the
first answer is what is scored (production retries an invalid answer; the valid-answer rate says
how often it would need to). Reported per model:
- missed mandatory violations: a rule the text breaks answered PASS or NOT_APPLICABLE in a valid
  answer (the deciding number: in production that is a false PASS),
- false flags: a rule the text keeps answered FAIL,
- valid-answer rate (verified), and the failure codes of the rest,
- citation accuracy (cited evidence exists) and span accuracy (quoted spans are exact),
- how many PASS answers cite evidence,
- latency p50/p95 per call, tokens, and cost from the price file.

Raw model answers are scored: production also distrusts a PASS on text aimed at the checker,
which this report leaves out on purpose. Before any call the script estimates the worst case
(every answer at the output cap) and refuses to start above --budget-usd; it also stops once the
measured spend reaches it. The default cases are synthetic and developer-labelled, not the
expert-labelled set.
"""

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, Self

import boto3
from botocore.config import Config
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import Settings
from app.models.enums import PolicyEdgeType, PolicyLanguage
from app.schemas.validations import RuleResultValue
from app.services import judging
from app.services.providers.base import ProviderConfigError, ProviderError
from app.services.providers.bedrock_judge import BedrockTextJudge
from app.services.providers.text_judge import JudgeAnswer, JudgeReply, JudgeRequest, TextJudge
from scripts.import_policy_graph import PolicyGraphImport, load_synthetic_graph

DATA = Path(__file__).parent / "data"
DEFAULT_CASES: Final = DATA / "judge_eval_synthetic.json"
DEFAULT_PRICES: Final = DATA / "bedrock_judge_prices.json"
DEFAULT_BUDGET_USD: Final = Decimal("1.0")
# Characters per input token for the pre-run estimate: low, so the estimate is on the high side.
CHARS_PER_TOKEN: Final = 3
_TOKENS_PER_PRICE_UNIT: Final = 1_000_000
_NO_PASS: Final = frozenset({RuleResultValue.PASS, RuleResultValue.NOT_APPLICABLE})


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalCase(_Model):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    expect_fail: list[str]
    allow_fail: list[str] = Field(default_factory=list)
    unreadable: bool = False
    why: str = Field(min_length=1)


class CaseFile(_Model):
    note: str = Field(alias="_note")
    rules: list[str] = Field(min_length=1)
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _labels_name_the_rules(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        for case in self.cases:
            labelled = {*case.expect_fail, *case.allow_fail}
            if not labelled <= set(self.rules):
                raise ValueError(f"{case.id}: labels name rules not in 'rules'")
            if set(case.expect_fail) & set(case.allow_fail):
                raise ValueError(f"{case.id}: a rule is both expect_fail and allow_fail")
        return self


class Price(_Model):
    # USD per 1M tokens. Decimal, like every amount here: float sums drift and the spend cap
    # compares them.
    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)


class PriceFile(_Model):
    note: str = Field(alias="_note")
    source: str
    published: str
    models: dict[str, Price]


@dataclass
class Spend:
    budget_usd: Decimal
    total_usd: Decimal = Decimal(0)

    @property
    def exhausted(self) -> bool:
        return self.total_usd >= self.budget_usd


@dataclass
class CaseResult:
    case_id: str
    valid: bool
    failure: str | None = None
    answers: dict[str, str] = field(default_factory=dict)
    provider_error: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: Decimal | None = None
    evidence_ok: bool | None = None
    spans_total: int = 0
    spans_ok: int = 0
    passes: int = 0
    passes_with_evidence: int = 0


@dataclass
class ModelReport:
    model_id: str
    calls: int = 0
    valid: int = 0
    provider_errors: int = 0
    access_error: str | None = None
    stopped_for_budget: bool = False
    failures: dict[str, int] = field(default_factory=dict)
    violations: int = 0
    caught: int = 0
    missed: int = 0
    unassessed_violations: int = 0
    clean_rules: int = 0
    false_flags: int = 0
    unassessed_clean: int = 0
    unreadable_passed: int = 0
    parsed: int = 0
    evidence_ok: int = 0
    spans_total: int = 0
    spans_ok: int = 0
    passes: int = 0
    passes_with_evidence: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    cases: list[CaseResult] = field(default_factory=list)


def load_cases(path: Path) -> CaseFile:
    return CaseFile.model_validate_json(path.read_text(encoding="utf-8"))


def load_prices(path: Path) -> PriceFile:
    return PriceFile.model_validate_json(path.read_text(encoding="utf-8"))


def judge_context(graph: PolicyGraphImport, rule_ids: list[str]) -> judging.JudgeContext:
    """The rules as production sends them, with their English supporting clauses as evidence."""
    nodes = {node.node_key: node for node in graph.nodes}
    rules = {rule.rule_id: rule for rule in graph.rules}
    judged = [
        judging.JudgedRule(
            rule_id=rule_id,
            node_key=rules[rule_id].node_key,
            title=rules[rule_id].title,
            requirement=rules[rule_id].requirement,
        )
        for rule_id in rule_ids
    ]
    keys = {rule.node_key for rule in judged}
    evidence = {
        edge.to_node_key: nodes[edge.to_node_key].text
        for edge in graph.edges
        if edge.edge_type is PolicyEdgeType.SUPPORTED_BY
        and edge.from_node_key in keys
        and nodes[edge.to_node_key].language is PolicyLanguage.EN
    }
    return judging.JudgeContext(rules=tuple(judged), evidence=evidence)


def mandatory_rules(graph: PolicyGraphImport) -> frozenset[str]:
    return frozenset(rule.rule_id for rule in graph.rules if rule.mandatory)


def worst_case_usd(
    cases: CaseFile, context: judging.JudgeContext, price: Price, max_output_tokens: int
) -> Decimal:
    # Every answer at the output cap, and input counted generously from its characters.
    total = Decimal(0)
    for case in cases.cases:
        request = judging.build_request(case.text, context, max_output_tokens)
        input_tokens = math.ceil(len(request.system + request.user) / CHARS_PER_TOKEN)
        total += input_tokens * price.input + max_output_tokens * price.output
    return total / _TOKENS_PER_PRICE_UNIT


def score_answer(
    result: CaseResult, reply: JudgeReply, request: JudgeRequest, context: judging.JudgeContext
) -> None:
    # Verification first (what production does with the answer), then the finer measures on any
    # answer that parses, valid or not.
    try:
        answers = judging.verify(reply, request, context)
    except judging.InvalidJudgeAnswerError as error:
        result.failure = error.failure.value
    else:
        result.valid = True
        result.answers = {rule_id: answer.result.value for rule_id, answer in answers.items()}
    try:
        parsed = judging.parse_answer(reply.raw)
    except judging.InvalidJudgeAnswerError:
        return
    _measure(result, parsed, request.text, context)


def _measure(result: CaseResult, answer: JudgeAnswer, text: str, ctx: judging.JudgeContext) -> None:
    result.evidence_ok = all(
        set(item.evidence_node_keys) <= ctx.evidence.keys() for item in answer.results
    )
    for item in answer.results:
        for finding in item.findings:
            result.spans_total += 1
            in_range = 0 <= finding.start < finding.end <= len(text)
            if in_range and text[finding.start : finding.end] == finding.excerpt:
                result.spans_ok += 1
        if item.result is RuleResultValue.PASS:
            result.passes += 1
            result.passes_with_evidence += bool(item.evidence_node_keys)


def tally(
    report: ModelReport, cases: CaseFile, results: list[CaseResult], mandatory: frozenset[str]
) -> None:
    by_id = {case.id: case for case in cases.cases}
    latencies: list[int] = []
    failures: Counter[str] = Counter()
    for result in results:
        case = by_id[result.case_id]
        report.calls += 1
        if result.provider_error is not None:
            report.provider_errors += 1
            continue
        if result.latency_ms is not None:
            latencies.append(result.latency_ms)
        report.input_tokens += result.input_tokens or 0
        report.output_tokens += result.output_tokens or 0
        report.cost_usd += result.cost_usd or Decimal(0)
        if result.evidence_ok is not None:
            report.parsed += 1
            report.evidence_ok += result.evidence_ok
        report.spans_total += result.spans_total
        report.spans_ok += result.spans_ok
        report.passes += result.passes
        report.passes_with_evidence += result.passes_with_evidence
        if not result.valid:
            failures[result.failure or "UNKNOWN"] += 1
        else:
            report.valid += 1
        _tally_rules(report, case, result, cases.rules, mandatory)
    report.failures = dict(failures)
    report.latency_p50_ms = _percentile(latencies, 50)
    report.latency_p95_ms = _percentile(latencies, 95)


def _tally_rules(
    report: ModelReport,
    case: EvalCase,
    result: CaseResult,
    rules: list[str],
    mandatory: frozenset[str],
) -> None:
    # An invalid answer leaves every rule UNABLE_TO_ASSESS in production: safe, never a miss.
    for rule_id in rules:
        answer = (
            RuleResultValue(result.answers[rule_id])
            if result.valid
            else RuleResultValue.UNABLE_TO_ASSESS
        )
        if case.unreadable:
            report.unreadable_passed += answer in _NO_PASS
        elif rule_id in case.expect_fail:
            if rule_id not in mandatory:
                continue
            report.violations += 1
            if answer is RuleResultValue.FAIL:
                report.caught += 1
            elif answer in _NO_PASS:
                report.missed += 1
            else:
                report.unassessed_violations += 1
        elif rule_id not in case.allow_fail:
            report.clean_rules += 1
            if answer is RuleResultValue.FAIL:
                report.false_flags += 1
            elif answer is not RuleResultValue.PASS:
                report.unassessed_clean += 1


def _percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return float(ordered[rank])


async def evaluate_model(
    judge: TextJudge,
    cases: CaseFile,
    context: judging.JudgeContext,
    price: Price,
    max_output_tokens: int,
    spend: Spend,
) -> list[CaseResult]:
    """One call per case. Stops at the first configuration or access error (the model can't be
    used), or once the shared spend reaches the budget."""
    results: list[CaseResult] = []
    for case in cases.cases:
        if spend.exhausted:
            break
        request = judging.build_request(case.text, context, max_output_tokens)
        result = CaseResult(case_id=case.id, valid=False)
        started = time.monotonic()
        try:
            reply = await judge.judge(request)
        except ProviderConfigError as error:
            result.provider_error = error.message
            results.append(result)
            break
        except ProviderError as error:
            result.provider_error = error.message
            results.append(result)
            continue
        result.latency_ms = reply.latency_ms or round((time.monotonic() - started) * 1000)
        result.input_tokens = reply.input_tokens
        result.output_tokens = reply.output_tokens
        cost = (
            (reply.input_tokens or 0) * price.input + (reply.output_tokens or 0) * price.output
        ) / _TOKENS_PER_PRICE_UNIT
        result.cost_usd = cost
        spend.total_usd += cost
        score_answer(result, reply, request, context)
        results.append(result)
    return results


def build_report(
    model_id: str, cases: CaseFile, results: list[CaseResult], mandatory: frozenset[str]
) -> ModelReport:
    report = ModelReport(model_id=model_id, cases=results)
    tally(report, cases, results, mandatory)
    first_error = next((r.provider_error for r in results if r.provider_error), None)
    if results and results[-1].provider_error is not None and len(results) < len(cases.cases):
        report.access_error = first_error
    return report


def format_markdown(
    reports: list[ModelReport], cases: CaseFile, prices: PriceFile, spend: Spend, ran_at: datetime
) -> str:
    lines = [
        f"# Judge model evaluation ({ran_at:%Y-%m-%d %H:%M} UTC)",
        "",
        f"{len(cases.cases)} synthetic, developer-labelled English cases (`{DEFAULT_CASES.name}`);"
        f" rules {', '.join(cases.rules)}. One call per case and model, the production prompt "
        f"`{judging.PROMPT_TEMPLATE_VERSION}`, raw answers scored. Prices: {prices.source} "
        f"(published {prices.published}). Spend {spend.total_usd:.4f} of {spend.budget_usd:.2f} "
        "USD.",
        "",
        "| Model | Missed mandatory violations | False flags | Valid answers | Citations ok |"
        " Model's offsets exact | PASS citing evidence | p50 / p95 ms | Tokens in / out |"
        " Cost USD |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        if report.access_error is not None and report.valid == 0:
            lines.append(f"| `{report.model_id}` | not run: {report.access_error} |||||||||")
            continue
        lines.append(
            f"| `{report.model_id}` | {report.missed} of {report.violations} "
            f"(caught {report.caught}, unassessed {report.unassessed_violations}) | "
            f"{report.false_flags} of {report.clean_rules} "
            f"(unassessed {report.unassessed_clean}) | {_ratio(report.valid, report.calls)} | "
            f"{_ratio(report.evidence_ok, report.parsed)} | "
            f"{_ratio(report.spans_ok, report.spans_total)} | "
            f"{_ratio(report.passes_with_evidence, report.passes)} | "
            f"{_ms(report.latency_p50_ms)} / {_ms(report.latency_p95_ms)} | "
            f"{report.input_tokens} / {report.output_tokens} | {report.cost_usd:.4f} |"
        )
    lines.append("")
    for report in reports:
        notes = []
        if report.failures:
            notes.append(
                "invalid answers: "
                + ", ".join(f"{code} {count}" for code, count in sorted(report.failures.items()))
            )
        if report.unreadable_passed:
            notes.append(f"PASS on unreadable text: {report.unreadable_passed}")
        if report.provider_errors:
            notes.append(f"provider errors: {report.provider_errors}")
        if report.stopped_for_budget:
            notes.append("stopped: budget reached")
        if notes:
            lines.append(f"- `{report.model_id}`: {'; '.join(notes)}.")
    return "\n".join(lines) + "\n"


def _ratio(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{part}/{whole} ({100 * part / whole:.0f}%)"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _judge(model_id: str, args: argparse.Namespace) -> BedrockTextJudge:
    config = Config(
        connect_timeout=args.timeout,
        read_timeout=args.timeout,
        retries={"max_attempts": args.retries, "mode": "standard"},
    )
    client = boto3.Session(profile_name=args.profile).client(
        "bedrock-runtime", region_name=args.region, config=config
    )
    return BedrockTextJudge(
        model_id=model_id,
        client=client,
        max_output_tokens=args.max_output_tokens,
        call_budget_seconds=args.timeout * (args.retries + 1),
    )


def _default(name: str) -> object:
    # The production setting's default, without loading (or needing) the API's environment.
    return Settings.model_fields[name].default


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", required=True, help="comma-separated Bedrock model ids")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--budget-usd", type=Decimal, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--profile", default=None, help="AWS profile (default: the usual chain)")
    parser.add_argument("--region", default=_default("aws_region"))
    parser.add_argument(
        "--max-output-tokens", type=int, default=_default("validation_model_max_output_tokens")
    )
    parser.add_argument(
        "--timeout", type=float, default=_default("validation_model_timeout_seconds")
    )
    parser.add_argument("--retries", type=int, default=_default("bedrock_max_retries"))
    parser.add_argument("--dry-run", action="store_true", help="estimate the cost and stop")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    prices = load_prices(args.prices)
    graph = load_synthetic_graph()
    context = judge_context(graph, cases.rules)
    model_ids = [model.strip() for model in args.models.split(",") if model.strip()]
    unpriced = [model for model in model_ids if model not in prices.models]
    if unpriced:
        print(f"No price for {', '.join(unpriced)} in {args.prices}; refusing to run.")
        return 2
    estimate = sum(
        worst_case_usd(cases, context, prices.models[model], args.max_output_tokens)
        for model in model_ids
    )
    print(
        f"Worst-case estimate {estimate:.4f} USD for {len(model_ids)} models x "
        f"{len(cases.cases)} cases (budget {args.budget_usd:.2f} USD)."
    )
    if estimate > args.budget_usd:
        print("Estimate is above the budget; nothing was called.")
        return 2
    if args.dry_run:
        return 0
    spend = Spend(budget_usd=args.budget_usd)
    mandatory = mandatory_rules(graph)
    reports = []
    for model_id in model_ids:
        results = await evaluate_model(
            _judge(model_id, args),
            cases,
            context,
            prices.models[model_id],
            args.max_output_tokens,
            spend,
        )
        report = build_report(model_id, cases, results, mandatory)
        report.stopped_for_budget = spend.exhausted and len(results) < len(cases.cases)
        reports.append(report)
        print(f"{model_id}: {report.calls} calls, spend so far {spend.total_usd:.4f} USD")
    markdown = format_markdown(reports, cases, prices, spend, datetime.now(tz=UTC))
    print(markdown)
    if args.out_md is not None:
        args.out_md.write_text(markdown, encoding="utf-8")
    if args.out_json is not None:
        args.out_json.write_text(
            # Amounts as exact decimal strings.
            json.dumps([asdict(report) for report in reports], indent=1, default=str),
            encoding="utf-8",
        )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main(_arguments())))


if __name__ == "__main__":
    main()
