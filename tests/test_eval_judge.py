# The judge model evaluation (scripts.eval_judge) without a network: the case file, the scoring
# and the spend cap, with a fake judge that answers from a table.

import json
from collections.abc import Mapping
from typing import Any

import pytest

from app.models.enums import CheckType, Gate, Severity
from app.schemas.tenancy import StudyFacts
from app.schemas.validations import SourceReference
from app.services import judging, rules
from app.services.providers.base import ProviderConfigError
from app.services.providers.text_judge import JudgeReply, JudgeRequest
from scripts import eval_judge
from scripts.import_policy_graph import load_synthetic_graph
from scripts.seed_demo import SYNTHETIC_FACTS

FOUR_MODELS = (
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "openai.gpt-oss-120b-1:0",
    "mistral.mistral-large-3-675b-instruct",
)
CLEAN = "We are looking for volunteers for a research study about high blood pressure."
BREAKS_REC_01 = "This investigational pill lowers blood pressure quickly."


def _cases(*cases: dict[str, Any]) -> eval_judge.CaseFile:
    return eval_judge.CaseFile.model_validate(
        {"_note": "test", "rules": ["REC-01", "REC-04"], "cases": list(cases)}
    )


def _answer(results: Mapping[str, str], evidence: tuple[str, ...] = ("CL-FDA-1",)) -> str:
    return json.dumps(
        {
            "results": [
                {
                    "rule_id": rule_id,
                    "result": result,
                    "explanation": "test",
                    "evidence_node_keys": list(evidence),
                }
                for rule_id, result in results.items()
            ]
        }
    )


class TableJudge:
    # Answers each text from a table; a text missing from it raises the given error.

    model_id = "amazon.nova-lite-v1:0"
    is_simulated = False

    def __init__(self, answers: Mapping[str, str], error: Exception | None = None) -> None:
        self._answers = answers
        self._error = error
        self.calls = 0

    async def judge(self, request: JudgeRequest) -> JudgeReply:
        self.calls += 1
        if request.text not in self._answers:
            assert self._error is not None
            raise self._error
        return JudgeReply(
            raw=self._answers[request.text],
            model_id=self.model_id,
            stop_reason="end_turn",
            input_tokens=1000,
            output_tokens=200,
            latency_ms=50,
            is_simulated=False,
        )


def _context() -> judging.JudgeContext:
    return eval_judge.judge_context(load_synthetic_graph(), ["REC-01", "REC-04"])


async def _run(
    judge: TableJudge, cases: eval_judge.CaseFile, budget: float = 1.0
) -> eval_judge.ModelReport:
    price = eval_judge.Price(input=1.0, output=1.0)
    spend = eval_judge.Spend(budget_usd=budget)
    results = await eval_judge.evaluate_model(judge, cases, _context(), price, 2000, spend)
    mandatory = eval_judge.mandatory_rules(load_synthetic_graph())
    return eval_judge.build_report(judge.model_id, cases, results, mandatory)


def test_the_case_file_is_valid_and_every_text_leaves_its_rules_to_the_model() -> None:
    # A prohibited phrase would be caught by code and never reach the model in production.
    cases = eval_judge.load_cases(eval_judge.DEFAULT_CASES)
    graph = load_synthetic_graph()
    by_id = {rule.rule_id: rule for rule in graph.rules}
    graph_rules = [
        rules.GraphRule(
            rule_id=rule_id,
            rule_version=1,
            node_key=by_id[rule_id].node_key,
            mandatory=by_id[rule_id].mandatory,
            gate=Gate.POLICY,
            severity=Severity.MAJOR,
            source=SourceReference(source_id="synthetic", clause=rule_id),
            check_type=CheckType.SEMANTIC,
            content=rules.RuleContent(
                title=by_id[rule_id].title,
                requirement=by_id[rule_id].requirement,
                exact_wording=None,
                prohibited_phrases=tuple(by_id[rule_id].prohibited_phrases),
            ),
        )
        for rule_id in cases.rules
    ]
    facts = StudyFacts.model_validate(SYNTHETIC_FACTS)

    for case in cases.cases:
        sent = rules.rules_for_model(case.text, graph_rules, facts)
        assert [rule.rule_id for rule in sent] == cases.rules, case.id


def test_labels_must_name_the_evaluated_rules() -> None:
    with pytest.raises(ValueError, match="labels name rules not in 'rules'"):
        _cases({"id": "x", "text": "t", "expect_fail": ["REC-99"], "why": "w"})


def test_the_model_sees_the_rules_and_their_english_clauses() -> None:
    context = _context()

    assert [rule.rule_id for rule in context.rules] == ["REC-01", "REC-04"]
    assert set(context.evidence) == {"CL-FDA-1", "CL-FDA-4"}


async def test_a_pass_on_a_broken_rule_is_a_miss_and_a_fail_on_a_kept_one_a_false_flag() -> None:
    cases = _cases(
        {"id": "breaks", "text": BREAKS_REC_01, "expect_fail": ["REC-01"], "why": "w"},
        {"id": "clean", "text": CLEAN, "expect_fail": [], "why": "w"},
    )
    judge = TableJudge(
        {
            BREAKS_REC_01: _answer({"REC-01": "PASS", "REC-04": "PASS"}),
            CLEAN: _answer({"REC-01": "PASS", "REC-04": "FAIL"}),
        }
    )

    report = await _run(judge, cases)

    assert (report.violations, report.missed, report.caught) == (1, 1, 0)
    assert (report.clean_rules, report.false_flags) == (3, 1)
    assert (report.valid, report.calls) == (2, 2)
    assert (report.passes, report.passes_with_evidence) == (3, 3)


async def test_an_invalid_answer_is_never_a_miss_and_its_code_is_counted() -> None:
    # Production leaves every rule of an invalid answer UNABLE_TO_ASSESS: safe, not a PASS.
    cases = _cases({"id": "breaks", "text": BREAKS_REC_01, "expect_fail": ["REC-01"], "why": "w"})
    judge = TableJudge({BREAKS_REC_01: _answer({"REC-01": "PASS"})})  # REC-04 missing

    report = await _run(judge, cases)

    assert (report.valid, report.missed, report.unassessed_violations) == (0, 0, 1)
    assert report.failures == {"WRONG_RULES": 1}


async def test_wrong_spans_and_unknown_evidence_are_counted() -> None:
    cases = _cases({"id": "breaks", "text": BREAKS_REC_01, "expect_fail": ["REC-01"], "why": "w"})
    answer = json.loads(_answer({"REC-01": "FAIL", "REC-04": "PASS"}, evidence=("NOT-REAL",)))
    answer["results"][0]["findings"] = [{"start": 0, "end": 4, "excerpt": "That"}]
    judge = TableJudge({BREAKS_REC_01: json.dumps(answer)})

    report = await _run(judge, cases)

    assert report.failures == {"UNKNOWN_EVIDENCE": 1}
    assert (report.evidence_ok, report.parsed) == (0, 1)
    assert (report.spans_ok, report.spans_total) == (0, 1)


async def test_an_access_error_stops_that_model_and_is_reported() -> None:
    cases = _cases(
        {"id": "a", "text": CLEAN, "expect_fail": [], "why": "w"},
        {"id": "b", "text": BREAKS_REC_01, "expect_fail": ["REC-01"], "why": "w"},
    )
    denied = ProviderConfigError("Bedrock judge model m failed (AccessDeniedException)")
    judge = TableJudge({}, error=denied)

    report = await _run(judge, cases)

    assert judge.calls == 1
    assert report.access_error is not None
    assert "AccessDeniedException" in report.access_error


async def test_the_run_stops_once_the_spend_reaches_the_budget() -> None:
    # Each call costs 1,200 tokens at 1 USD per 1M tokens: 0.0012 USD.
    cases = _cases(
        *({"id": f"c{n}", "text": f"{CLEAN} {n}", "expect_fail": [], "why": "w"} for n in range(5))
    )
    judge = TableJudge(
        {f"{CLEAN} {n}": _answer({"REC-01": "PASS", "REC-04": "PASS"}) for n in range(5)}
    )

    report = await _run(judge, cases, budget=0.0024)

    assert judge.calls == 2
    assert report.calls == 2


def test_the_worst_case_for_the_four_models_fits_a_one_dollar_budget() -> None:
    cases = eval_judge.load_cases(eval_judge.DEFAULT_CASES)
    prices = eval_judge.load_prices(eval_judge.DEFAULT_PRICES)
    context = eval_judge.judge_context(load_synthetic_graph(), cases.rules)

    total = sum(
        eval_judge.worst_case_usd(cases, context, prices.models[model], 2000)
        for model in FOUR_MODELS
    )

    assert 0 < total < 1.0
