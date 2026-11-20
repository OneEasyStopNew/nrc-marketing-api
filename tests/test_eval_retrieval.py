# Retrieval evaluation harness (NRC-5, scripts/eval_retrieval.py): metrics, case file, runs.

from collections.abc import Sequence
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import RetrievalSettings, get_settings
from app.models.enums import (
    Channel,
    PolicyLanguage,
    RetrievalMode,
    ValidationInputKind,
)
from app.services.policy_context import AuroraPolicyContextProvider
from app.services.policy_index import build_policy_index
from app.services.providers.base import ProviderRateLimitedError
from app.services.providers.embeddings import (
    EmbeddingProvider,
    EmbeddingPurpose,
    SimulatedEmbeddingProvider,
)
from scripts.eval_retrieval import (
    DEFAULT_CASES,
    CaseResult,
    CostEstimate,
    EvalCase,
    EvalCaseFile,
    EvalSetupError,
    Ratio,
    build_report,
    choose_embedder,
    estimate_tokens,
    evaluate,
    format_report,
    load_cases,
    percentile,
    summarize,
)
from scripts.import_policy_graph import import_policy_graph, load_synthetic_graph
from tests.conftest import World, add_policy_bundle, as_tenant_owner, owner_session


def _case(**overrides: object) -> EvalCase:
    fields: dict[str, object] = {
        "case_id": "en-test",
        "language": PolicyLanguage.EN,
        "input_kind": ValidationInputKind.BRIEF,
        "channel": Channel.FACEBOOK,
        "text": "This proven treatment is safe and effective.",
        "required_rules": ("REC-01",),
        "relevant_evidence": ("CL-FDA-1",),
        "search_targets": ("CL-FDA-1",),
    }
    return EvalCase.model_validate(fields | overrides)


def _provider(embedder: EmbeddingProvider | None = None) -> AuroraPolicyContextProvider:
    return AuroraPolicyContextProvider(get_settings(), embedder)


async def _synthetic(connection: AsyncConnection, world: World) -> UUID:
    assert world.a.policy_bundle_id is not None
    async with owner_session(connection, world.a.tenant_id) as db:
        await import_policy_graph(
            db, world.a.tenant_id, world.a.policy_bundle_id, load_synthetic_graph()
        )
    return world.a.policy_bundle_id


async def _evaluate(
    connection: AsyncConnection,
    world: World,
    bundle_id: UUID,
    cases: tuple[EvalCase, ...],
    provider: AuroraPolicyContextProvider,
) -> list[CaseResult]:
    # As the owner with the tenant bound, like the script.
    async with owner_session(connection, world.a.tenant_id) as db:
        return await evaluate(
            db, world.a.tenant_id, bundle_id, world.a.snapshot_id, cases, provider
        )


# ---- Metrics


def test_percentile_is_nearest_rank() -> None:
    values = [40.0, 10.0, 30.0, 20.0]

    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == 40.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) == 0.0


def test_ratio_without_labels_is_neither_pass_nor_fail() -> None:
    assert Ratio(found=0, expected=0).value is None
    assert Ratio(found=3, expected=4).value == 0.75


def test_token_estimate_rounds_up_per_text() -> None:
    assert estimate_tokens(["abcd", "abcde", ""]) == 1 + 2 + 0


def test_cost_is_unpriced_without_a_price() -> None:
    cost = CostEstimate("model", 1_000_000, 2_000_000, None, None)
    priced = CostEstimate("model", 1_000_000, 2_000_000, 0.02, "test price")

    assert (cost.query_cost_usd, cost.index_cost_usd) == (None, None)
    assert (priced.query_cost_usd, priced.index_cost_usd) == (0.02, 0.04)


def test_full_mode_without_an_embedder_is_refused() -> None:
    # RetrievalSettings loads without the API's settings (Cognito, DATABASE_URL).
    settings = RetrievalSettings(embedding_mode="off")

    assert choose_embedder("lexical", settings) is None
    assert choose_embedder("auto", settings) is None
    with pytest.raises(EvalSetupError, match="--mode full needs an embedder"):
        choose_embedder("full", settings)


# ---- Case file


def test_search_targets_must_be_relevant() -> None:
    with pytest.raises(ValidationError, match="search_targets must be in relevant_evidence"):
        _case(relevant_evidence=(), search_targets=("CL-FDA-1",))


def test_case_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="case_id values must be unique"):
        EvalCaseFile(labelled_by="test", cases=(_case(), _case()))


def test_shipped_cases_cover_both_languages_and_label_real_keys() -> None:
    case_file = load_cases(DEFAULT_CASES)
    graph = load_synthetic_graph()
    rule_ids = {rule.rule_id for rule in graph.rules}
    node_keys = {node.node_key for node in graph.nodes}

    assert "not expert-reviewed" in case_file.labelled_by
    assert {case.language for case in case_file.cases} == {PolicyLanguage.EN, PolicyLanguage.ES}
    assert {case.input_kind for case in case_file.cases} >= {
        ValidationInputKind.BRIEF,
        ValidationInputKind.TEXT,
    }
    for case in case_file.cases:
        assert set(case.required_rules) <= rule_ids, case.case_id
        assert set(case.relevant_evidence) <= node_keys, case.case_id


# ---- Runs on the test database


async def test_synthetic_set_has_full_mandatory_coverage(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id = await _synthetic(connection, world)
    case_file = load_cases(DEFAULT_CASES)

    results = await _evaluate(connection, world, bundle_id, case_file.cases, _provider())
    report = build_report(
        case_file,
        results,
        (bundle_id, 1),
        world.a.snapshot_id,
        mode=RetrievalMode.LEXICAL_ONLY,
        top_k=5,
        cost=CostEstimate(None, 0, 0, None, None),
    )

    assert report.coverage_complete
    assert report.overall.cases == len(case_file.cases)
    assert set(report.by_language) == {"en", "es"}
    assert report.overall.evidence_recall.value == 1.0
    assert all(r.retrieval_mode is RetrievalMode.LEXICAL_ONLY for r in results)
    # Full-text search can't find a paraphrase or a clause with no text in the post's language.
    missed = {r.case_id: r.search_missing for r in results if r.search_missing}
    assert missed == {"en-brief-paraphrase": ("CL-FDA-4",), "es-brief-free": ("CL-FDA-5",)}
    assert "Cost: no embedding calls" in format_report(report)


async def test_missing_mandatory_rule_fails_coverage(
    connection: AsyncConnection, world: World
) -> None:
    # REC-09 applies to images only, so a brief never selects it: the label is a miss.
    bundle_id = await _synthetic(connection, world)
    case_file = EvalCaseFile(
        labelled_by="test", cases=(_case(required_rules=("REC-01", "REC-09")),)
    )

    results = await _evaluate(connection, world, bundle_id, case_file.cases, _provider())
    report = build_report(
        case_file,
        results,
        (bundle_id, 1),
        world.a.snapshot_id,
        mode=RetrievalMode.LEXICAL_ONLY,
        top_k=5,
        cost=CostEstimate(None, 0, 0, None, None),
    )

    assert not report.coverage_complete
    assert report.by_rule["REC-09"] == Ratio(found=0, expected=1)
    assert results[0].required_missing == ("REC-09",)
    assert "MANDATORY RULE MISSING REC-09" in format_report(report)


async def test_unknown_labels_stop_the_run(connection: AsyncConnection, world: World) -> None:
    bundle_id = await _synthetic(connection, world)

    with pytest.raises(EvalSetupError, match="labels not approved in the bundle: CL-TYPO, REC-99"):
        await _evaluate(
            connection,
            world,
            bundle_id,
            (_case(required_rules=("REC-99",), relevant_evidence=("CL-TYPO",), search_targets=()),),
            _provider(),
        )


async def test_full_mode_needs_a_ready_index(connection: AsyncConnection, world: World) -> None:
    bundle_id = await _synthetic(connection, world)

    with pytest.raises(EvalSetupError, match="INDEX_NOT_READY"):
        await _evaluate(
            connection, world, bundle_id, (_case(),), _provider(SimulatedEmbeddingProvider())
        )


async def test_full_mode_searches_both_ways(connection: AsyncConnection, world: World) -> None:
    bundle_id = await _synthetic(connection, world)
    embedder = SimulatedEmbeddingProvider()
    async with owner_session(connection, world.a.tenant_id) as db:
        await build_policy_index(db, world.a.tenant_id, bundle_id, embedder)

    results = await _evaluate(connection, world, bundle_id, (_case(),), _provider(embedder))

    assert results[0].retrieval_mode is RetrievalMode.FULL
    assert results[0].search_missing == ()
    assert "SEMANTIC_SEARCH_SKIPPED" not in results[0].missing_context


class _ThrottledEmbedder(SimulatedEmbeddingProvider):
    # Same model id as the simulator (so its index is READY), but every call is throttled.

    async def embed(self, texts: Sequence[str], purpose: EmbeddingPurpose) -> list[list[float]]:
        raise ProviderRateLimitedError("throttled")


def _result(case_id: str, latency_ms: float, *, reused: bool) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        language=PolicyLanguage.EN,
        required_expected=("REC-01",),
        required_missing=(),
        required_extra=(),
        relevant_expected=(),
        relevant_missing=(),
        search_expected=(),
        search_missing=(),
        retrieval_mode=RetrievalMode.LEXICAL_ONLY,
        latency_ms=latency_ms,
        reused=reused,
        missing_context=(),
    )


def test_reused_manifests_are_left_out_of_latency() -> None:
    group = summarize([_result("a", 10.0, reused=False), _result("b", 0.5, reused=True)])

    assert (group.cases, group.reused) == (2, 1)
    assert group.latency_p50_ms == group.latency_p95_ms == 10.0


async def test_report_shows_snapshot_extra_rules_and_small_sample_note(
    connection: AsyncConnection, world: World
) -> None:
    # Labelling only REC-01 for a brief leaves the brief's other mandatory rules unlabelled.
    bundle_id = await _synthetic(connection, world)
    case_file = EvalCaseFile(labelled_by="test", cases=(_case(),))

    results = await _evaluate(connection, world, bundle_id, case_file.cases, _provider())
    text_report = format_report(
        build_report(
            case_file,
            results,
            (bundle_id, 1),
            world.a.snapshot_id,
            mode=RetrievalMode.LEXICAL_ONLY,
            top_k=5,
            cost=CostEstimate(None, 0, 0, None, None),
        )
    )

    assert f"Study snapshot: {world.a.snapshot_id}" in text_report
    assert "en-test: REC-02, REC-03, REC-04, REC-05, REC-08" in text_report
    assert "Only 1 timed case(s): p95 is the slowest one" in text_report


async def test_unapproved_labels_stop_the_run(connection: AsyncConnection, world: World) -> None:
    bundle_id = await _synthetic(connection, world)
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            text(
                "UPDATE policy_nodes SET approved = false, review_state = 'PROPOSED'"
                " WHERE policy_bundle_id = :b AND node_key = 'CL-FDA-1'"
            ),
            {"b": bundle_id},
        )

    with pytest.raises(EvalSetupError, match="labels not approved in the bundle: CL-FDA-1"):
        await _evaluate(connection, world, bundle_id, (_case(),), _provider())


async def test_bundle_without_graph_gets_no_index_hint(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        empty = await add_policy_bundle(connection, world.a.tenant_id, 2)
    case = _case(required_rules=(), relevant_evidence=(), search_targets=())

    with pytest.raises(EvalSetupError, match=r"retrieval unavailable \(GRAPH_MISSING\)$"):
        await _evaluate(connection, world, empty, (case,), _provider())


async def test_embedding_failure_stops_with_its_code(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id = await _synthetic(connection, world)
    async with owner_session(connection, world.a.tenant_id) as db:
        await build_policy_index(db, world.a.tenant_id, bundle_id, SimulatedEmbeddingProvider())

    with pytest.raises(EvalSetupError, match="embedding call failed on case en-test"):
        await _evaluate(connection, world, bundle_id, (_case(),), _provider(_ThrottledEmbedder()))
