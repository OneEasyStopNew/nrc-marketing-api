"""Evaluate policy retrieval (NRC-5) on labelled cases: coverage, evidence recall, latency, cost.

Usage (runs as the database owner, MIGRATION_DATABASE_URL):
    uv run python -m scripts.eval_retrieval --tenant-id <uuid> [--bundle-id <uuid>]
        [--snapshot-id <uuid>] [--cases <file>] [--mode auto|full|lexical]
        [--price-per-million-tokens <usd> --price-source <text>] [--json-out <file>]

Each case is one post text with its scope, labelled with the mandatory rules the scope must
select, the evidence a reviewer needs, and the nodes the text itself should surface by search.
The script builds a context manifest per case against the bundle (the tenant's active bundle by
default) and reports, overall, per language and per rule:
- mandatory-rule coverage: labelled rules present in required_rules (must be 100%; the exit
  status is non-zero otherwise),
- evidence recall: labelled evidence present in the manifest by any route,
- search recall@k: search targets found by the case's own search within the top k,
- latency p50/p95 of building one manifest, and an estimated embedding cost.

Everything runs in one transaction that is rolled back: no manifest is kept. Full mode needs a
READY index for the configured embedding model (scripts.build_policy_index). With the simulator
the vectors carry no meaning, so its semantic numbers say nothing about a real model. The default
case file is synthetic and developer-labelled; release numbers need the expert set (F32).
"""

import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import OwnerDatabaseSettings, RetrievalSettings
from app.core.errors import PolicyContextUnavailableError
from app.db.session import TenantSession, bind_tenant
from app.models import PolicyBundle, PolicyNode, PolicyRuleRecord, StudySnapshot
from app.models.enums import (
    BundleLifecycleState,
    Channel,
    PolicyLanguage,
    RetrievalMode,
    ValidationInputKind,
)
from app.models.policy_graph import SEARCHABLE_NODE_TYPES
from app.services.policy_context import (
    AuroraPolicyContextProvider,
    ContextManifest,
    ContextRequest,
    ContextScope,
)
from app.services.policy_context.manifest import MAX_STATEMENT_CHARS, Statement
from app.services.providers.embeddings import EmbeddingProvider
from scripts.embedding_setup import load_embedder, load_settings

DEFAULT_CASES: Final = Path(__file__).resolve().parent / "data" / "retrieval_eval_synthetic.json"
# A rough tokens-per-character ratio for English text; Bedrock reports exact counts per call, but
# an estimate is enough to compare models. Reported as an estimate.
CHARS_PER_TOKEN: Final = 4
Mode = Literal["auto", "full", "lexical"]


class EvalSetupError(Exception):
    # The run can't start (unknown labels, no bundle or snapshot, no index): nothing was measured.
    pass


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9-]{1,80}$")
    language: PolicyLanguage
    input_kind: ValidationInputKind
    channel: Channel
    text: str = Field(min_length=1, max_length=MAX_STATEMENT_CHARS)
    required_rules: tuple[str, ...]
    relevant_evidence: tuple[str, ...] = ()
    search_targets: tuple[str, ...] = ()
    note: str = ""

    @model_validator(mode="after")
    def _targets_are_relevant(self) -> Self:
        # A node the text should surface is evidence a reviewer needs, so it's labelled in both.
        if not set(self.search_targets) <= set(self.relevant_evidence):
            raise ValueError(f"{self.case_id}: search_targets must be in relevant_evidence")
        return self


class EvalCaseFile(BaseModel):
    # extra="ignore": the file carries a "_note" for readers.
    model_config = ConfigDict(extra="ignore", frozen=True)

    labelled_by: str
    cases: tuple[EvalCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        return self


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    language: PolicyLanguage
    required_expected: tuple[str, ...]
    required_missing: tuple[str, ...]
    # Mandatory rules selected but not labelled: conservative selection, not a failure.
    required_extra: tuple[str, ...]
    relevant_expected: tuple[str, ...]
    relevant_missing: tuple[str, ...]
    search_expected: tuple[str, ...]
    search_missing: tuple[str, ...]
    retrieval_mode: RetrievalMode
    latency_ms: float
    reused: bool
    missing_context: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ratio:
    found: int
    expected: int

    @property
    def value(self) -> float | None:
        # None when nothing was expected (no labels): neither a pass nor a fail.
        return None if self.expected == 0 else self.found / self.expected


@dataclass(frozen=True, slots=True)
class Group:
    cases: int
    mandatory_coverage: Ratio
    evidence_recall: Ratio
    search_recall: Ratio
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass(frozen=True, slots=True)
class CostEstimate:
    model_id: str | None
    query_tokens: int
    index_tokens: int
    price_per_million_tokens: float | None
    price_source: str | None

    @property
    def query_cost_usd(self) -> float | None:
        return _cost(self.query_tokens, self.price_per_million_tokens)

    @property
    def index_cost_usd(self) -> float | None:
        return _cost(self.index_tokens, self.price_per_million_tokens)


@dataclass(frozen=True, slots=True)
class Report:
    labelled_by: str
    policy_bundle_id: UUID
    policy_bundle_version: int
    retrieval_mode: RetrievalMode
    embedding_model_id: str | None
    top_k: int
    overall: Group
    by_language: dict[str, Group]
    by_rule: dict[str, Ratio]
    cost: CostEstimate
    cases: tuple[CaseResult, ...] = field(default=())

    @property
    def coverage_complete(self) -> bool:
        return self.overall.mandatory_coverage.found == self.overall.mandatory_coverage.expected


def load_cases(path: Path) -> EvalCaseFile:
    return EvalCaseFile.model_validate_json(path.read_text(encoding="utf-8"))


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (0 for no values): always one of the measured values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def estimate_tokens(texts: Iterable[str]) -> int:
    return sum(math.ceil(len(text) / CHARS_PER_TOKEN) for text in texts)


def score_case(
    case: EvalCase, manifest: ContextManifest, latency_ms: float, *, reused: bool
) -> CaseResult:
    required = {rule.rule_id for rule in manifest.required_rules}
    mandatory = {rule.rule_id for rule in manifest.required_rules if rule.mandatory}
    # Exceptions are listed apart from evidence; a reviewer needs both.
    present = {item.node_key for item in manifest.evidence} | {
        item.node_key for item in manifest.exceptions
    }
    searched = {item.node_key for item in manifest.evidence if item.matches}
    return CaseResult(
        case_id=case.case_id,
        language=case.language,
        required_expected=case.required_rules,
        required_missing=tuple(r for r in case.required_rules if r not in required),
        required_extra=tuple(sorted(mandatory - set(case.required_rules))),
        relevant_expected=case.relevant_evidence,
        relevant_missing=tuple(n for n in case.relevant_evidence if n not in present),
        search_expected=case.search_targets,
        search_missing=tuple(n for n in case.search_targets if n not in searched),
        retrieval_mode=manifest.retrieval_mode,
        latency_ms=latency_ms,
        reused=reused,
        missing_context=tuple(sorted({item.code.value for item in manifest.missing_context})),
    )


def summarize(results: Sequence[CaseResult]) -> Group:
    latencies = [result.latency_ms for result in results]
    return Group(
        cases=len(results),
        mandatory_coverage=_ratio(results, "required_expected", "required_missing"),
        evidence_recall=_ratio(results, "relevant_expected", "relevant_missing"),
        search_recall=_ratio(results, "search_expected", "search_missing"),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )


def by_rule(results: Sequence[CaseResult]) -> dict[str, Ratio]:
    expected: defaultdict[str, int] = defaultdict(int)
    missing: defaultdict[str, int] = defaultdict(int)
    for result in results:
        for rule_id in result.required_expected:
            expected[rule_id] += 1
        for rule_id in result.required_missing:
            missing[rule_id] += 1
    return {
        rule_id: Ratio(found=count - missing[rule_id], expected=count)
        for rule_id, count in sorted(expected.items())
    }


def build_report(
    case_file: EvalCaseFile,
    results: Sequence[CaseResult],
    bundle: tuple[UUID, int],
    *,
    mode: RetrievalMode,
    top_k: int,
    cost: CostEstimate,
) -> Report:
    languages: defaultdict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        languages[result.language.value].append(result)
    return Report(
        labelled_by=case_file.labelled_by,
        policy_bundle_id=bundle[0],
        policy_bundle_version=bundle[1],
        retrieval_mode=mode,
        embedding_model_id=cost.model_id,
        top_k=top_k,
        overall=summarize(results),
        by_language={lang: summarize(group) for lang, group in sorted(languages.items())},
        by_rule=by_rule(results),
        cost=cost,
        cases=tuple(results),
    )


async def evaluate(
    db: AsyncSession,
    tenant_id: UUID,
    bundle_id: UUID,
    snapshot_id: UUID,
    cases: Sequence[EvalCase],
    provider: AuroraPolicyContextProvider,
) -> list[CaseResult]:
    """Build one manifest per case in the caller's transaction and score it.

    Raises:
        EvalSetupError: a label names a rule or node the bundle doesn't have (a typo would
            otherwise read as a retrieval miss), or the context can't be built (e.g. no READY
            index in full mode).
    """
    await _check_labels(db, tenant_id, bundle_id, cases)
    evaluated_at = datetime.now(tz=UTC)
    results: list[CaseResult] = []
    for case in cases:
        request = ContextRequest(
            policy_bundle_id=bundle_id,
            study_snapshot_id=snapshot_id,
            scope=ContextScope(
                input_kind=case.input_kind,
                channel=case.channel,
                language=case.language,
                evaluated_at=evaluated_at,
            ),
            statements=(Statement(text=case.text),),
        )
        started = time.perf_counter()
        try:
            stored = await provider.build(db, tenant_id, request, actor="eval_retrieval")
        except PolicyContextUnavailableError as error:
            reason = (error.details or {}).get("reason", error.code)
            raise EvalSetupError(
                f"retrieval unavailable ({reason}); in full mode build the index first with "
                "scripts.build_policy_index"
            ) from None
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(score_case(case, stored.manifest, latency_ms, reused=stored.reused))
    return results


async def _check_labels(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID, cases: Sequence[EvalCase]
) -> None:
    rule_ids = set(
        await db.scalars(
            select(PolicyRuleRecord.rule_id).where(
                PolicyRuleRecord.tenant_id == tenant_id,
                PolicyRuleRecord.policy_bundle_id == bundle_id,
            )
        )
    )
    node_keys = set(
        await db.scalars(
            select(PolicyNode.node_key).where(
                PolicyNode.tenant_id == tenant_id, PolicyNode.policy_bundle_id == bundle_id
            )
        )
    )
    unknown = sorted(
        {r for case in cases for r in case.required_rules if r not in rule_ids}
        | {n for case in cases for n in case.relevant_evidence if n not in node_keys}
    )
    if unknown:
        raise EvalSetupError(f"labels not in the bundle: {', '.join(unknown)}")


def _ratio(results: Sequence[CaseResult], expected: str, missing: str) -> Ratio:
    total = sum(len(getattr(result, expected)) for result in results)
    lost = sum(len(getattr(result, missing)) for result in results)
    return Ratio(found=total - lost, expected=total)


def _cost(tokens: int, price_per_million: float | None) -> float | None:
    return None if price_per_million is None else tokens * price_per_million / 1_000_000


# ---- Running against a database (owner connection, rolled back)


@dataclass(frozen=True, slots=True)
class RunOptions:
    tenant_id: UUID
    bundle_id: UUID | None
    snapshot_id: UUID | None
    cases_path: Path
    mode: Mode
    price_per_million_tokens: float | None
    price_source: str | None


async def run(options: RunOptions, settings: RetrievalSettings) -> Report:
    case_file = load_cases(options.cases_path)
    embedder = choose_embedder(options.mode, settings)
    provider = AuroraPolicyContextProvider(settings, embedder)
    engine = create_async_engine(
        load_settings(OwnerDatabaseSettings).migration_database_url.get_secret_value(),
        poolclass=NullPool,
    )
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            await bind_tenant(db, options.tenant_id)
            try:
                bundle = await _bundle(db, options.tenant_id, options.bundle_id)
                snapshot_id = await _snapshot(db, options.tenant_id, options.snapshot_id)
                index_texts = await _searchable_texts(db, options.tenant_id, bundle[0])
                results = await evaluate(
                    db, options.tenant_id, bundle[0], snapshot_id, case_file.cases, provider
                )
            finally:
                # Measurement only: no manifest from this run is kept.
                await db.rollback()
    finally:
        await engine.dispose()
    model_id = embedder.model_id if embedder is not None else None
    return build_report(
        case_file,
        results,
        bundle,
        mode=RetrievalMode.FULL if embedder is not None else RetrievalMode.LEXICAL_ONLY,
        top_k=settings.policy_retrieval_top_k,
        cost=CostEstimate(
            model_id=model_id,
            query_tokens=estimate_tokens(c.text for c in case_file.cases) if model_id else 0,
            index_tokens=estimate_tokens(index_texts) if model_id else 0,
            price_per_million_tokens=options.price_per_million_tokens,
            price_source=options.price_source,
        ),
    )


def choose_embedder(mode: Mode, settings: RetrievalSettings) -> EmbeddingProvider | None:
    if mode == "lexical":
        return None
    embedder = load_embedder(settings)
    if embedder is None and mode == "full":
        raise EvalSetupError(
            "--mode full needs an embedder: set EMBEDDING_MODE to simulator (local or demo) or "
            "live, or use --mode lexical"
        )
    return embedder


async def _bundle(db: AsyncSession, tenant_id: UUID, bundle_id: UUID | None) -> tuple[UUID, int]:
    # The given bundle, or the tenant's active one (the highest ACTIVE version).
    query = select(PolicyBundle.id, PolicyBundle.version).where(PolicyBundle.tenant_id == tenant_id)
    if bundle_id is not None:
        query = query.where(PolicyBundle.id == bundle_id)
    else:
        query = query.where(PolicyBundle.lifecycle_state == BundleLifecycleState.ACTIVE)
    row = (await db.execute(query.order_by(PolicyBundle.version.desc()).limit(1))).one_or_none()
    if row is None:
        raise EvalSetupError("no such policy bundle for the tenant (or no active one)")
    return row.id, row.version


async def _snapshot(db: AsyncSession, tenant_id: UUID, snapshot_id: UUID | None) -> UUID:
    # The given study snapshot, or the tenant's most recent one.
    query = select(StudySnapshot.id).where(StudySnapshot.tenant_id == tenant_id)
    if snapshot_id is not None:
        query = query.where(StudySnapshot.id == snapshot_id)
    found = await db.scalar(
        query.order_by(StudySnapshot.created_at.desc(), StudySnapshot.id).limit(1)
    )
    if found is None:
        raise EvalSetupError("no such study snapshot for the tenant")
    return found


async def _searchable_texts(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> list[str]:
    # What an index build embeds, for the one-off index cost.
    return list(
        await db.scalars(
            select(PolicyNode.text).where(
                PolicyNode.tenant_id == tenant_id,
                PolicyNode.policy_bundle_id == bundle_id,
                PolicyNode.approved.is_(True),
                PolicyNode.node_type.in_(SEARCHABLE_NODE_TYPES),
            )
        )
    )


# ---- Output


def format_report(report: Report) -> str:
    lines = [
        f"Retrieval evaluation: bundle {report.policy_bundle_id} v{report.policy_bundle_version}, "
        f"mode {report.retrieval_mode.value}, model {report.embedding_model_id or 'none'}, "
        f"top_k {report.top_k}",
        f"Cases labelled by: {report.labelled_by}",
    ]
    if report.embedding_model_id is not None and report.embedding_model_id.startswith("simulator"):
        lines.append("Simulator vectors carry no meaning: semantic numbers don't measure a model.")
    lines += ["", _header(), *_row("all", report.overall)]
    lines += [line for lang, group in report.by_language.items() for line in _row(lang, group)]
    lines += ["", "Mandatory coverage by rule:"]
    lines += [f"  {rule_id:<10} {_pct(ratio)}" for rule_id, ratio in report.by_rule.items()]
    lines += ["", _cost_line(report.cost), "", "Misses:"]
    misses = [
        f"  {r.case_id}: "
        + "; ".join(
            f"{label} {', '.join(keys)}"
            for label, keys in (
                ("MANDATORY RULE MISSING", r.required_missing),
                ("evidence missing", r.relevant_missing),
                ("not found by search", r.search_missing),
            )
            if keys
        )
        for r in report.cases
        if r.required_missing or r.relevant_missing or r.search_missing
    ]
    lines += misses or ["  none"]
    return "\n".join(lines)


def report_json(report: Report) -> str:
    return json.dumps(asdict(report), indent=2, default=str)


def _header() -> str:
    return (
        f"{'group':<6} {'cases':>5}  {'mandatory':>13}  {'evidence':>13}  "
        f"{'search@k':>13}  {'p50 ms':>7}  {'p95 ms':>7}"
    )


def _row(name: str, group: Group) -> list[str]:
    return [
        f"{name:<6} {group.cases:>5}  {_pct(group.mandatory_coverage):>13}  "
        f"{_pct(group.evidence_recall):>13}  {_pct(group.search_recall):>13}  "
        f"{group.latency_p50_ms:>7.1f}  {group.latency_p95_ms:>7.1f}"
    ]


def _pct(ratio: Ratio) -> str:
    value = ratio.value
    shown = "n/a" if value is None else f"{value:.0%}"
    return f"{shown} ({ratio.found}/{ratio.expected})"


def _cost_line(cost: CostEstimate) -> str:
    if cost.model_id is None:
        return "Cost: no embedding calls (lexical only)."
    tokens = (
        f"~{cost.query_tokens} query tokens for these cases, ~{cost.index_tokens} to index the "
        f"bundle (estimated at {CHARS_PER_TOKEN} characters per token)"
    )
    if cost.price_per_million_tokens is None:
        return f"Cost: {tokens}; unpriced (pass --price-per-million-tokens and --price-source)."
    return (
        f"Cost: {tokens}; ${cost.query_cost_usd:.6f} queries, ${cost.index_cost_usd:.6f} index at "
        f"${cost.price_per_million_tokens}/1M tokens ({cost.price_source})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--bundle-id", type=UUID)
    parser.add_argument("--snapshot-id", type=UUID)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--mode", choices=("auto", "full", "lexical"), default="auto")
    parser.add_argument("--price-per-million-tokens", type=float)
    parser.add_argument("--price-source", help="Where the price comes from, with its date.")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if (args.price_per_million_tokens is None) != (args.price_source is None):
        parser.error("--price-per-million-tokens and --price-source go together")
    options = RunOptions(
        tenant_id=args.tenant_id,
        bundle_id=args.bundle_id,
        snapshot_id=args.snapshot_id,
        cases_path=args.cases,
        mode=args.mode,
        price_per_million_tokens=args.price_per_million_tokens,
        price_source=args.price_source,
    )
    try:
        report = asyncio.run(run(options, load_settings(RetrievalSettings)))
    except EvalSetupError as error:
        raise SystemExit(f"Evaluation not run: {error}") from None
    print(format_report(report))
    if args.json_out is not None:
        args.json_out.write_text(report_json(report), encoding="utf-8")
    if not report.coverage_complete:
        raise SystemExit("Mandatory-rule coverage is below 100%.")


if __name__ == "__main__":
    main()
