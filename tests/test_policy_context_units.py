"""Pure parts of policy retrieval (NRC-5): the condition evaluator and the bounded walk."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.models.enums import (
    Applicability,
    Channel,
    PolicyEdgeType,
    PolicyLanguage,
    PostFormat,
    ValidationInputKind,
)
from app.schemas.tenancy import StudyFacts
from app.services.policy_context.applicability import Conditions, evaluate
from app.services.policy_context.graph import APPLIED_BY, adjacency, walk
from app.services.policy_context.manifest import ContextScope
from scripts.seed_demo import SYNTHETIC_FACTS

AT = datetime(2026, 11, 18, 12, tzinfo=UTC)
FACTS = StudyFacts.model_validate(SYNTHETIC_FACTS)
STUDY = uuid4()
SCOPE = ContextScope(
    input_kind=ValidationInputKind.TEXT,
    channel=Channel.FACEBOOK,
    language=PolicyLanguage.EN,
    evaluated_at=AT,
)


def test_no_conditions_always_apply() -> None:
    assert evaluate(Conditions(), SCOPE, STUDY, FACTS).applies is Applicability.TRUE


def test_effective_end_is_exclusive() -> None:
    ends_now = evaluate(Conditions(effective_to=AT), SCOPE, STUDY, FACTS)
    starts_now = evaluate(Conditions(effective_from=AT), SCOPE, STUDY, FACTS)

    assert ends_now.reasons == ("NO_LONGER_EFFECTIVE",)
    assert starts_now.applies is Applicability.TRUE


def test_a_failed_condition_wins_over_an_unknown_one() -> None:
    verdict = evaluate(Conditions(audiences=("adults",), channels=("tiktok",)), SCOPE, STUDY, FACTS)

    assert verdict.applies is Applicability.FALSE
    assert verdict.reasons == ("CHANNEL_NOT_MATCHED",)
    assert verdict.unresolved == ("audience",)


def test_post_format_restriction_matches_or_stays_unresolved() -> None:
    story_only = Conditions(post_formats=("story_image",))
    feed = SCOPE.model_copy(update={"post_format": PostFormat.FEED_IMAGE})

    assert evaluate(story_only, feed, STUDY, FACTS).reasons == ("POST_FORMAT_NOT_MATCHED",)
    assert evaluate(story_only, SCOPE, STUDY, FACTS).unresolved == ("post_format",)


def test_unknown_study_fact_name_is_unresolved_not_dropped() -> None:
    verdict = evaluate(Conditions(required_study_fact="no_such_fact"), SCOPE, STUDY, FACTS)

    assert verdict.applies is Applicability.UNRESOLVED
    assert verdict.unresolved == ("study_fact:no_such_fact",)


def test_a_zero_study_fact_counts_as_present() -> None:
    zero_age = FACTS.model_copy(update={"min_age": 0})
    no_age = FACTS.model_copy(update={"min_age": None})

    assert evaluate(Conditions(required_study_fact="min_age"), SCOPE, STUDY, zero_age).applies is (
        Applicability.TRUE
    )
    assert evaluate(Conditions(required_study_fact="min_age"), SCOPE, STUDY, no_age).reasons == (
        "STUDY_FACT_MISSING:min_age",
    )


def test_study_restriction_matches_the_snapshot_study() -> None:
    other = evaluate(Conditions(study_ids=(uuid4(),)), SCOPE, STUDY, FACTS)
    same = evaluate(Conditions(study_ids=(STUDY,)), SCOPE, STUDY, FACTS)

    assert other.reasons == ("STUDY_NOT_MATCHED",)
    assert same.applies is Applicability.TRUE


def _ids(*keys: str) -> dict[str, UUID]:
    return {key: uuid4() for key in keys}


def test_walk_is_breadth_first_sorted_and_follows_applies_to_backwards() -> None:
    ids = _ids("R", "B", "A", "INT", "C")
    keys = {node_id: key for key, node_id in ids.items()}
    edges = [
        (PolicyEdgeType.SUPPORTED_BY, ids["R"], ids["B"]),
        (PolicyEdgeType.SUPPORTED_BY, ids["R"], ids["A"]),
        (PolicyEdgeType.APPLIES_TO, ids["INT"], ids["R"]),
        (PolicyEdgeType.SUPPORTED_BY, ids["A"], ids["C"]),
        (PolicyEdgeType.SUPERSEDES, ids["R"], ids["C"]),
    ]

    result = walk(ids["R"], adjacency(edges, keys), keys, max_hops=3)

    assert [step.path for step in result.reached] == [
        ("R", APPLIED_BY, "INT"),
        ("R", "supported_by", "A"),
        ("R", "supported_by", "B"),
        ("R", "supported_by", "A", "supported_by", "C"),
    ]
    assert result.truncated is False


def test_walk_stops_at_the_hop_limit_and_says_so() -> None:
    ids = _ids("R", "A", "B")
    keys = {node_id: key for key, node_id in ids.items()}
    edges = [
        (PolicyEdgeType.EXCEPT_WHEN, ids["R"], ids["A"]),
        (PolicyEdgeType.EXCEPT_WHEN, ids["A"], ids["B"]),
    ]

    result = walk(ids["R"], adjacency(edges, keys), keys, max_hops=1)

    assert [step.path[-1] for step in result.reached] == ["A"]
    assert result.truncated is True


def test_edges_to_unknown_nodes_are_ignored() -> None:
    ids = _ids("R")
    keys = {ids["R"]: "R"}
    graph = adjacency([(PolicyEdgeType.SUPPORTED_BY, ids["R"], uuid4())], keys)

    assert walk(ids["R"], graph, keys, max_hops=3).reached == ()
