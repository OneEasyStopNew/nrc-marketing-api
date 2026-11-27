import copy
import itertools
import json
from pathlib import Path

import jsonschema
import pytest

from tests.nrc_policy_ref.contract import (
    BundleError,
    build_graph,
    content_sha256,
    evaluate,
    load_bundle,
    select_rules,
)

BUNDLE_PATH = Path(__file__).parents[2] / "scripts" / "data" / "nrc-ipb-candidate-bundle-v0.3.json"
FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "labelled_fixtures.json").read_text(encoding="utf-8")
)["fixtures"]
BUNDLE = load_bundle(BUNDLE_PATH)
BASE = next(f for f in FIXTURES if f["fixture_id"] == "FX-01")


def run(fixture, bundle=BUNDLE, **overrides):
    args = dict(
        semantic_results=fixture["semantic_results"],
        semantic_hits=tuple(fixture["semantic_hits"]),
        resolutions=fixture["resolutions"],
    )
    args.update(overrides)
    return evaluate(bundle, fixture["context"], tuple(fixture["stages"]), **args)


# ---- import guards ----
def test_draft_bundle_refused_at_runtime():
    with pytest.raises(BundleError, match="runtime needs APPROVED"):
        load_bundle(BUNDLE_PATH, mode="runtime")


def test_tampered_bundle_refused(tmp_path):
    data = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    data["rules"][1]["outcome_policy"]["mandatory"] = False
    (tmp_path / "b.json").write_text(json.dumps(data))
    with pytest.raises(BundleError, match="content_sha256"):
        load_bundle(tmp_path / "b.json")


def test_tampered_quotation_refused(tmp_path):
    data = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    data["rules"][1]["source_quotations"][0]["text"] += " (edited)"
    data["bundle"]["content_sha256"] = content_sha256(data)
    (tmp_path / "b.json").write_text(json.dumps(data))
    with pytest.raises(BundleError, match="quotation checksum"):
        load_bundle(tmp_path / "b.json")


def test_any_non_en_is_not_a_language():
    data = copy.deepcopy(BUNDLE)
    data["scope"]["enabled_languages"] = ["any_non_en"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            data,
            json.loads((Path(__file__).parent / "bundle.schema.json").read_text(encoding="utf-8")),
        )


# ---- structure ----
def test_source_quotes_never_used_as_post_wording():
    for rule in BUNDLE["rules"]:
        wording = rule["required_post_wording"]
        if wording:
            assert (
                wording["source"].startswith("study_snapshot.")
                and wording["nrc_supplies_text"] is False
            )
        assert all(q["use"] == "EVIDENCE_ONLY" for q in rule["source_quotations"])


def test_every_rule_has_explicit_outcomes():
    for rule in BUNDLE["rules"]:
        pol = rule["outcome_policy"]
        assert isinstance(pol["mandatory"], bool), rule["rule_id"]
        assert pol["on_violation"] and pol["on_missing_evidence"]
        if pol["mandatory"]:
            assert "UNABLE_TO_ASSESS" in pol["blocks_readiness_when"]


def test_scope_gates():
    scope = BUNDLE["scope"]
    assert scope["enabled_languages"] == ["en"]
    assert {"x", "reddit"}.isdisjoint(scope["enabled_channels"])


# ---- retrieval independent of semantic top-k ----
def test_mandatory_selection_independent_of_topk():
    stages = tuple(BASE["stages"])
    baseline = {
        s.rule_id
        for s in select_rules(BUNDLE, BASE["context"], stages)
        if s.status != "NOT_APPLICABLE"
    }
    every_id = [r["rule_id"] for r in BUNDLE["rules"]]
    for hits in ([], ["FED-09"], every_id, ["NOT-A-RULE"], list(reversed(every_id))):
        selected = select_rules(BUNDLE, BASE["context"], stages, tuple(hits))
        mandatory = {
            s.rule_id
            for s in selected
            if s.status != "NOT_APPLICABLE"
            and next(r for r in BUNDLE["rules"] if r["rule_id"] == s.rule_id)["outcome_policy"][
                "mandatory"
            ]
        }
        assert mandatory == baseline


# ---- missing context can never produce PASS ----
CONTEXT_KEYS = [
    "regulatory_applicability",
    "has_test_article",
    "irb_of_record",
    "l2_rules_status",
    "approved_languages",
    "approved_contacts",
    "required_disclosures",
]


@pytest.mark.parametrize("key", CONTEXT_KEYS)
def test_missing_study_context_blocks(key):
    fixture = copy.deepcopy(BASE)
    fixture["context"]["study"].pop(key)
    assert run(fixture).ready is False


@pytest.mark.parametrize(
    "pair",
    list(itertools.combinations(["language", "channel", "distribution", "creative_version_id"], 1)),
)
def test_missing_post_context_blocks(pair):
    fixture = copy.deepcopy(BASE)
    fixture["context"]["post"].pop(pair[0])
    assert run(fixture).ready is False


def test_dropping_any_semantic_result_blocks():
    applicable = {
        rid for rid, (outcome, _) in run(BASE).results.items() if outcome != "NOT_APPLICABLE"
    }
    for rid in BASE["semantic_results"]:
        if rid not in applicable:
            continue  # e.g. CH-02/CH-03 do not apply to a Facebook post
        results = {k: v for k, v in BASE["semantic_results"].items() if k != rid}
        assert run(BASE, semantic_results=results).ready is False, rid


# ---- labelled fixtures ----
@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["fixture_id"] for f in FIXTURES])
def test_labelled_fixture(fixture):
    result = run(fixture)
    assert result.ready is fixture["expected"]["ready"], result.results
    assert sorted(result.blocking) == fixture["expected"]["blocking"], result.results
    for rid, reason in fixture["expected"]["reasons"].items():
        assert result.results[rid][1] == reason
    assert sorted(result.basis_flags) == fixture["expected"]["basis_flags"], result.basis_flags


# ---- GraphRAG mapping ----
def test_graph_mapping():
    nodes, edges = build_graph(BUNDLE)
    rule_nodes = {n["id"] for n in nodes if n["type"] == "Rule"}
    assert "CH-06" not in rule_nodes  # retired
    assert not [e for e in edges if e["type"] == "HAS_EXCEPTION"]  # nothing approved yet
    _, sandbox_edges = build_graph(BUNDLE, include_proposed_exceptions=True)
    assert [e for e in sandbox_edges if e["type"] == "HAS_EXCEPTION"]
    sourced = {e["from"] for e in edges if e["type"] == "DERIVED_FROM"}
    assert {
        r["rule_id"] for r in BUNDLE["rules"] if r["provenance_status"] != "NRC_CONTROL_PROPOSED"
    } <= sourced


def test_reviewer_resolution_only_for_needs_review_rules():
    for rule in BUNDLE["rules"]:
        rr = rule["outcome_policy"]["review_resolution"]
        if rr["allowed"]:
            assert rule["outcome_policy"]["on_violation"] == "NEEDS_REVIEW", rule["rule_id"]


def test_paraphrased_rules_have_source_locations():
    for rule in BUNDLE["rules"]:
        if rule["provenance_status"] == "SOURCE_PARAPHRASED":
            assert rule["source_locations"], rule["rule_id"]


def test_rule_approval_records_pending():
    assert all(
        r["approval"]["status"] == "PENDING" and r["approval"]["approved_by"] is None
        for r in BUNDLE["rules"]
    )


def test_cross_references_resolve():
    ids = {r["rule_id"] for r in BUNDLE["rules"]}
    refs = {x["ref_id"] for x in BUNDLE["external_references"]}
    for d in BUNDLE["decisions"]:
        assert set(d["affects"]) - {"ALL"} <= ids, d["decision_id"]
    for c in BUNDLE["channel_readiness"]:
        assert set(c["paid"]["rules"]) | set(c["organic"]["rules"]) <= ids
        assert set(c["paid"]["sources"]) <= refs
    for r in BUNDLE["rules"]:
        assert {loc["source_ref"] for loc in r["source_locations"]} <= refs


def test_incomplete_traversal_blocks_every_mandatory_rule():
    fixture = copy.deepcopy(BASE)
    fixture["context"]["retrieval_manifest"] = {
        "gaps": [{"rule_id": "*", "type": "INCOMPLETE_TRAVERSAL"}]
    }
    result = run(fixture)
    assert result.ready is False
    mandatory = {r["rule_id"] for r in BUNDLE["rules"] if r["outcome_policy"]["mandatory"]}
    assert all(
        out[0] in ("UNABLE_TO_ASSESS", "NOT_APPLICABLE", "FAIL")
        for rid, out in result.results.items()
        if rid in mandatory
    )


def test_governance_roles_separated():
    roles = BUNDLE["governance"]["roles"]
    assert "activate a bundle" in roles["policy_owner"]["may_not"]
    assert any("approve" in x for x in roles["policy_admin"]["may_not"])
    assert BUNDLE["bundle"]["approval"]["required_role"] == "policy_owner"
    assert BUNDLE["bundle"]["activation"]["required_role"] == "policy_admin"


def test_required_wording_validated_by_its_own_approval_not_source_quotes():
    for rule in BUNDLE["rules"]:
        wording = rule["required_post_wording"]
        if wording:
            assert wording["provenance_type"] == "STUDY_APPROVED_WORDING"
            assert {"version", "approval", "text_sha256"} <= set(wording["fields"])


def test_tiktok_provisional():
    tiktok = next(c for c in BUNDLE["channel_readiness"] if c["channel"] == "tiktok")
    assert tiktok["paid"]["status"] == "PROVISIONAL_GATED"
    assert tiktok["paid"]["reason_code"] == "PAID_ELIGIBILITY_UNVERIFIED"
