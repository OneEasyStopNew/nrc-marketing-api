"""Reference implementation of the NRC policy-bundle contract (v0.2).

Covers: import guards, structured rule selection, deterministic checks, readiness aggregation, GraphRAG mapping.
Semantic judgments (model calls) are inputs here, not computed: tests feed labelled results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA = json.loads((Path(__file__).parent / "bundle.schema.json").read_text(encoding="utf-8"))
MISSING = object()


class BundleError(Exception):
    """Bundle failed an import guard."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(bundle: dict) -> str:
    body = {k: v for k, v in bundle.items() if k != "bundle"}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def load_bundle(path: str | Path, *, mode: str = "sandbox") -> dict:
    """mode='runtime' only accepts an approved, ACTIVE bundle; 'sandbox' accepts drafts for fixture tests."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    jsonschema.validate(bundle, SCHEMA)
    if content_sha256(bundle) != bundle["bundle"]["content_sha256"]:
        raise BundleError("content_sha256 mismatch: bundle edited after hashing")
    for rule in bundle["rules"]:
        for quote in rule["source_quotations"]:
            if _sha(quote["text"]) != quote["text_sha256"]:
                raise BundleError(f"{rule['rule_id']}: quotation checksum mismatch")
    if mode == "runtime":
        meta = bundle["bundle"]
        if (
            meta["status"] != "ACTIVE"
            or meta["activation"]["status"] != "ACTIVE"
            or meta["approval"]["status"] != "APPROVED"
        ):
            raise BundleError(
                f"bundle {meta['bundle_version']} is {meta['status']}; runtime needs APPROVED + ACTIVE"
            )
    elif mode != "sandbox":
        raise BundleError(f"unknown mode {mode}")
    return bundle


def _get(ctx: dict, dotted: str) -> Any:
    node: Any = ctx
    for part in dotted.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            return MISSING
        node = node[part]
    return node


@dataclass(frozen=True)
class Selection:
    rule_id: str
    status: str  # SELECTED | NOT_APPLICABLE | UNABLE_TO_ASSESS
    reason: str = ""
    basis_flag: str = ""


def _applies(rule: dict, ctx: dict) -> Selection:
    rid, appl = rule["rule_id"], rule["applicability"]
    flag = ""
    for key, cond in appl.items():
        if key == "study.regulatory_applicability":
            ra = _get(ctx, "study.regulatory_applicability")
            if ra is MISSING:
                return Selection(rid, "UNABLE_TO_ASSESS", "study.regulatory_applicability missing")
            if not any(ra.get(fw) is True for fw in cond["any_of"]):
                ext = appl.get("nrc_baseline_extension") or {}
                if ext.get("status") == "REJECTED" or not ext:
                    return Selection(rid, "NOT_APPLICABLE", "framework not applicable to study")
                flag = (
                    "APPLIED_VIA_APPROVED_NRC_BASELINE"
                    if ext.get("status") == "APPROVED"
                    else "APPLIED_VIA_PROPOSED_NRC_BASELINE"
                )
        elif key == "study.has_test_article":
            value = _get(ctx, "study.has_test_article")
            if value is MISSING:
                return Selection(rid, "UNABLE_TO_ASSESS", "study.has_test_article missing")
            if value is not cond:
                return Selection(rid, "NOT_APPLICABLE", "no test article")
        elif key == "study.has_compensation":
            study = ctx.get("study", {})
            if "compensation_statement" not in study:
                return Selection(rid, "UNABLE_TO_ASSESS", "study.compensation_statement missing")
            if (study["compensation_statement"] is not None) is not cond:
                return Selection(rid, "NOT_APPLICABLE", "no compensation")
        elif key in ("channel", "distribution"):
            value = _get(ctx, f"post.{key}")
            if value is MISSING:
                return Selection(rid, "UNABLE_TO_ASSESS", f"post.{key} missing")
            if value not in cond:
                return Selection(rid, "NOT_APPLICABLE", f"{key} {value} out of scope")
    return Selection(rid, "SELECTED", basis_flag=flag)


def select_rules(
    bundle: dict, ctx: dict, stages: tuple[str, ...], semantic_hits: tuple[str, ...] = ()
) -> list[Selection]:
    """Mandatory rules come only from structured predicates; semantic hits may add advisory rules, never remove any."""
    by_id = {r["rule_id"]: r for r in bundle["rules"]}
    out = [
        _applies(r, ctx)
        for r in bundle["rules"]
        if r["stage"] in stages and r["outcome_policy"]["mandatory"]
    ]
    for rid in semantic_hits:
        rule = by_id.get(rid)
        if rule and not rule["outcome_policy"]["mandatory"] and rule["stage"] in stages:
            out.append(_applies(rule, ctx))
    return out


URL_OR_PHONE = re.compile(r"https?://\S+|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|@\w+")
MONEY = re.compile(r"\$\s?\d")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _approved(item: dict) -> bool:
    """Approved wording carries its own provenance; it is never checked against source quotations."""
    appr = item.get("approval") or {}
    return bool(
        item.get("version")
        and item.get("study_id")
        and appr.get("approval_ref")
        and appr.get("approved_at")
        and item.get("text_sha256") == _sha(item.get("text", ""))
    )


GAP_OUTCOME = {
    "SOURCE_MISSING": "UNABLE_TO_ASSESS",
    "UNRESOLVED_APPLICABILITY": "UNABLE_TO_ASSESS",
    "INCOMPLETE_TRAVERSAL": "UNABLE_TO_ASSESS",
    "CONFLICT": "NEEDS_REVIEW",
}


def _gap_for(rule_id: str, ctx: dict) -> str | None:
    for gap in (ctx.get("retrieval_manifest") or {}).get("gaps", []):
        if gap["rule_id"] in (rule_id, "*"):
            return gap["type"]
    return None


def deterministic_check(rule: dict, ctx: dict, bundle: dict) -> tuple[str, str] | None:
    """Returns (outcome, reason) for checks this reference covers; None means a semantic result is required."""
    rid, pol, study, post = (
        rule["rule_id"],
        rule["outcome_policy"],
        ctx.get("study", {}),
        ctx.get("post", {}),
    )
    text = _norm(post.get("text", ""))
    if rid == "CTX-01":
        gaps = [p for p in rule["applicability"]["requires_context"] if _get(ctx, p) is MISSING]
        if study.get("l2_rules_status") not in (None, "RECORDED", "CONFIRMED_NONE"):
            gaps.append("study.l2_rules_status invalid")
        return ("PASS", "") if not gaps else (pol["on_violation"], "missing: " + ", ".join(gaps))
    if rid in ("FED-01", "FED-10"):
        att = ctx.get("attestation")
        ok = bool(
            att
            and att.get("reviewer_id")
            and att.get("attested_at_utc")
            and att.get("creative_version_id") == post.get("creative_version_id")
        )
        return ("PASS", "") if ok else (pol["on_violation"], pol["reason_code"])
    if rid == "FED-11":
        lang = post.get("language")
        if lang is None:
            return (pol["on_missing_evidence"], "post.language missing")
        ok = lang in bundle["scope"]["enabled_languages"] and lang in study.get(
            "approved_languages", []
        )
        return ("PASS", "") if ok else (pol["on_violation"], pol["reason_code"])
    if rid == "CTRL-01":
        approved = set(study.get("approved_contacts") or [])
        stray = [c for c in URL_OR_PHONE.findall(text) if c.rstrip(".,") not in approved]
        return ("PASS", "") if not stray else ("FAIL", f"unapproved contact(s): {stray}")
    if rid == "FACT-02":
        needed = [
            d
            for d in study.get("required_disclosures", [])
            if d["language"] == post.get("language")
        ]
        if not needed:
            return (pol["on_missing_evidence"], "no approved disclosure for this language")
        if not all(_approved(d) for d in needed):
            return (pol["on_missing_evidence"], "disclosure approval record incomplete")
        absent = [d["text"] for d in needed if _norm(d["text"]) not in text]
        return ("PASS", "") if not absent else ("FAIL", "required disclosure missing or altered")
    if rid == "FED-04":
        if not any(t in text.lower() for t in rule["check"]["triggers"]):
            return ("PASS", "")
        phrases = [
            p["text"]
            for p in study.get("approved_phrases", [])
            if p["phrase_type"] == "investigational_explanation"
            and p["language"] == post.get("language")
            and _approved(p)
        ]
        return (
            ("PASS", "")
            if any(_norm(p) in text for p in phrases)
            else ("FAIL", "trigger term without approved investigational explanation")
        )
    if rid == "FACT-01" and study.get("compensation_statement") is None and MONEY.search(text):
        return ("FAIL", "payment amount stated but study has no approved compensation")
    if rid == "CH-02":
        bad = {"customer_match", "your_data_segments", "audience_expansion", "lookalike"} & set(
            post.get("targeting", {}).get("audiences", [])
        )
        if bad:
            return ("FAIL", f"advertiser-curated audiences not allowed: {sorted(bad)}")
        certified = post.get("google_healthcare_certified")
        if certified is None:
            return (pol["on_missing_evidence"], "Google healthcare certification flag missing")
        return None if certified else ("FAIL", "no Google healthcare certification")
    if rid == "CTRL-02":
        return (
            ("FAIL", "health-condition targeting")
            if post.get("targeting", {}).get("health_conditions")
            else ("PASS", "")
        )
    if rid == "CH-01":
        low = text.lower()
        verb = re.search(
            r"\b(" + "|".join(map(re.escape, rule["check"]["cure_claim_terms"])) + r")\b", low
        )
        disease = next(
            (
                d
                for d in rule["check"]["cure_claim_diseases"]
                if re.search(rf"\b{re.escape(d)}", low)
            ),
            None,
        )
        return ("FAIL", f"cure claim for listed disease: {disease}") if verb and disease else None
    if rid == "CH-03":
        if (ctx.get("channel_eligibility", {}).get("tiktok") or {}).get("status") != "CONFIRMED":
            return (pol["on_missing_evidence"], "PAID_ELIGIBILITY_UNVERIFIED")
        min_age = post.get("targeting", {}).get("min_age")
        if min_age is None:
            return (pol["on_missing_evidence"], "targeting.min_age missing")
        return None if min_age >= 18 else (pol["on_violation"], pol["reason_code"])
    if rid in ("CH-04", "CH-05", "CH-07", "CH-08", "CH-09", "ORG-01"):
        return (pol["on_violation"], pol["reason_code"])
    return None


@dataclass
class Readiness:
    ready: bool
    results: dict[str, tuple[str, str]] = field(default_factory=dict)
    blocking: list[str] = field(default_factory=list)
    basis_flags: dict[str, str] = field(default_factory=dict)


def evaluate(
    bundle: dict,
    ctx: dict,
    stages: tuple[str, ...],
    semantic_results: dict[str, str] | None = None,
    semantic_hits: tuple[str, ...] = (),
    resolutions: dict[str, dict] | None = None,
) -> Readiness:
    semantic_results, resolutions = semantic_results or {}, resolutions or {}
    by_id = {r["rule_id"]: r for r in bundle["rules"]}
    res = Readiness(ready=True)
    for sel in select_rules(bundle, ctx, stages, semantic_hits):
        rule = by_id[sel.rule_id]
        pol = rule["outcome_policy"]
        if sel.status != "SELECTED":
            outcome = (sel.status, sel.reason)
        else:
            outcome = deterministic_check(rule, ctx, bundle) or (
                (
                    pol["on_violation"]
                    if semantic_results[sel.rule_id] == "VIOLATION"
                    else semantic_results[sel.rule_id],
                    "semantic",
                )
                if sel.rule_id in semantic_results
                else ("NOT_EVALUATED", "no semantic result")
            )
        if sel.basis_flag:
            res.basis_flags[sel.rule_id] = sel.basis_flag
        if outcome[0] == "NEEDS_REVIEW" and sel.rule_id in resolutions:
            r = resolutions[sel.rule_id]
            allowed = pol["review_resolution"]
            if (
                allowed.get("allowed")
                and allowed.get("resolves_outcome") == "NEEDS_REVIEW"
                and r.get("role") == allowed["role"]
                and r.get("rationale")
                and r.get("to") in allowed["may_resolve_to"]
            ):
                outcome = (r["to"], f"resolved by {r['role']}")
        gap = _gap_for(sel.rule_id, ctx) if pol["mandatory"] else None
        if (
            gap and outcome[0] != "FAIL"
        ):  # gaps never soften a FAIL, and are applied after review resolution
            outcome = (GAP_OUTCOME[gap], gap)
        res.results[sel.rule_id] = outcome
        if pol["mandatory"] and outcome[0] not in ("PASS", "NOT_APPLICABLE"):
            res.ready = False
            res.blocking.append(sel.rule_id)
    if "context" in stages and ctx.get("study", {}).get("l2_rules_status") == "RECORDED":
        l2 = ctx.get(
            "l2_results"
        )  # results of the study-scoped L2 bundle, evaluated in the same run
        if not l2:
            res.results["L2"] = ("UNABLE_TO_ASSESS", "L2_NOT_EVALUATED")
            res.ready = False
            res.blocking.append("L2")
        for rid, out in (l2 or {}).items():
            res.results[f"L2:{rid}"] = (out, "study-scoped L2 rule")
            if out not in ("PASS", "NOT_APPLICABLE"):
                res.ready = False
                res.blocking.append(f"L2:{rid}")
    return res


def build_graph(
    bundle: dict, *, include_proposed_exceptions: bool = False
) -> tuple[list[dict], list[dict]]:
    """Nodes/edges per graphrag_mapping. Only APPROVED exceptions become HAS_EXCEPTION edges at runtime."""
    nodes = [{"id": s["ref_id"], "type": "SourceReference"} for s in bundle["external_references"]]
    nodes += [
        {"id": d["decision_id"], "type": "Decision", "status": d["status"]}
        for d in bundle["decisions"]
    ]
    edges: list[dict] = []
    for r in bundle["rules"]:
        rid = r["rule_id"]
        nodes.append(
            {
                "id": rid,
                "type": "Rule",
                "mandatory": r["outcome_policy"]["mandatory"],
                "stage": r["stage"],
                "enabled": r["enabled_in_nrc"],
            }
        )
        edges += [
            {"from": rid, "type": "DERIVED_FROM", "to": s}
            for s in r["relations"].get("derived_from", [])
        ]
        for i, quote in enumerate(r["source_quotations"]):
            qid = f"{rid}-Q{i + 1}"
            nodes.append({"id": qid, "type": "SourceQuotation", "sha256": quote["text_sha256"]})
            edges.append({"from": rid, "type": "QUOTES", "to": qid})
        for fw in r["applicability"].get("study.regulatory_applicability", {}).get("any_of", []):
            edges.append({"from": rid, "type": "APPLIES_WHEN_FRAMEWORK", "to": fw})
        edges += [
            {"from": rid, "type": "APPLIES_TO_CHANNEL", "to": c}
            for c in r["applicability"].get("channel", [])
        ]
        for exc in r["exceptions"]:
            if exc["status"] == "APPROVED" or include_proposed_exceptions:
                nodes.append(
                    {"id": exc["exception_id"], "type": "Exception", "status": exc["status"]}
                )
                edges.append({"from": rid, "type": "HAS_EXCEPTION", "to": exc["exception_id"]})
        if r["required_post_wording"]:
            edges.append(
                {
                    "from": rid,
                    "type": "REQUIRES_DISCLOSURE",
                    "to": r["required_post_wording"]["source"],
                }
            )
        for kind in ("interprets", "implements"):
            edges += [
                {"from": rid, "type": kind.upper(), "to": t} for t in r["relations"].get(kind, [])
            ]
        ref = r["basis"].get("decision_ref")
        if ref:
            edges.append({"from": rid, "type": "DECIDED_BY", "to": ref})
    return nodes, edges
