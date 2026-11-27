"""Writes fixtures/labelled_fixtures.json. Each fixture is labelled with the expected readiness and blocking rules."""
import copy
import hashlib
import json
from pathlib import Path

APPROVAL = {"approval_ref": "FIXTURE-IRB-APPROVED-MATERIALS-v1", "approved_by": "fixture-approver", "approved_at": "2026-11-20T00:00:00Z",
            "approval_source": "IRB-approved study materials"}
STUDY = {
    "id": "FIXTURE-HTN-01",
    "regulatory_applicability": {"fda_21cfr50_56": True, "fda_ind_21cfr312": True, "fda_ide_21cfr812": False, "hhs_45cfr46": False,
                                 "approved_by": "fixture-approver", "approved_at": "2026-11-20T00:00:00Z"},
    "has_test_article": True, "product_type": "drug", "irb_of_record": "Fixture IRB", "l2_rules_status": "CONFIRMED_NONE",
    "approved_languages": ["en"], "approved_contacts": ["https://example.org/htn-study"],
    "required_disclosures": [{"disclosure_id": "D1", "study_id": "FIXTURE-HTN-01", "language": "en", "version": "1",
                              "text": "This is a research study. Taking part is voluntary.", "approval": APPROVAL}],
    "approved_phrases": [{"phrase_id": "P1", "phrase_type": "investigational_explanation", "study_id": "FIXTURE-HTN-01", "language": "en", "version": "1",
                          "text": "The study medicine is investigational, which means it is not approved by the FDA.", "approval": APPROVAL}],
    "compensation_statement": None, "facts": {"condition": "high blood pressure"},
}
POST = {"channel": "facebook", "distribution": "paid", "language": "en", "creative_version_id": "v3",
        "text": "Adults with high blood pressure may be eligible for a research study. This is a research study. Taking part is voluntary. Learn more: https://example.org/htn-study",
        "targeting": {"audiences": [], "health_conditions": []}}
ATTEST = {"reviewer_id": "rev-1", "attested_at_utc": "2026-11-20T10:00:00Z", "creative_version_id": "v3"}
SEMANTIC_PASS = {r: "PASS" for r in ("FED-02", "FED-03", "FED-05", "FED-06", "FED-08", "FACT-01", "IMG-01", "CH-01", "CH-02", "CH-03")}
for _item in STUDY["required_disclosures"] + STUDY["approved_phrases"]:
    _item["text_sha256"] = hashlib.sha256(_item["text"].encode("utf-8")).hexdigest()
ALL = ["context", "validation", "reviewer_gate", "publish_readiness"]
TIKTOK_OK = {"tiktok": {"status": "CONFIRMED", "evidence_ref": "fixture-tiktok-confirmation", "confirmed_by": "fixture", "confirmed_at": "2026-11-23T00:00:00Z"}}
AGE18 = {"audiences": [], "health_conditions": [], "min_age": 18}


def fx(fid, label, *, study=None, post=None, attestation=ATTEST, semantic=None, hits=(), resolutions=None, l2=None, elig=None, manifest=None,
       ready, blocking=(), reasons=None, flags=()):
    ctx = {"study": study if study is not None else copy.deepcopy(STUDY), "post": post if post is not None else copy.deepcopy(POST)}
    if attestation is not None:
        ctx["attestation"] = attestation
    if l2 is not None:
        ctx["l2_results"] = l2
    if elig is not None:
        ctx["channel_eligibility"] = elig
    if manifest is not None:
        ctx["retrieval_manifest"] = manifest
    return {"fixture_id": fid, "label": label, "stages": ALL, "context": ctx,
            "semantic_results": SEMANTIC_PASS if semantic is None else semantic, "semantic_hits": list(hits),
            "resolutions": resolutions or {}, "expected": {"ready": ready, "blocking": sorted(blocking), "reasons": reasons or {}, "basis_flags": sorted(flags)}}


def with_(base, **changes):
    out = copy.deepcopy(base)
    out.update(changes)
    return out


def without(base, key):
    out = copy.deepcopy(base)
    out.pop(key)
    return out


F = [
    fx("FX-01", "Compliant English Facebook paid post, full context, valid attestation", ready=True),
    fx("FX-02", "Same post; semantic top-k returns nothing: mandatory rules still selected", hits=(), ready=True),
    fx("FX-03", "Same post; top-k returns only advisory FED-09 (advisory violation does not block)", hits=("FED-09",),
       semantic=with_(SEMANTIC_PASS, **{"FED-09": "VIOLATION"}), ready=True),
    fx("FX-04", "IRB of record missing: no PASS possible", study=without(STUDY, "irb_of_record"), ready=False, blocking=["CTX-01"]),
    fx("FX-05", "has_test_article unknown: FDA content rules cannot be ruled out", study=without(STUDY, "has_test_article"),
       ready=False, blocking=["CTX-01", "FED-02", "FED-03", "FED-04"]),
    fx("FX-06", "Regulatory applicability missing", study=without(STUDY, "regulatory_applicability"), ready=False,
       blocking=["CTX-01", "FED-01", "FED-02", "FED-03", "FED-04", "FED-05", "FED-06", "FED-07", "FED-08", "FED-10", "FED-11", "IMG-01"]),
    fx("FX-07", "Semantic judge returned nothing for FED-05", semantic=without(SEMANTIC_PASS, "FED-05"), ready=False, blocking=["FED-05"]),
    fx("FX-08", "FED-08 flagged NEEDS_REVIEW, unresolved", semantic=with_(SEMANTIC_PASS, **{"FED-08": "VIOLATION"}), ready=False, blocking=["FED-08"]),
    fx("FX-09", "FED-08 NEEDS_REVIEW resolved to PASS by policy reviewer with rationale", semantic=with_(SEMANTIC_PASS, **{"FED-08": "VIOLATION"}),
       resolutions={"FED-08": {"role": "nrc_policy_reviewer", "to": "PASS", "rationale": "Phrase 'limited spots' is factual per snapshot"}}, ready=True),
    fx("FX-10", "FED-08 resolution without rationale is ignored", semantic=with_(SEMANTIC_PASS, **{"FED-08": "VIOLATION"}),
       resolutions={"FED-08": {"role": "nrc_policy_reviewer", "to": "PASS"}}, ready=False, blocking=["FED-08"]),
    fx("FX-11", "Spanish post: language gated in v0.2", post=with_(POST, language="es"), ready=False, blocking=["FACT-02", "FED-11"]),
    fx("FX-12", "TikTok paid (US), 18+, no recorded eligibility confirmation", post=with_(POST, channel="tiktok", targeting=AGE18),
       ready=False, blocking=["CH-03"], reasons={"CH-03": "PAID_ELIGIBILITY_UNVERIFIED"}),
    fx("FX-13", "TikTok organic: paid rule not applied, organic unsourced", post=with_(POST, channel="tiktok", distribution="organic"),
       ready=False, blocking=["ORG-01"], reasons={"ORG-01": "ORGANIC_POLICY_UNSOURCED"}),
    fx("FX-14", "X channel request: not enabled", post=with_(POST, channel="x"), ready=False, blocking=["CH-08"], reasons={"CH-08": "CHANNEL_NOT_ENABLED"}),
    fx("FX-15", "Stale attestation: approved v2, publishing v3", attestation=with_(ATTEST, creative_version_id="v2"), ready=False,
       blocking=["FED-01", "FED-10"], reasons={"FED-01": "NO_VALID_ATTESTATION"}),
    fx("FX-16", "No attestation at all", attestation=None, ready=False, blocking=["FED-01", "FED-10"]),
    fx("FX-17", "'$50' with no approved compensation (v0.1 demo bug)", post=with_(POST, text=POST["text"] + " Get paid $50 for your time."),
       ready=False, blocking=["FACT-01"]),
    fx("FX-18", "Unapproved link in post", post=with_(POST, text=POST["text"] + " Or visit https://other.example.com"), ready=False, blocking=["CTRL-01"]),
    fx("FX-19", "'New treatment' without approved investigational explanation", post=with_(POST, text=POST["text"] + " Try a new treatment."),
       ready=False, blocking=["FED-04"]),
    fx("FX-20", "'New treatment' with approved investigational explanation",
       post=with_(POST, text=POST["text"] + " Try a new treatment. The study medicine is investigational, which means it is not approved by the FDA."), ready=True),
    fx("FX-21", "YouTube paid with customer match audience", post=with_(POST, channel="youtube", google_healthcare_certified=True,
       targeting={"audiences": ["customer_match"], "health_conditions": []}), ready=False, blocking=["CH-02"]),
    fx("FX-22", "YouTube paid, certified, predefined audiences only", post=with_(POST, channel="youtube", google_healthcare_certified=True,
       targeting={"audiences": ["in_market"], "health_conditions": []}), ready=True),
    fx("FX-23", "YouTube paid, certification flag missing", post=with_(POST, channel="youtube", targeting={"audiences": [], "health_conditions": []}),
       ready=False, blocking=["CH-02"]),
    fx("FX-24", "Facebook paid with health-condition targeting (NRC control)", post=with_(POST, targeting={"audiences": [], "health_conditions": ["hypertension"]}),
       ready=False, blocking=["CTRL-02"]),
    fx("FX-25", "Required disclosure altered", post=with_(POST, text=POST["text"].replace("voluntary", "optional")), ready=False, blocking=["FACT-02"]),
    fx("FX-26", "L2 rule status blank is not 'none'", study=with_(STUDY, l2_rules_status=None), ready=False, blocking=["CTX-01"]),
    fx("FX-27", "HHS-only study (no FDA): FDA-derived rules applied via proposed NRC baseline and flagged",
       study=with_(STUDY, regulatory_applicability=with_(STUDY["regulatory_applicability"], fda_21cfr50_56=False, fda_ind_21cfr312=False, hhs_45cfr46=True)),
       ready=True, flags=["FED-02", "FED-03", "FED-04", "FED-05", "FED-06", "FED-08", "IMG-01"]),
    fx("FX-28", "IRB-specific (L2) rules recorded but not evaluated", study=with_(STUDY, l2_rules_status="RECORDED"), ready=False, blocking=["L2"],
       reasons={"L2": "L2_NOT_EVALUATED"}),
    fx("FX-29", "IRB-specific (L2) rules recorded and passing", study=with_(STUDY, l2_rules_status="RECORDED"), l2={"IRB-ACME-01": "PASS"}, ready=True),
    fx("FX-30", "IRB-specific (L2) rule failing", study=with_(STUDY, l2_rules_status="RECORDED"), l2={"IRB-ACME-01": "FAIL"}, ready=False, blocking=["L2:IRB-ACME-01"]),
    fx("FX-32", "TikTok paid (US), eligibility confirmed, no 18+ restriction", post=with_(POST, channel="tiktok", targeting={"audiences": [], "health_conditions": [], "min_age": 13}),
       elig=TIKTOK_OK, ready=False, blocking=["CH-03"], reasons={"CH-03": "TIKTOK_US_CONDITION_NOT_MET"}),
    fx("FX-33", "TikTok paid (US), eligibility confirmed, no age setting declared", post=with_(POST, channel="tiktok"), elig=TIKTOK_OK, ready=False, blocking=["CH-03"],
       reasons={"CH-03": "targeting.min_age missing"}),
    fx("FX-34", "Facebook paid: cure claim for a disease on Meta's list", post=with_(POST, text=POST["text"] + " Our study medicine may cure diabetes."),
       ready=False, blocking=["CH-01"], reasons={"CH-01": "cure claim for listed disease: diabetes"}),
    fx("FX-35", "TikTok paid (US), eligibility confirmed, 18+", post=with_(POST, channel="tiktok", targeting=AGE18), elig=TIKTOK_OK, ready=True),
    fx("FX-36", "Deterministic disclosure check passes but retrieval reports its source missing", manifest={"gaps": [{"rule_id": "FACT-02", "type": "SOURCE_MISSING"}]},
       ready=False, blocking=["FACT-02"], reasons={"FACT-02": "SOURCE_MISSING"}),
    fx("FX-37", "Retrieval reports a rule conflict on FED-05", manifest={"gaps": [{"rule_id": "FED-05", "type": "CONFLICT"}]},
       ready=False, blocking=["FED-05"], reasons={"FED-05": "CONFLICT"}),
    fx("FX-38", "Retrieval gap does not soften a FAIL", post=with_(POST, text=POST["text"] + " Get paid $50 for your time."),
       manifest={"gaps": [{"rule_id": "FACT-01", "type": "SOURCE_MISSING"}]}, ready=False, blocking=["FACT-01"],
       reasons={"FACT-01": "payment amount stated but study has no approved compensation"}),
    fx("FX-39", "Unresolved applicability reported for FED-02", manifest={"gaps": [{"rule_id": "FED-02", "type": "UNRESOLVED_APPLICABILITY"}]},
       ready=False, blocking=["FED-02"], reasons={"FED-02": "UNRESOLVED_APPLICABILITY"}),
    fx("FX-40", "Required disclosure lacks an approval record", study=with_(STUDY, required_disclosures=[without(STUDY["required_disclosures"][0], "approval")]),
       ready=False, blocking=["FACT-02"], reasons={"FACT-02": "disclosure approval record incomplete"}),
    fx("FX-41", "Reviewer resolution cannot clear a retrieval conflict", manifest={"gaps": [{"rule_id": "FED-08", "type": "CONFLICT"}]},
       semantic=with_(SEMANTIC_PASS, **{"FED-08": "VIOLATION"}),
       resolutions={"FED-08": {"role": "nrc_policy_reviewer", "to": "PASS", "rationale": "reviewed"}}, ready=False, blocking=["FED-08"], reasons={"FED-08": "CONFLICT"}),
    fx("FX-31", "Reviewer cannot override a FAIL (FED-02)", semantic=with_(SEMANTIC_PASS, **{"FED-02": "VIOLATION"}),
       resolutions={"FED-02": {"role": "nrc_policy_reviewer", "to": "PASS", "rationale": "Judge false positive"}}, ready=False, blocking=["FED-02"]),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures" / "labelled_fixtures.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"schema": "nrc.policy-fixtures/0.2", "bundle_version": "0.3.0", "fixtures": F}, indent=2))
    print(len(F), "fixtures ->", out)
