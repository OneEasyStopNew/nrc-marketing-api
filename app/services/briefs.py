"""Brief checks beyond single fields (NRC-2).

Saving a draft checks each field on its own (length, https, CTA value) plus the site against the
study snapshot the draft is pinned to. Rules that span fields, such as "a post with a link needs a
CTA", never block a save: drafts autosave while the user is still filling them in. They are
reported as brief_issues instead, and approval (NRC-12) refuses a version that still has any.
"""

import unicodedata
from collections.abc import Mapping

from app.core.errors import FieldValidationError
from app.models.enums import FORMAT_SPECS, BriefField, PostFormat
from app.schemas.creatives import BriefIssue, BriefIssueCode
from app.schemas.tenancy import StudyFacts

SITE_NOT_APPROVED = "site_not_approved"


def brief_issues(
    post_format: PostFormat, values: Mapping[BriefField, object | None]
) -> list[BriefIssue]:
    """The link fields still empty when some are filled, in BriefField order."""
    link_fields = FORMAT_SPECS[post_format].link_fields
    if all(values.get(field) is None for field in link_fields):
        return []
    return [
        BriefIssue(field=field, code=BriefIssueCode.REQUIRED_WITH_LINK)
        for field in BriefField
        if field in link_fields and values.get(field) is None
    ]


def check_site(site: str | None, facts: Mapping[str, object]) -> None:
    """Refuse a site that isn't one of the snapshot's approved sites (exact match after NFC).

    Raises:
        FieldValidationError: the site is not in the snapshot's sites.
    """
    if site is None:
        return
    approved = {
        unicodedata.normalize("NFC", name).strip()
        for name in StudyFacts.model_validate(facts).sites
    }
    if site not in approved:
        raise FieldValidationError(
            BriefField.SITE.value,
            "must be one of the study's approved sites",
            SITE_NOT_APPROVED,
        )
