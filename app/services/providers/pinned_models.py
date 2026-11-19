# Approved Higgsfield models and their pinned settings (F24, Vishnu's review of 17 Nov 2026).
# Plain data with no client imports, so the API can list models and check requests without
# loading the provider adapter (provider calls run only in the worker).

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class PinnedModel:
    """The approved configuration of one model (F24, Vishnu's review of 17 Nov 2026).

    Every request to the model sends exactly `params`, never a provider default and never a
    caller's override, so the free /estimate prices the same body that is submitted.
    """

    aspect_ratios: frozenset[str]
    params: Mapping[str, str]
    # The whole prompt sent (template plus brief): the provider's limit, or ours if it has none.
    max_prompt_chars: int
    # TODO(F28, Firoz): temporary room kept for the server-side style template, which isn't built
    # yet. Remove when templates are stored per version: compute the brief room from the actual
    # template length instead, and update test_brief_room_leaves_space_for_the_template.
    template_reserve_chars: int
    # Text-in-image model: used only when the draft carries text to show in the image.
    text_in_image: bool
    # Shown in the web app's model picker, so the frontend never hard-codes model names.
    label: str
    # Provider estimate for one image at 1:1 with a typical template-plus-brief prompt, and the
    # day it was checked. An estimate, never the billed amount (that comes from reconciliation).
    estimated_usd_per_image: Decimal
    price_checked_at: date
    # False when the provider prices by tokens, so the actual estimate varies with the prompt
    # and ratio; each job is still estimated exactly before submission.
    price_is_fixed: bool


# Probed for free with /estimate (10, 16 and 17 Nov 2026) and confirmed by paid runs. Add a model
# only after probing its ratios and prompt limit with the smoke script's estimate.
PINNED_MODELS: Final[Mapping[str, PinnedModel]] = MappingProxyType(
    {
        # Soul 2: flat $0.006 at 1080p for prompts of 8 to 20,000 characters and every ratio
        # below; no provider prompt limit was found, so 5,000 is our bound.
        "higgsfield-ai/soul/v2/standard": PinnedModel(
            aspect_ratios=frozenset({"9:16", "16:9", "4:3", "3:4", "1:1", "2:3", "3:2"}),
            params=MappingProxyType({"resolution": "1080p"}),
            max_prompt_chars=5000,
            template_reserve_chars=1200,
            text_in_image=False,
            label="Soul 2",
            estimated_usd_per_image=Decimal("0.006"),
            price_checked_at=date(2026, 11, 18),
            price_is_fixed=True,
        ),
        # Marketing Studio Image: token-priced underneath; low quality at 2k is $0.018-0.048 by
        # ratio and prompt length. Its default (high, 2k) is $0.444, so quality is always sent.
        # 5,000 characters is the provider's limit (5,001 is a 400).
        "marketing-studio/image": PinnedModel(
            aspect_ratios=frozenset({"1:1", "9:16", "16:9", "4:3", "3:4"}),
            params=MappingProxyType({"quality": "low", "resolution": "2k"}),
            max_prompt_chars=5000,
            template_reserve_chars=1200,
            text_in_image=False,
            # $0.028 at 1:1 for the ~1,000-character tmpl-v2 prompts (18 Nov); 9:16 $0.022-0.023.
            label="Marketing Studio",
            estimated_usd_per_image=Decimal("0.028"),
            price_checked_at=date(2026, 11, 18),
            price_is_fixed=False,
        ),
        # Ideogram 4.0: text in images. Only rendering_speed changes its price (DEFAULT $0.060);
        # it has no resolution field. 2,048 characters is the provider's limit.
        "ideogram/v4.0": PinnedModel(
            aspect_ratios=frozenset({"1:1", "9:16", "16:9", "4:5", "3:4"}),
            params=MappingProxyType({"rendering_speed": "DEFAULT"}),
            max_prompt_chars=2048,
            template_reserve_chars=600,
            text_in_image=True,
            label="Ideogram 4.0",
            estimated_usd_per_image=Decimal("0.060"),
            price_checked_at=date(2026, 11, 18),
            price_is_fixed=True,
        ),
    }
)
# Aspect ratios each pinned model accepts; any other ratio is a 400 from Higgsfield.
SUPPORTED_ASPECT_RATIOS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {model: pinned.aspect_ratios for model, pinned in PINNED_MODELS.items()}
)


def max_brief_chars(model: str) -> int:
    # Characters left for the creator's brief once the template's room is kept.
    pinned = PINNED_MODELS[model]
    return pinned.max_prompt_chars - pinned.template_reserve_chars
