"""Imported policy bundle documents (NRC-4, policy bundle v0.3).

- policy_bundles.contract_document and contract_sha256: a bundle imported from the policy
  deliverable keeps the whole document as imported, with its verified content_sha256, so the
  import is lossless and can be exported again. Both NULL for bundles built in the editor (CHECK
  contract_has_checksum).
- policy_rules.contract: each imported rule's own record (layer, stage, outcome_policy, approval,
  applicability, relations), which validation will read per rule.

Expand-only: nullable columns; the existing grants cover them (only the owner import writes them).

Revision ID: c4e7a2b9d150
Revises: 8e41b6d0a7c3
Create Date: 2026-11-26 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "c4e7a2b9d150"
down_revision: str | None = "8e41b6d0a7c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHECKSUM_CHECK = (
    "(contract_document IS NULL) = (contract_sha256 IS NULL)"
    " AND (contract_sha256 IS NULL OR contract_sha256 ~ '^[0-9a-f]{64}$')"
)


def upgrade() -> None:
    op.add_column("policy_bundles", sa.Column("contract_document", JSONB(), nullable=True))
    op.add_column("policy_bundles", sa.Column("contract_sha256", sa.String(64), nullable=True))
    op.create_check_constraint("contract_has_checksum", "policy_bundles", CHECKSUM_CHECK)
    op.add_column("policy_rules", sa.Column("contract", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("policy_rules", "contract")
    op.drop_constraint("contract_has_checksum", "policy_bundles", type_="check")
    op.drop_column("policy_bundles", "contract_sha256")
    op.drop_column("policy_bundles", "contract_document")
