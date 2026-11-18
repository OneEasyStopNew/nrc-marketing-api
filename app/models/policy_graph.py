# The policy graph (NRC-5): sources, clauses, nodes, rules, edges and the retrieval index.
#
# Rows of one bundle are immutable once the bundle is active. Owner scripts write them until the
# NRC-4 admin APIs exist; the API and worker roles only read. Every link is a composite foreign key
# that includes tenant_id (and policy_bundle_id inside the graph), so a row can never point at
# another tenant's or another bundle's row.

from datetime import datetime
from typing import Final
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import (
    CheckType,
    Gate,
    IndexBuildState,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    Severity,
    enum_column,
)

# Titan Text Embeddings V2, Cohere Embed Multilingual and Cohere Embed v4 all support 1024, so
# the column does not depend on which model the evaluation picks.
EMBEDDING_DIMENSIONS = 1024
# Same pattern as app.schemas.policies.PolicyRule.rule_id (which imports it from here).
RULE_ID_PATTERN = r"^[A-Z][A-Z0-9-]{1,39}$"
# What search returns and the embedding index covers: supporting text. Rules are selected by
# their conditions and exceptions reached through the graph, so neither is searched or embedded.
SEARCHABLE_NODE_TYPES: Final = (
    PolicyNodeType.CLAUSE,
    PolicyNodeType.DISCLOSURE,
    PolicyNodeType.INTERPRETATION,
    PolicyNodeType.EXAMPLE,
)

# An unrecognised language would silently fall back to the English text-search config.
_LANGUAGE_CHECK = "language IN ({})".format(", ".join(f"'{lang.value}'" for lang in PolicyLanguage))
_SEARCH_VECTOR_SQL = (
    "to_tsvector(CASE language WHEN 'es' THEN 'spanish'::regconfig "
    "ELSE 'english'::regconfig END, text)"
)
_EMPTY_ARRAY = text("'{}'")
_FALSE = text("false")
# Which bundle a node-level row belongs to, shared by every table that references policy_nodes.
_NODE_COLUMNS = ["policy_nodes.tenant_id", "policy_nodes.policy_bundle_id", "policy_nodes.id"]


class PolicySource(Base):
    # A source document, independent of any bundle: a re-imported file reuses its row (checksum).

    __tablename__ = "policy_sources"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "checksum_sha256"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"))
    title: Mapped[str] = mapped_column(String(300))
    source_ref: Mapped[str] = mapped_column(String(300))
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    # S3 key once NRC-3 uploads exist.
    storage_key: Mapped[str | None] = mapped_column(String(512))
    language: Mapped[PolicyLanguage] = mapped_column(enum_column(PolicyLanguage))
    is_synthetic: Mapped[bool]
    created_by: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PolicySourceClause(Base):
    __tablename__ = "policy_source_clauses"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "source_id", "clause_ref"),
        ForeignKeyConstraint(
            ["tenant_id", "source_id"], ["policy_sources.tenant_id", "policy_sources.id"]
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    source_id: Mapped[UUID]
    # For example "§3.2".
    clause_ref: Mapped[str] = mapped_column(String(100))
    section: Mapped[str | None] = mapped_column(String(200))
    page: Mapped[int | None]
    text: Mapped[str] = mapped_column(Text)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    language: Mapped[PolicyLanguage] = mapped_column(enum_column(PolicyLanguage))


class PolicyNode(Base):
    # The unit that is searched and cited. node_key is what the API shows, never the UUID.

    __tablename__ = "policy_nodes"
    __table_args__ = (
        # The target of every composite foreign key that points at a node.
        UniqueConstraint("tenant_id", "policy_bundle_id", "id"),
        UniqueConstraint("tenant_id", "policy_bundle_id", "node_key"),
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"], ["policy_bundles.tenant_id", "policy_bundles.id"]
        ),
        # A composite key with a NULL column is not enforced (MATCH SIMPLE): that is wanted here,
        # a node may have no source clause.
        ForeignKeyConstraint(
            ["tenant_id", "source_clause_id"],
            ["policy_source_clauses.tenant_id", "policy_source_clauses.id"],
        ),
        CheckConstraint(_LANGUAGE_CHECK, name="language"),
        # An inverted window would make a mandatory rule silently never apply.
        CheckConstraint(
            "effective_from IS NULL OR effective_to IS NULL OR effective_from <= effective_to",
            name="effective_window",
        ),
        Index("ix_policy_nodes_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    policy_bundle_id: Mapped[UUID]
    node_key: Mapped[str] = mapped_column(String(80))
    node_type: Mapped[PolicyNodeType] = mapped_column(enum_column(PolicyNodeType))
    text: Mapped[str] = mapped_column(Text)
    # Plain column rather than Language: the generated search_vector reads it.
    language: Mapped[PolicyLanguage] = mapped_column(enum_column(PolicyLanguage))
    source_clause_id: Mapped[UUID | None]
    # Unapproved proposals never take part in retrieval.
    approved: Mapped[bool] = mapped_column(server_default=_FALSE)
    # Applicability of RULE and EXCEPTION nodes; an empty array means no restriction.
    input_kinds: Mapped[list[str]] = mapped_column(ARRAY(String(64)), server_default=_EMPTY_ARRAY)
    channels: Mapped[list[str]] = mapped_column(ARRAY(String(64)), server_default=_EMPTY_ARRAY)
    languages: Mapped[list[str]] = mapped_column(ARRAY(String(64)), server_default=_EMPTY_ARRAY)
    audiences: Mapped[list[str]] = mapped_column(ARRAY(String(64)), server_default=_EMPTY_ARRAY)
    jurisdictions: Mapped[list[str]] = mapped_column(ARRAY(String(64)), server_default=_EMPTY_ARRAY)
    study_ids: Mapped[list[UUID]] = mapped_column(ARRAY(Uuid()), server_default=_EMPTY_ARRAY)
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # StudyFacts field that must be non-empty for an exception to apply (checked on write).
    required_study_fact: Mapped[str | None] = mapped_column(String(64))
    # Only search predicates read it, so ORM loads skip it.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR, Computed(_SEARCH_VECTOR_SQL, persisted=True), deferred=True
    )


class PolicyRuleRecord(Base):
    # Metadata of one RULE node. Applicability and approval live on the node, so rules and
    # exceptions share one predicate evaluator and one approval flag. That the node is a RULE node
    # is checked on write (NRC-4).

    __tablename__ = "policy_rules"
    __table_args__ = (
        UniqueConstraint("tenant_id", "policy_bundle_id", "node_id"),
        UniqueConstraint("tenant_id", "policy_bundle_id", "rule_id"),
        ForeignKeyConstraint(["tenant_id", "policy_bundle_id", "node_id"], _NODE_COLUMNS),
        CheckConstraint(f"rule_id ~ '{RULE_ID_PATTERN}'", name="rule_id_pattern"),
        CheckConstraint("rule_version >= 1", name="rule_version_positive"),
        # An empty phrase would match at every word boundary and fail every post.
        CheckConstraint("array_position(prohibited_phrases, '') IS NULL", name="no_empty_phrase"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    policy_bundle_id: Mapped[UUID]
    node_id: Mapped[UUID]
    rule_id: Mapped[str] = mapped_column(String(40))
    rule_version: Mapped[int]
    title: Mapped[str] = mapped_column(String(200))
    requirement: Mapped[str] = mapped_column(String(1000))
    gate: Mapped[Gate] = mapped_column(enum_column(Gate))
    severity: Mapped[Severity] = mapped_column(enum_column(Severity))
    mandatory: Mapped[bool]
    check_type: Mapped[CheckType] = mapped_column(enum_column(CheckType))
    # Required language, never paraphrased.
    exact_wording: Mapped[str | None] = mapped_column(Text)
    # Checked first, case-insensitive and whole-word; a hit fails the rule without a model call.
    prohibited_phrases: Mapped[list[str]] = mapped_column(
        ARRAY(String(200)), server_default=_EMPTY_ARRAY
    )


class PolicyEdge(Base):
    # Only approved edges are followed. Both endpoints are in the same bundle and tenant.

    __tablename__ = "policy_edges"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "policy_bundle_id",
            "edge_type",
            "from_node_id",
            "to_node_id",
            name="uq_policy_edges_endpoints",
        ),
        # Two foreign keys to the same table need explicit names.
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "from_node_id"],
            _NODE_COLUMNS,
            name="fk_policy_edges_from_node_policy_nodes",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id", "to_node_id"],
            _NODE_COLUMNS,
            name="fk_policy_edges_to_node_policy_nodes",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "provenance_clause_id"],
            ["policy_source_clauses.tenant_id", "policy_source_clauses.id"],
            name="fk_policy_edges_provenance_clause_policy_source_clauses",
        ),
        CheckConstraint("from_node_id <> to_node_id", name="no_self_edge"),
        # NRC-4 AC: every approved edge links to an approved clause or an interpretation note.
        CheckConstraint(
            "NOT approved OR provenance_clause_id IS NOT NULL"
            " OR (provenance_note IS NOT NULL AND btrim(provenance_note) <> '')",
            name="approved_has_provenance",
        ),
        Index("ix_policy_edges_from_node", "tenant_id", "policy_bundle_id", "from_node_id"),
        Index("ix_policy_edges_to_node", "tenant_id", "policy_bundle_id", "to_node_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    policy_bundle_id: Mapped[UUID]
    edge_type: Mapped[PolicyEdgeType] = mapped_column(enum_column(PolicyEdgeType))
    from_node_id: Mapped[UUID]
    to_node_id: Mapped[UUID]
    approved: Mapped[bool] = mapped_column(server_default=_FALSE)
    provenance_clause_id: Mapped[UUID | None]
    provenance_note: Mapped[str | None] = mapped_column(Text)


class PolicyNodeEmbedding(Base):
    # One vector per node and embedding model. No ANN index: a bundle has a few hundred nodes, so
    # exact cosine search filtered by bundle is cheap and never returns fewer than k rows.

    __tablename__ = "policy_node_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(["tenant_id", "policy_bundle_id", "node_id"], _NODE_COLUMNS),
    )

    tenant_id: Mapped[UUID] = mapped_column(primary_key=True)
    policy_bundle_id: Mapped[UUID] = mapped_column(primary_key=True)
    node_id: Mapped[UUID] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    # Hash of the exact text that was embedded.
    content_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PolicyIndexBuild(Base):
    # One build of a bundle's index per embedding model; retrieval needs the READY one.

    __tablename__ = "policy_index_builds"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "policy_bundle_id"], ["policy_bundles.tenant_id", "policy_bundles.id"]
        ),
        # A failed or superseded build stays as history; only one may be READY.
        Index(
            "uq_policy_index_builds_ready",
            "tenant_id",
            "policy_bundle_id",
            "model_id",
            unique=True,
            postgresql_where=text(f"state = '{IndexBuildState.READY.value}'"),
        ),
        # Retrieval compares these with the bundle's approved nodes before trusting the index.
        CheckConstraint(
            f"state <> '{IndexBuildState.READY.value}' OR (node_count IS NOT NULL"
            " AND content_checksum IS NOT NULL AND completed_at IS NOT NULL)",
            name="ready_is_complete",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID]
    policy_bundle_id: Mapped[UUID]
    model_id: Mapped[str] = mapped_column(String(100))
    dimensions: Mapped[int]
    # Known once the build finishes.
    node_count: Mapped[int | None]
    # Hash over the sorted node ids and content hashes.
    content_checksum: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[IndexBuildState] = mapped_column(enum_column(IndexBuildState))
    failure_code: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
