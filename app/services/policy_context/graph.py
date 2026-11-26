# Bounded breadth-first walk over a bundle's approved edges: pure, so it is tested without a DB.

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from app.models.enums import PolicyEdgeType

# Edge followed backwards from a rule to its interpretations (applies_to points at the rule).
APPLIED_BY: Final = "applied_by"
# Edges the walk follows forwards. supersedes is resolved during rule selection, not walked.
_FORWARD: Final = frozenset(
    {PolicyEdgeType.SUPPORTED_BY, PolicyEdgeType.REQUIRES_DISCLOSURE, PolicyEdgeType.EXCEPT_WHEN}
)

Adjacency = Mapping[UUID, Sequence[tuple[str, UUID]]]


@dataclass(frozen=True, slots=True)
class Reached:
    node_id: UUID
    # Node keys and edge labels from the start node: (start, label, key, label, key, ...).
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Walk:
    reached: tuple[Reached, ...]
    # True when nodes beyond the hop limit were left unvisited.
    truncated: bool


def adjacency(
    edges: Sequence[tuple[PolicyEdgeType, UUID, UUID]], keys: Mapping[UUID, str]
) -> dict[UUID, list[tuple[str, UUID]]]:
    """Neighbours of each node, sorted by (label, node key) so every walk is deterministic.

    Edges whose endpoints aren't in `keys` (the bundle's approved nodes) are ignored, so an
    approved edge to an unapproved node is never followed.
    """
    neighbours: dict[UUID, list[tuple[str, UUID]]] = {}
    for edge_type, source, target in edges:
        if source not in keys or target not in keys:
            continue
        if edge_type in _FORWARD:
            neighbours.setdefault(source, []).append((edge_type.value, target))
        elif edge_type is PolicyEdgeType.APPLIES_TO:
            neighbours.setdefault(target, []).append((APPLIED_BY, source))
    for items in neighbours.values():
        items.sort(key=lambda item: (item[0], keys[item[1]]))
    return neighbours


def walk(start: UUID, graph: Adjacency, keys: Mapping[UUID, str], max_hops: int) -> Walk:
    """Every node reachable from `start` within `max_hops`, each once, in breadth-first order.

    The visited set stops cycles; the hop limit bounds the result. Reaching the limit with
    unvisited neighbours left is reported as truncated, never silently.
    """
    visited = {start}
    reached: list[Reached] = []
    frontier: deque[tuple[UUID, tuple[str, ...], int]] = deque([(start, (keys[start],), 0)])
    truncated = False
    while frontier:
        node, path, depth = frontier.popleft()
        for label, neighbour in graph.get(node, ()):
            if neighbour in visited:
                continue
            if depth >= max_hops:
                truncated = True
                continue
            visited.add(neighbour)
            step = (*path, label, keys[neighbour])
            reached.append(Reached(neighbour, step))
            frontier.append((neighbour, step, depth + 1))
    return Walk(tuple(reached), truncated)
