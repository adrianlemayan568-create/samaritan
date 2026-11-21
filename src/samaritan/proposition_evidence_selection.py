"""Shared deterministic resolution of proposition evidence selections.

The resolver is deliberately independent of Proposition objects. Both construction
and repository-backed validation use the same candidate-selection rules so a
forged proposition cannot choose one source from an ambiguous manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from samaritan.contracts import Event, Relationship
from samaritan.proposition_construction_contracts import (
    PropositionConstructionDefinition,
    RelationshipParameterBinding,
)
from samaritan.proposition_semantics import PropositionSemantics


class EvidenceSelectionError(ValueError):
    """Raised when a manifest cannot resolve registered bindings uniquely."""


@dataclass(frozen=True, slots=True)
class ResolvedPropositionSelection:
    """Immutable event-role selection derived only from registered definitions."""

    event_roles: tuple[tuple[str, Event], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_roles, tuple):
            raise TypeError("event_roles must be a tuple")
        names = tuple(name for name, _ in self.event_roles)
        if len(names) != len(set(names)):
            raise ValueError("resolved event role names must be unique")
        if not all(isinstance(event, Event) for _, event in self.event_roles):
            raise TypeError("resolved event roles must contain Event records")

    def event_for(self, role_name: str) -> Event | None:
        return next((event for name, event in self.event_roles if name == role_name), None)


def resolve_event_roles(
    semantics: PropositionSemantics,
    construction: PropositionConstructionDefinition,
    selected_events: tuple[Event, ...],
) -> ResolvedPropositionSelection:
    """Resolve every registered event role from the manifest's selected events."""
    if not isinstance(semantics, PropositionSemantics):
        raise TypeError("semantics must be PropositionSemantics")
    if not isinstance(construction, PropositionConstructionDefinition):
        raise TypeError("construction must be PropositionConstructionDefinition")
    if not isinstance(selected_events, tuple) or not all(
        isinstance(event, Event) for event in selected_events
    ):
        raise TypeError("selected_events must be a tuple of Event records")

    definitions = {role.role_name: role for role in semantics.roles}
    bindings = {binding.semantic_role: binding for binding in construction.event_bindings}
    required = {
        name for name, role in definitions.items()
        if role.role_kind == "event" and role.required
    }
    if not required.issubset(bindings):
        raise EvidenceSelectionError(
            "construction event bindings must cover every required semantic event role"
        )

    resolved: list[tuple[str, Event]] = []
    used_event_ids: dict[str, str] = {}
    for role_name, binding in bindings.items():
        definition = definitions.get(role_name)
        if definition is None or definition.role_kind != "event":
            raise EvidenceSelectionError(
                f"event binding references an undeclared event role: {role_name}"
            )
        if definition.event_type != binding.event_type:
            raise EvidenceSelectionError(
                f"event binding type conflicts with semantics for {role_name}"
            )
        matches = tuple(
            event for event in selected_events
            if event.event_type == binding.event_type
        )
        if len(matches) > 1:
            raise EvidenceSelectionError(
                f"expected at most one {binding.event_type} event for {role_name}; found {len(matches)}"
            )
        if not matches:
            if definition.required:
                raise EvidenceSelectionError(
                    f"expected exactly one {binding.event_type} event for {role_name}; found 0"
                )
            continue
        event = matches[0]
        previous_role = used_event_ids.get(event.event_id)
        if previous_role is not None:
            raise EvidenceSelectionError(
                f"one source event cannot satisfy multiple semantic roles: "
                f"{previous_role} and {role_name}"
            )
        used_event_ids[event.event_id] = role_name
        resolved.append((role_name, event))
    return ResolvedPropositionSelection(tuple(resolved))


def resolve_relationship(
    relationships: tuple[Relationship, ...],
    binding: RelationshipParameterBinding,
    source_entity_id: str,
    target_entity_id: str,
    *,
    required: bool,
) -> Relationship | None:
    """Resolve one relationship binding against all explicitly selected observations."""
    if not isinstance(relationships, tuple) or not all(
        isinstance(item, Relationship) for item in relationships
    ):
        raise TypeError("relationships must be a tuple of Relationship records")
    if not isinstance(binding, RelationshipParameterBinding):
        raise TypeError("binding must be RelationshipParameterBinding")
    matches = tuple(
        item for item in relationships
        if item.relationship_type == binding.relationship_type
        and item.source_entity_id == source_entity_id
        and item.target_entity_id == target_entity_id
    )
    if len(matches) > 1:
        raise EvidenceSelectionError(
            f"expected at most one relationship of type {binding.relationship_type}; found {len(matches)}"
        )
    if not matches:
        if required:
            raise EvidenceSelectionError(
                f"expected exactly one relationship of type {binding.relationship_type}; found 0"
            )
        return None
    return matches[0]
