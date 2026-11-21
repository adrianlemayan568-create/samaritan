"""Shared deterministic resolution of proposition evidence selections.

The resolver is deliberately independent of Proposition objects. Both construction
and repository-backed validation use the same candidate-selection rules so a
forged proposition cannot choose one source from an ambiguous manifest.
"""

from __future__ import annotations

import json
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
        if not all(type(event) is Event for _, event in self.event_roles):
            raise TypeError("resolved event roles must contain Event records")

    def event_for(self, role_name: str) -> Event | None:
        return next((event for name, event in self.event_roles if name == role_name), None)


def resolve_event_roles(
    semantics: PropositionSemantics,
    construction: PropositionConstructionDefinition,
    selected_events: tuple[Event, ...],
) -> ResolvedPropositionSelection:
    """Resolve every registered event role from the manifest's selected events."""
    if not type(semantics) is PropositionSemantics:
        raise TypeError("semantics must be PropositionSemantics")
    if not type(construction) is PropositionConstructionDefinition:
        raise TypeError("construction must be PropositionConstructionDefinition")
    if not isinstance(selected_events, tuple) or not all(
        type(event) is Event for event in selected_events
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


@dataclass(frozen=True, slots=True)
class ResolvedEntityRole:
    """One entity role and the exact participant events that resolve it."""

    role_name: str
    entity_id: str
    event_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]


def resolve_entity_roles(
    semantics: PropositionSemantics,
    construction: PropositionConstructionDefinition,
    event_selection: ResolvedPropositionSelection,
) -> tuple[ResolvedEntityRole, ...]:
    """Resolve entity roles from the selected event roles and registered participants."""
    if not type(event_selection) is ResolvedPropositionSelection:
        raise TypeError("event_selection must be ResolvedPropositionSelection")
    semantic_roles = {item.role_name: item for item in semantics.roles}
    selected_events = dict(event_selection.event_roles)
    resolved: list[ResolvedEntityRole] = []
    for binding in construction.entity_bindings:
        definition = semantic_roles.get(binding.semantic_role)
        if definition is None or definition.role_kind != "entity":
            raise EvidenceSelectionError(
                f"entity binding references an undeclared entity role: {binding.semantic_role}"
            )
        entity_ids: list[str] = []
        source_events: list[Event] = []
        missing_dependency = False
        for participant_binding in binding.participant_bindings:
            event = selected_events.get(participant_binding.event_role)
            if event is None:
                missing_dependency = True
                break
            matches = [
                participant.entity_id
                for participant in event.participants
                if participant.role == participant_binding.participant_role
            ]
            if len(matches) != 1:
                raise EvidenceSelectionError(
                    f"expected exactly one participant with role "
                    f"{participant_binding.participant_role!r} in "
                    f"{participant_binding.event_role}; found {len(matches)}"
                )
            entity_ids.append(matches[0])
            source_events.append(event)
        if missing_dependency:
            if definition.required:
                raise EvidenceSelectionError(
                    f"required entity role {binding.semantic_role} depends on an absent optional event role"
                )
            continue
        if len(set(entity_ids)) != 1:
            raise EvidenceSelectionError(
                f"entity role {binding.semantic_role} resolves to different entities across source events"
            )
        resolved.append(
            ResolvedEntityRole(
                role_name=binding.semantic_role,
                entity_id=entity_ids[0],
                event_ids=tuple(dict.fromkeys(item.event_id for item in source_events)),
                observation_ids=tuple(dict.fromkeys(item.observation_id for item in source_events)),
            )
        )
    return tuple(resolved)


@dataclass(frozen=True, slots=True)
class ResolvedParameterValue:
    """A typed parameter value and the exact source records that produced it."""

    parameter_name: str
    value: str | int | bool
    event_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    relationship_ids: tuple[str, ...] = ()


def resolve_parameter_values(
    semantics: PropositionSemantics,
    construction: PropositionConstructionDefinition,
    event_selection: ResolvedPropositionSelection,
    entity_roles: tuple[ResolvedEntityRole, ...],
    relationships: tuple[Relationship, ...],
) -> tuple[ResolvedParameterValue, ...]:
    """Resolve every registered parameter binding from the shared source selection."""
    if type(semantics) is not PropositionSemantics:
        raise TypeError("semantics must be PropositionSemantics")
    if type(construction) is not PropositionConstructionDefinition:
        raise TypeError("construction must be PropositionConstructionDefinition")
    if type(event_selection) is not ResolvedPropositionSelection:
        raise TypeError("event_selection must be ResolvedPropositionSelection")
    if not isinstance(entity_roles, tuple) or not all(
        type(item) is ResolvedEntityRole for item in entity_roles
    ):
        raise TypeError("entity_roles must be a tuple of ResolvedEntityRole objects")
    if not isinstance(relationships, tuple) or not all(
        type(item) is Relationship for item in relationships
    ):
        raise TypeError("relationships must be a tuple of Relationship records")

    parameter_definitions = {
        item.parameter_name: item for item in semantics.parameters
    }
    events_by_role = dict(event_selection.event_roles)
    entities_by_role = {item.role_name: item.entity_id for item in entity_roles}
    resolved: list[ResolvedParameterValue] = []

    for binding in construction.event_attribute_bindings:
        event = events_by_role.get(binding.event_role)
        definition = parameter_definitions.get(binding.parameter_name)
        if definition is None:
            raise EvidenceSelectionError(
                f"construction binds an undeclared semantic parameter: {binding.parameter_name}"
            )
        if event is None:
            if definition.required:
                raise EvidenceSelectionError(
                    f"required parameter {binding.parameter_name} depends on an absent optional event role"
                )
            continue
        attributes = json.loads(event.normalized_data_json)
        if binding.attribute_name not in attributes:
            raise EvidenceSelectionError(
                f"registered event attribute is missing: {binding.attribute_name}"
            )
        value = attributes[binding.attribute_name]
        valid = (
            isinstance(value, str) if definition.value_type == "string"
            else type(value) is int if definition.value_type == "integer"
            else type(value) is bool
        )
        if not valid:
            raise EvidenceSelectionError(
                f"event attribute type mismatch for parameter {binding.parameter_name}"
            )
        resolved.append(
            ResolvedParameterValue(
                parameter_name=binding.parameter_name,
                value=value,
                event_ids=(event.event_id,),
                observation_ids=(event.observation_id,),
            )
        )

    for binding in construction.temporal_parameter_bindings:
        earlier = events_by_role.get(binding.earlier_event_role)
        later = events_by_role.get(binding.later_event_role)
        definition = parameter_definitions.get(binding.parameter_name)
        if definition is None:
            raise EvidenceSelectionError(
                f"construction binds an undeclared semantic parameter: {binding.parameter_name}"
            )
        if earlier is None or later is None:
            if definition.required:
                raise EvidenceSelectionError(
                    f"required parameter {binding.parameter_name} depends on an absent optional event role"
                )
            continue
        seconds = (later.occurred_at - earlier.occurred_at).total_seconds()
        if seconds < 0 or not seconds.is_integer():
            raise EvidenceSelectionError(
                "temporal difference must be a non-negative whole number of seconds"
            )
        resolved.append(
            ResolvedParameterValue(
                parameter_name=binding.parameter_name,
                value=int(seconds),
                event_ids=(earlier.event_id, later.event_id),
                observation_ids=tuple(dict.fromkeys(
                    (earlier.observation_id, later.observation_id)
                )),
            )
        )

    for binding in construction.relationship_parameter_bindings:
        source_id = entities_by_role.get(binding.source_entity_role)
        target_id = entities_by_role.get(binding.target_entity_role)
        definition = parameter_definitions.get(binding.parameter_name)
        if definition is None:
            raise EvidenceSelectionError(
                f"construction binds an undeclared semantic parameter: {binding.parameter_name}"
            )
        if source_id is None or target_id is None:
            if definition.required:
                raise EvidenceSelectionError(
                    f"required parameter {binding.parameter_name} depends on an absent optional entity role"
                )
            continue
        relationship = resolve_relationship(
            relationships,
            binding,
            source_id,
            target_id,
            required=definition.required,
        )
        if relationship is None:
            continue
        if binding.relationship_field == "relationship_id":
            value = relationship.relationship_id
        elif binding.relationship_field == "relationship_type":
            value = relationship.relationship_type
        else:
            value = relationship.observed_at.isoformat()
        resolved.append(
            ResolvedParameterValue(
                parameter_name=binding.parameter_name,
                value=value,
                observation_ids=(relationship.observation_id,),
                relationship_ids=(relationship.relationship_id,),
            )
        )
    return tuple(resolved)


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
        type(item) is Relationship for item in relationships
    ):
        raise TypeError("relationships must be a tuple of Relationship records")
    if not type(binding) is RelationshipParameterBinding:
        raise TypeError("binding must be RelationshipParameterBinding")
    matches = tuple(
        item for item in relationships
        if item.relationship_type == binding.relationship_type
        and item.source_entity_id == source_entity_id
        and item.target_entity_id == target_entity_id
    )
    if len(matches) > 1:
        raise EvidenceSelectionError(
            f"expected exactly one relationship of type {binding.relationship_type}; found {len(matches)}"
        )
    if not matches:
        if required:
            raise EvidenceSelectionError(
                f"expected exactly one relationship of type {binding.relationship_type}; found 0"
            )
        return None
    return matches[0]
