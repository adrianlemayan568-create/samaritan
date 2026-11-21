"""Shared deterministic resolution of proposition evidence selections.

The resolver is deliberately independent of Proposition objects. Both construction
and repository-backed validation use the same candidate-selection rules so a
forged proposition cannot choose one source from an ambiguous manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import Event, Observation, Relationship
from samaritan.database import DatabaseIntegrityError
from samaritan.proposition_construction_contracts import (
    PropositionConstructionDefinition,
    RelationshipParameterBinding,
)
from samaritan.proposition_contracts import PropositionEvidenceManifest
from samaritan.proposition_semantics import PropositionSemantics
from samaritan.repository import Repository


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


@dataclass(frozen=True, slots=True)
class ResolvedManifestSelection:
    """Exact event/observation records selected by one immutable manifest."""

    events: tuple[Event, ...]
    selected_observations: tuple[Observation, ...]
    provenance_observations: tuple[Observation, ...]


def resolve_manifest_selection(
    repository: Repository,
    manifest: PropositionEvidenceManifest,
) -> ResolvedManifestSelection:
    """Resolve event and observation references without widening event-only scope."""
    if type(repository) is not Repository:
        raise TypeError("repository must be a Repository")
    if type(manifest) is not PropositionEvidenceManifest:
        raise TypeError("manifest must be a PropositionEvidenceManifest")

    events: dict[str, Event] = {}
    observations: dict[str, Observation] = {}
    selected_observations: dict[str, Observation] = {}
    provenance_observations: dict[str, Observation] = {}

    def load_observation(observation_id: str) -> Observation:
        observation = observations.get(observation_id)
        if observation is None:
            observation = repository.get_observation(observation_id)
            if observation is None:
                raise DatabaseIntegrityError(
                    f"referenced observation does not exist: {observation_id}"
                )
            if type(observation) is not Observation or observation.observation_id != observation_id:
                raise DatabaseIntegrityError(
                    "repository returned an invalid or mismatched observation"
                )
            observations[observation_id] = observation
        return observation

    def load_event(event_id: str) -> Event:
        event = events.get(event_id)
        if event is None:
            event = repository.get_event(event_id)
            if event is None:
                raise DatabaseIntegrityError(
                    f"referenced event does not exist: {event_id}"
                )
            if type(event) is not Event or event.event_id != event_id:
                raise DatabaseIntegrityError(
                    "repository returned an invalid or mismatched event"
                )
            parent = load_observation(event.observation_id)
            if parent.observation_id != event.observation_id:
                raise DatabaseIntegrityError("event observation provenance mismatch")
            events[event_id] = event
            provenance_observations[parent.observation_id] = parent
        return event

    for reference in manifest.evidence:
        if type(reference) is not EvidenceReference:
            raise TypeError("manifest evidence must contain EvidenceReference objects")
        if reference.event_id is not None:
            load_event(reference.event_id)
            continue

        observation = load_observation(reference.observation_id)
        selected_observations[observation.observation_id] = observation
        provenance_observations[observation.observation_id] = observation
        offset = 0
        while True:
            batch = repository.list_events(
                observation_id=observation.observation_id,
                limit=500,
                offset=offset,
            )
            if not isinstance(batch, tuple) or not all(
                type(item) is Event for item in batch
            ):
                raise DatabaseIntegrityError(
                    "repository returned invalid events for a manifest observation"
                )
            for item in batch:
                if item.observation_id != observation.observation_id:
                    raise DatabaseIntegrityError(
                        "event observation provenance mismatch"
                    )
                canonical_event = load_event(item.event_id)
                if canonical_event.observation_id != observation.observation_id:
                    raise DatabaseIntegrityError(
                        "canonical event observation provenance mismatch"
                    )
            if len(batch) < 500:
                break
            offset += 500

    return ResolvedManifestSelection(
        events=tuple(events.values()),
        selected_observations=tuple(selected_observations.values()),
        provenance_observations=tuple(provenance_observations.values()),
    )


def resolve_event_roles(
    semantics: PropositionSemantics,
    construction: PropositionConstructionDefinition,
    selected_events: tuple[Event, ...],
) -> ResolvedPropositionSelection:
    """Resolve event roles using all registered participant-consistency constraints."""
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

    # A participant binding constrains which source event may fill its event role.
    # It is not merely a later check performed after event selection.
    participant_constraints: dict[str, list[tuple[str, str]]] = {}
    for entity_binding in construction.entity_bindings:
        definition = definitions.get(entity_binding.semantic_role)
        if definition is None or definition.role_kind != "entity":
            raise EvidenceSelectionError(
                f"entity binding references an undeclared entity role: {entity_binding.semantic_role}"
            )
        for participant_binding in entity_binding.participant_bindings:
            participant_constraints.setdefault(
                participant_binding.event_role, []
            ).append((entity_binding.semantic_role, participant_binding.participant_role))

    role_options: list[tuple[str, bool, tuple[tuple[Event, dict[str, str]], ...]]] = []
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

        candidates: list[tuple[Event, dict[str, str]]] = []
        for event in selected_events:
            if event.event_type != binding.event_type:
                continue
            resolved_entities: dict[str, str] = {}
            valid_candidate = True
            for entity_role, participant_role in participant_constraints.get(role_name, ()):
                participant_ids = tuple(
                    participant.entity_id
                    for participant in event.participants
                    if participant.role == participant_role
                )
                if len(participant_ids) != 1:
                    valid_candidate = False
                    break
                resolved_entities[entity_role] = participant_ids[0]
            if valid_candidate:
                candidates.append((event, resolved_entities))

        if not candidates and definition.required:
            raw_count = sum(
                event.event_type == binding.event_type for event in selected_events
            )
            raise EvidenceSelectionError(
                f"expected exactly one participant-consistent {binding.event_type} event "
                f"for {role_name}; found 0 valid candidates among {raw_count} type matches"
            )
        role_options.append((role_name, definition.required, tuple(candidates)))

    # Search for complete assignments, pruning as soon as a registered entity
    # role would resolve to conflicting participant IDs. Stop after two results:
    # one proves uniqueness; two are sufficient to prove ambiguity.
    solutions: list[tuple[tuple[str, Event], ...]] = []
    assignment: list[tuple[str, Event]] = []
    assigned_entities: dict[str, str] = {}
    used_event_ids: dict[str, str] = {}

    def search(index: int) -> None:
        if len(solutions) >= 2:
            return
        if index == len(role_options):
            selected_role_names = {name for name, _ in assignment}
            for entity_binding in construction.entity_bindings:
                entity_definition = definitions[entity_binding.semantic_role]
                dependency_roles = {
                    item.event_role for item in entity_binding.participant_bindings
                }
                if not dependency_roles.issubset(selected_role_names) and entity_definition.required:
                    return
                if dependency_roles.issubset(selected_role_names) and entity_binding.semantic_role not in assigned_entities:
                    return
            solutions.append(tuple(assignment))
            return

        role_name, required_role, candidates = role_options[index]
        # An optional event role is omitted only when no candidate satisfies
        # its own registered participant constraints.
        choices: tuple[tuple[Event, dict[str, str]] | None, ...]
        choices = candidates if candidates else (None,)
        for choice in choices:
            if choice is None:
                search(index + 1)
                continue
            event, candidate_entities = choice
            previous_role = used_event_ids.get(event.event_id)
            if previous_role is not None:
                continue
            conflicts = any(
                entity_role in assigned_entities
                and assigned_entities[entity_role] != entity_id
                for entity_role, entity_id in candidate_entities.items()
            )
            if conflicts:
                continue

            newly_assigned = [
                entity_role for entity_role in candidate_entities
                if entity_role not in assigned_entities
            ]
            for entity_role in newly_assigned:
                assigned_entities[entity_role] = candidate_entities[entity_role]
            used_event_ids[event.event_id] = role_name
            assignment.append((role_name, event))
            search(index + 1)
            assignment.pop()
            del used_event_ids[event.event_id]
            for entity_role in newly_assigned:
                del assigned_entities[entity_role]

    search(0)
    if not solutions:
        raise EvidenceSelectionError(
            "no complete event-role assignment satisfies all registered participant constraints"
        )
    if len(solutions) > 1:
        raise EvidenceSelectionError(
            "event-role selection is ambiguous: multiple complete assignments satisfy "
            "the registered event-type and participant constraints"
        )
    return ResolvedPropositionSelection(solutions[0])


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
