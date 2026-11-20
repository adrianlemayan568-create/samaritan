"""Deterministic construction of structured propositions from an EvidenceSet."""

from __future__ import annotations

import json

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import Entity, Event, Observation, Relationship
from samaritan.database import DatabaseIntegrityError
from samaritan.evaluation_contracts import EvidenceSet
from samaritan.proposition_construction_contracts import (
    PropositionConstructionDefinition,
    PropositionConstructionRegistry,
)
from samaritan.proposition_contracts import (
    Proposition,
    PropositionFieldProvenance,
    PropositionRoleValue,
    PropositionTemporalScope,
    deterministic_proposition_id,
)
from samaritan.proposition_semantics import (
    PropositionSemantics,
    PropositionSemanticsRegistry,
)
from samaritan.repository import Repository


class PropositionConstructionError(ValueError):
    """Raised when supplied evidence cannot unambiguously construct a proposition."""


class PropositionConstructor:
    """Resolve only registered typed bindings; never execute dynamic expressions."""

    def __init__(
        self,
        repository: Repository,
        semantics_registry: PropositionSemanticsRegistry,
        construction_registry: PropositionConstructionRegistry,
    ) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("repository must be a Repository")
        if not isinstance(semantics_registry, PropositionSemanticsRegistry):
            raise TypeError("semantics_registry must be a PropositionSemanticsRegistry")
        if not isinstance(construction_registry, PropositionConstructionRegistry):
            raise TypeError("construction_registry must be a PropositionConstructionRegistry")
        self.repository = repository
        self.semantics_registry = semantics_registry
        self.construction_registry = construction_registry

    def construct(
        self,
        evidence_set: EvidenceSet,
        construction_id: str,
        construction_version: int,
    ) -> Proposition:
        """Construct one proposition from the exact supplied evidence selection."""
        if not isinstance(evidence_set, EvidenceSet):
            raise TypeError("evidence_set must be an EvidenceSet")
        construction = self.construction_registry.get(construction_id, construction_version)
        semantics = self.semantics_registry.get(
            construction.proposition_type,
            construction.semantics_version,
        )
        if construction.proposition_type != semantics.proposition_type:
            raise PropositionConstructionError("construction and semantics proposition types differ")

        event_definitions = {
            item.semantic_role: item for item in construction.event_bindings
        }
        entity_definitions = {
            item.semantic_role: item for item in construction.entity_bindings
        }
        semantic_roles = {item.role_name: item for item in semantics.roles}
        if set(event_definitions) != {
            name for name, role in semantic_roles.items() if role.role_kind == "event" and role.required
        }:
            raise PropositionConstructionError(
                "construction event bindings must cover every required semantic event role exactly"
            )
        if set(entity_definitions) != {
            name for name, role in semantic_roles.items() if role.role_kind == "entity" and role.required
        }:
            raise PropositionConstructionError(
                "construction entity bindings must cover every required semantic entity role exactly"
            )
        for name, binding in event_definitions.items():
            if semantic_roles[name].event_type != binding.event_type:
                raise PropositionConstructionError(
                    f"event binding type conflicts with semantics for {name}"
                )
        for name in entity_definitions:
            if semantic_roles[name].entity_type is None:
                raise PropositionConstructionError(f"entity role {name} has no declared entity type")

        parameter_definitions = {item.parameter_name: item for item in semantics.parameters}
        parameter_bindings = (
            construction.event_attribute_bindings
            + construction.temporal_parameter_bindings
            + construction.relationship_parameter_bindings
        )
        bindings_by_parameter = {item.parameter_name: item for item in parameter_bindings}
        if set(bindings_by_parameter) - set(parameter_definitions):
            raise PropositionConstructionError("construction binds an undeclared semantic parameter")
        for parameter_name, binding in bindings_by_parameter.items():
            parameter_type = parameter_definitions[parameter_name].value_type
            if binding in construction.event_attribute_bindings:
                if parameter_type not in {"string", "integer", "boolean"}:
                    raise PropositionConstructionError("unsupported event attribute parameter type")
            elif binding in construction.temporal_parameter_bindings and parameter_type != "integer":
                raise PropositionConstructionError("temporal difference parameters must be integers")
            elif binding in construction.relationship_parameter_bindings and parameter_type != "string":
                raise PropositionConstructionError("relationship field parameters must be strings")
        for parameter in semantics.parameters:
            if parameter.required and parameter.default_value is None and parameter.parameter_name not in bindings_by_parameter:
                raise PropositionConstructionError(
                    f"required semantic parameter has no registered binding: {parameter.parameter_name}"
                )

        events, observations = self._resolve_evidence(evidence_set.evidence)
        selected_events: dict[str, Event] = {}
        for role_name, binding in event_definitions.items():
            matches = [event for event in events.values() if event.event_type == binding.event_type]
            if len(matches) != 1:
                raise PropositionConstructionError(
                    f"expected exactly one {binding.event_type} event for {role_name}; found {len(matches)}"
                )
            selected_events[role_name] = matches[0]

        role_values: dict[str, PropositionRoleValue] = {}
        provenance: dict[str, PropositionFieldProvenance] = {}
        for role_name, event in selected_events.items():
            role_values[role_name] = PropositionRoleValue(
                role_name=role_name,
                role_kind="event",
                value_id=event.event_id,
                occurred_at=event.occurred_at,
            )
            provenance[role_name] = PropositionFieldProvenance(
                field_name=role_name,
                event_ids=(event.event_id,),
                observation_ids=(event.observation_id,),
            )

        participant_entity_ids: set[str] = set()
        participant_matches: dict[str, list[str]] = {}
        for entity_role, binding in entity_definitions.items():
            resolved_ids: list[str] = []
            for participant_binding in binding.participant_bindings:
                event = selected_events[participant_binding.event_role]
                matches = [
                    participant.entity_id
                    for participant in event.participants
                    if participant.role == participant_binding.participant_role
                ]
                if len(matches) != 1:
                    raise PropositionConstructionError(
                        f"expected exactly one participant with role "
                        f"{participant_binding.participant_role!r} in "
                        f"{participant_binding.event_role}; found {len(matches)}"
                    )
                resolved_ids.append(matches[0])
                participant_entity_ids.add(matches[0])
            if len(set(resolved_ids)) != 1:
                raise PropositionConstructionError(
                    f"entity role {entity_role} resolves to different entities across source events"
                )
            participant_matches[entity_role] = resolved_ids

        entities = self.repository.get_entities_by_ids(tuple(sorted(participant_entity_ids)))
        if not isinstance(entities, tuple) or not all(isinstance(item, Entity) for item in entities):
            raise DatabaseIntegrityError("repository returned invalid entity records")
        entities_by_id = {item.entity_id: item for item in entities}
        for entity_role, resolved_ids in participant_matches.items():
            entity_id = resolved_ids[0]
            entity = entities_by_id.get(entity_id)
            if entity is None:
                raise DatabaseIntegrityError(f"referenced entity does not exist: {entity_id}")
            expected_type = semantic_roles[entity_role].entity_type
            if entity.entity_type != expected_type:
                raise PropositionConstructionError(
                    f"entity type mismatch for {entity_role}: expected {expected_type}, found {entity.entity_type}"
                )
            role_values[entity_role] = PropositionRoleValue(
                role_name=entity_role,
                role_kind="entity",
                value_id=entity.entity_id,
            )
            source_events = [
                selected_events[item.event_role]
                for item in entity_definitions[entity_role].participant_bindings
            ]
            provenance[entity_role] = PropositionFieldProvenance(
                field_name=entity_role,
                event_ids=tuple(dict.fromkeys(item.event_id for item in source_events)),
                observation_ids=tuple(dict.fromkeys(item.observation_id for item in source_events)),
            )

        ordered_roles = tuple(role_values[name] for name in semantic_roles if name in role_values)
        event_times = {name: event.occurred_at for name, event in selected_events.items()}
        if semantics.temporal_scope is not None:
            scope_definition = semantics.temporal_scope
            starts_at = event_times[scope_definition.start_role]
            ends_at = event_times[scope_definition.end_role]
            start_inclusive = scope_definition.start_inclusive
            end_inclusive = scope_definition.end_inclusive
        else:
            starts_at = min(event_times.values())
            ends_at = max(event_times.values())
            start_inclusive = end_inclusive = True
        temporal_scope = PropositionTemporalScope(
            starts_at=starts_at,
            ends_at=ends_at,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
        )

        for constraint in semantics.temporal_constraints:
            earlier = event_times[constraint.earlier_role]
            later = event_times[constraint.later_role]
            if constraint.max_interval_seconds is not None:
                interval = (later - earlier).total_seconds()
                if interval > constraint.max_interval_seconds:
                    raise PropositionConstructionError(
                        f"temporal interval exceeds {constraint.max_interval_seconds} seconds: "
                        f"{constraint.earlier_role} -> {constraint.later_role}"
                    )

        parameter_values = {
            item.parameter_name: item.default_value
            for item in semantics.parameters
            if item.default_value is not None
        }
        for binding in construction.event_attribute_bindings:
            event = selected_events[binding.event_role]
            attributes = json.loads(event.normalized_data_json)
            if binding.attribute_name not in attributes:
                raise PropositionConstructionError(
                    f"registered event attribute is missing: {binding.attribute_name}"
                )
            value = attributes[binding.attribute_name]
            definition = parameter_definitions[binding.parameter_name]
            valid = (
                isinstance(value, str) if definition.value_type == "string"
                else type(value) is int if definition.value_type == "integer"
                else type(value) is bool
            )
            if not valid:
                raise PropositionConstructionError(
                    f"event attribute type mismatch for parameter {binding.parameter_name}"
                )
            parameter_values[binding.parameter_name] = value
            provenance[binding.parameter_name] = PropositionFieldProvenance(
                field_name=binding.parameter_name,
                event_ids=(event.event_id,),
                observation_ids=(event.observation_id,),
            )

        for binding in construction.temporal_parameter_bindings:
            earlier = selected_events[binding.earlier_event_role]
            later = selected_events[binding.later_event_role]
            seconds = (later.occurred_at - earlier.occurred_at).total_seconds()
            if seconds < 0 or not seconds.is_integer():
                raise PropositionConstructionError(
                    "temporal difference must be a non-negative whole number of seconds"
                )
            parameter_values[binding.parameter_name] = int(seconds)
            provenance[binding.parameter_name] = PropositionFieldProvenance(
                field_name=binding.parameter_name,
                event_ids=(earlier.event_id, later.event_id),
                observation_ids=tuple(dict.fromkeys((earlier.observation_id, later.observation_id))),
            )

        if construction.relationship_parameter_bindings:
            relationships = self._load_relationships(observations)
            for binding in construction.relationship_parameter_bindings:
                source_id = role_values[binding.source_entity_role].value_id
                target_id = role_values[binding.target_entity_role].value_id
                matches = [
                    item for item in relationships
                    if item.relationship_type == binding.relationship_type
                    and item.source_entity_id == source_id
                    and item.target_entity_id == target_id
                ]
                if len(matches) != 1:
                    raise PropositionConstructionError(
                        f"expected exactly one relationship of type {binding.relationship_type}; found {len(matches)}"
                    )
                relationship = matches[0]
                if binding.relationship_field == "relationship_id":
                    value = relationship.relationship_id
                elif binding.relationship_field == "relationship_type":
                    value = relationship.relationship_type
                else:
                    value = relationship.observed_at.isoformat()
                parameter_values[binding.parameter_name] = value
                provenance[binding.parameter_name] = PropositionFieldProvenance(
                    field_name=binding.parameter_name,
                    relationship_ids=(relationship.relationship_id,),
                    observation_ids=(relationship.observation_id,),
                )

        required_parameters = [
            item.parameter_name for item in semantics.parameters
            if item.required and item.parameter_name not in parameter_values
        ]
        if required_parameters:
            raise PropositionConstructionError(
                f"required semantic parameters have no registered binding: {required_parameters}"
            )
        parameters_json = json.dumps(
            parameter_values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        role_tuple = tuple(ordered_roles)
        proposition_id = deterministic_proposition_id(
            semantics.proposition_type,
            semantics.semantics_version,
            role_tuple,
            temporal_scope,
            parameters_json,
        )
        proposition = Proposition(
            proposition_id=proposition_id,
            proposition_type=semantics.proposition_type,
            semantics_version=semantics.semantics_version,
            roles=role_tuple,
            temporal_scope=temporal_scope,
            field_provenance=tuple(provenance[name] for name in (
                tuple(semantic_roles) + tuple(item.parameter_name for item in semantics.parameters)
            ) if name in provenance),
            construction_id=construction.construction_id,
            construction_version=construction.construction_version,
            parameters_json=parameters_json,
        )
        proposition.validate_against(semantics)
        return proposition

    def _load_relationships(
        self,
        observations: dict[str, Observation],
    ) -> tuple[Relationship, ...]:
        """Load all relationships attached to the selected source observations."""
        relationships = {}
        for observation_id in observations:
            offset = 0
            while True:
                batch = self.repository.list_relationships(
                    observation_id=observation_id,
                    limit=500,
                    offset=offset,
                )
                if not isinstance(batch, tuple):
                    raise DatabaseIntegrityError("repository returned invalid relationship records")
                for relationship in batch:
                    if relationship.observation_id != observation_id:
                        raise DatabaseIntegrityError("relationship observation provenance mismatch")
                    relationships[relationship.relationship_id] = relationship
                if len(batch) < 500:
                    break
                offset += 500
        return tuple(relationships.values())

    def _resolve_evidence(
        self,
        references: tuple[EvidenceReference, ...],
    ) -> tuple[dict[str, Event], dict[str, Observation]]:
        events: dict[str, Event] = {}
        observations: dict[str, Observation] = {}
        for reference in references:
            if reference.event_id is not None:
                event = self.repository.get_event(reference.event_id)
                if event is None:
                    raise DatabaseIntegrityError(
                        f"referenced event does not exist: {reference.event_id}"
                    )
                if not isinstance(event, Event) or event.event_id != reference.event_id:
                    raise DatabaseIntegrityError("repository returned an invalid or mismatched event")
                events[event.event_id] = event
                self._load_observation(event.observation_id, observations)
            else:
                observation = self._load_observation(reference.observation_id, observations)
                offset = 0
                while True:
                    related_events = self.repository.list_events(
                        observation_id=observation.observation_id,
                        limit=500,
                        offset=offset,
                    )
                    if not isinstance(related_events, tuple) or not all(
                        isinstance(item, Event) for item in related_events
                    ):
                        raise DatabaseIntegrityError("repository returned invalid events for an observation")
                    for event in related_events:
                        if event.observation_id != observation.observation_id:
                            raise DatabaseIntegrityError("event observation provenance mismatch")
                        events[event.event_id] = event
                    if len(related_events) < 500:
                        break
                    offset += 500
        return events, observations

    def _load_observation(
        self,
        observation_id: str,
        cache: dict[str, Observation],
    ) -> Observation:
        observation = cache.get(observation_id)
        if observation is None:
            observation = self.repository.get_observation(observation_id)
            if observation is None:
                raise DatabaseIntegrityError(
                    f"referenced observation does not exist: {observation_id}"
                )
            if not isinstance(observation, Observation) or observation.observation_id != observation_id:
                raise DatabaseIntegrityError("repository returned an invalid or mismatched observation")
            cache[observation_id] = observation
        return observation
