"""Repository-backed verification of proposition source provenance."""

from __future__ import annotations

import json

from samaritan.contracts import Entity, Event, Observation, Relationship
from samaritan.proposition_construction_contracts import PropositionConstructionRegistry
from samaritan.proposition_contracts import Proposition
from samaritan.proposition_semantics import PropositionSemanticsRegistry
from samaritan.repository import Repository


class PropositionProvenanceError(ValueError):
    """Raised when proposition provenance does not match authoritative source records."""


class PropositionProvenanceValidator:
    """Verify a proposition against its exact registered construction and source records."""

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

    def validate(self, proposition: Proposition) -> None:
        """Raise when any claimed field provenance is not supported by stored records."""
        if not isinstance(proposition, Proposition):
            raise TypeError("proposition must be a Proposition")

        construction = self.construction_registry.get(
            proposition.construction_id, proposition.construction_version
        )
        semantics = self.semantics_registry.get(
            proposition.proposition_type, proposition.semantics_version
        )
        if (
            construction.proposition_type != proposition.proposition_type
            or construction.semantics_version != proposition.semantics_version
        ):
            raise PropositionProvenanceError(
                "recorded construction does not target the proposition's exact semantics"
            )
        proposition.validate_against(semantics)

        roles = {item.role_name: item for item in proposition.roles}
        provenance = {item.field_name: item for item in proposition.field_provenance}
        parameters = json.loads(proposition.parameters_json)
        bound_parameter_names = {
            item.parameter_name
            for item in (
                construction.event_attribute_bindings
                + construction.temporal_parameter_bindings
                + construction.relationship_parameter_bindings
            )
        }
        for definition in semantics.parameters:
            if definition.parameter_name not in parameters:
                continue
            value = parameters[definition.parameter_name]
            if (
                definition.parameter_name not in bound_parameter_names
                and (
                    value != definition.default_value
                    or definition.parameter_name in provenance
                )
            ):
                raise PropositionProvenanceError(
                    f"parameter has no registered construction binding: {definition.parameter_name}"
                )
        events: dict[str, Event] = {}
        observations: dict[str, Observation] = {}

        def load_event(event_id: str) -> Event:
            event = events.get(event_id)
            if event is None:
                event = self.repository.get_event(event_id)
                if not isinstance(event, Event) or event.event_id != event_id:
                    raise PropositionProvenanceError(
                        f"provenance references a missing or invalid event: {event_id}"
                    )
                events[event_id] = event
            return event

        def load_observation(observation_id: str) -> Observation:
            observation = observations.get(observation_id)
            if observation is None:
                observation = self.repository.get_observation(observation_id)
                if not isinstance(observation, Observation) or observation.observation_id != observation_id:
                    raise PropositionProvenanceError(
                        f"provenance references a missing or invalid observation: {observation_id}"
                    )
                observations[observation_id] = observation
            return observation

        for item in proposition.field_provenance:
            for event_id in item.event_ids:
                event = load_event(event_id)
                if event.observation_id not in item.observation_ids:
                    raise PropositionProvenanceError(
                        f"event and observation provenance disagree for field: {item.field_name}"
                    )
            for observation_id in item.observation_ids:
                load_observation(observation_id)

        event_bindings = {item.semantic_role: item for item in construction.event_bindings}
        entity_bindings = {item.semantic_role: item for item in construction.entity_bindings}

        for role_name, binding in event_bindings.items():
            role = roles.get(role_name)
            if role is None:
                continue
            source = provenance[role_name]
            event = load_event(role.value_id)
            if (
                event.event_type != binding.event_type
                or event.occurred_at != role.occurred_at
                or source.event_ids != (event.event_id,)
                or source.observation_ids != (event.observation_id,)
                or source.relationship_ids
            ):
                raise PropositionProvenanceError(
                    f"event role value or provenance does not match its source event: {role_name}"
                )

        entity_ids: set[str] = set()
        for role_name, binding in entity_bindings.items():
            role = roles.get(role_name)
            if role is None:
                continue
            expected_event_ids: list[str] = []
            expected_observation_ids: list[str] = []
            for participant_binding in binding.participant_bindings:
                event_role = roles.get(participant_binding.event_role)
                if event_role is None:
                    raise PropositionProvenanceError(
                        f"entity binding references a missing proposition event role: {participant_binding.event_role}"
                    )
                event = load_event(event_role.value_id)
                matches = [
                    participant.entity_id
                    for participant in event.participants
                    if participant.role == participant_binding.participant_role
                ]
                if len(matches) != 1 or matches[0] != role.value_id:
                    raise PropositionProvenanceError(
                        f"entity role does not match its declared source participant: {role_name}"
                    )
                expected_event_ids.append(event.event_id)
                expected_observation_ids.append(event.observation_id)
            expected_events = tuple(dict.fromkeys(expected_event_ids))
            expected_observations = tuple(dict.fromkeys(expected_observation_ids))
            source = provenance[role_name]
            if (
                source.event_ids != expected_events
                or source.observation_ids != expected_observations
                or source.relationship_ids
            ):
                raise PropositionProvenanceError(
                    f"entity role provenance does not match its construction bindings: {role_name}"
                )
            entity_ids.add(role.value_id)

        entities = self.repository.get_entities_by_ids(tuple(sorted(entity_ids)))
        if not isinstance(entities, tuple) or not all(isinstance(item, Entity) for item in entities):
            raise PropositionProvenanceError("repository returned invalid entity records")
        entities_by_id = {item.entity_id: item for item in entities}
        semantic_roles = {item.role_name: item for item in semantics.roles}
        for role_name in entity_bindings:
            role = roles.get(role_name)
            if role is None:
                continue
            entity = entities_by_id.get(role.value_id)
            if entity is None or entity.entity_type != semantic_roles[role_name].entity_type:
                raise PropositionProvenanceError(
                    f"entity role does not resolve to the declared entity type: {role_name}"
                )

        for binding in construction.event_attribute_bindings:
            event_role = roles[binding.event_role]
            event = load_event(event_role.value_id)
            attributes = json.loads(event.normalized_data_json)
            if binding.attribute_name not in attributes:
                raise PropositionProvenanceError(
                    f"source event is missing the registered attribute for parameter: {binding.parameter_name}"
                )
            if parameters.get(binding.parameter_name) != attributes[binding.attribute_name]:
                raise PropositionProvenanceError(
                    f"parameter value does not match its registered event attribute: {binding.parameter_name}"
                )
            source = provenance[binding.parameter_name]
            if (
                source.event_ids != (event.event_id,)
                or source.observation_ids != (event.observation_id,)
                or source.relationship_ids
            ):
                raise PropositionProvenanceError(
                    f"parameter provenance does not match its event attribute source: {binding.parameter_name}"
                )

        for binding in construction.temporal_parameter_bindings:
            earlier = load_event(roles[binding.earlier_event_role].value_id)
            later = load_event(roles[binding.later_event_role].value_id)
            seconds = (later.occurred_at - earlier.occurred_at).total_seconds()
            source = provenance[binding.parameter_name]
            expected_observations = tuple(dict.fromkeys(
                (earlier.observation_id, later.observation_id)
            ))
            if (
                not seconds.is_integer()
                or seconds < 0
                or parameters.get(binding.parameter_name) != int(seconds)
                or source.event_ids != (earlier.event_id, later.event_id)
                or source.observation_ids != expected_observations
                or source.relationship_ids
            ):
                raise PropositionProvenanceError(
                    f"temporal parameter value or provenance is inconsistent: {binding.parameter_name}"
                )

        for binding in construction.relationship_parameter_bindings:
            source_id = roles[binding.source_entity_role].value_id
            target_id = roles[binding.target_entity_role].value_id
            source = provenance[binding.parameter_name]
            matches: list[Relationship] = []
            for observation_id in source.observation_ids:
                offset = 0
                while True:
                    batch = self.repository.list_relationships(
                        observation_id=observation_id, limit=500, offset=offset
                    )
                    if not isinstance(batch, tuple) or not all(
                        isinstance(item, Relationship) for item in batch
                    ):
                        raise PropositionProvenanceError(
                            "repository returned invalid relationship records"
                        )
                    matches.extend(
                        item for item in batch
                        if item.relationship_id in source.relationship_ids
                    )
                    if len(batch) < 500:
                        break
                    offset += 500
            if source.event_ids or len(source.relationship_ids) != 1 or len(matches) != 1:
                raise PropositionProvenanceError(
                    f"relationship provenance must identify exactly one stored relationship: {binding.parameter_name}"
                )
            relationship = matches[0]
            if (
                relationship.source_entity_id != source_id
                or relationship.target_entity_id != target_id
                or relationship.relationship_type != binding.relationship_type
                or source.observation_ids != (relationship.observation_id,)
            ):
                raise PropositionProvenanceError(
                    f"relationship provenance does not match its registered binding: {binding.parameter_name}"
                )
            if binding.relationship_field == "relationship_id":
                expected_value = relationship.relationship_id
            elif binding.relationship_field == "relationship_type":
                expected_value = relationship.relationship_type
            else:
                expected_value = relationship.observed_at.isoformat()
            if parameters.get(binding.parameter_name) != expected_value:
                raise PropositionProvenanceError(
                    f"relationship parameter value does not match its source: {binding.parameter_name}"
                )
