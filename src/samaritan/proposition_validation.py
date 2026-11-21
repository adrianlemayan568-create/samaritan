"""Repository-backed verification of proposition source provenance."""

from __future__ import annotations

import json

from samaritan.contracts import Entity, Event, Observation, Relationship
from samaritan.proposition_construction_contracts import PropositionConstructionRegistry
from samaritan.proposition_contracts import Proposition
from samaritan.proposition_semantics import PropositionSemanticsRegistry
from samaritan.repository import Repository
from samaritan.proposition_evidence_selection import (
    EvidenceSelectionError,
    resolve_event_roles,
    resolve_relationship,
)


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
        try:
            proposition.validate_against(semantics)
        except (TypeError, ValueError) as exc:
            raise PropositionProvenanceError(str(exc)) from exc

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

        event_bindings = {item.semantic_role: item for item in construction.event_bindings}
        entity_bindings = {item.semantic_role: item for item in construction.entity_bindings}
        bound_role_names = set(event_bindings) | set(entity_bindings)
        semantic_roles = {item.role_name: item for item in semantics.roles}
        required_role_names = {
            name for name, definition in semantic_roles.items() if definition.required
        }
        if not set(roles).issubset(bound_role_names):
            raise PropositionProvenanceError(
                "proposition roles must match exact registered construction bindings; every populated role must have a registered binding"
            )
        if not required_role_names.issubset(roles):
            raise PropositionProvenanceError(
                "proposition is missing one or more required semantic roles"
            )

        # Resolve the immutable evidence-selection manifest independently of
        # field provenance. Observation references select the observation and
        # its child records; event references select one event. An event's
        # parent observation is loaded only to verify provenance and does not
        # authorize unrelated observation-level records such as relationships.
        manifest_selected_observation_ids: set[str] = set()
        manifest_provenance_observation_ids: set[str] = set()
        manifest_event_ids: set[str] = set()
        for reference in proposition.evidence_manifest.evidence:
            if reference.event_id is not None:
                event = load_event(reference.event_id)
                # The event record identifies its owning observation. Loading
                # that observation verifies the referenced repository record
                # exists; there is no second observation ID in this reference
                # to compare against.
                observation = load_observation(event.observation_id)
                manifest_event_ids.add(event.event_id)
                manifest_provenance_observation_ids.add(observation.observation_id)
            else:
                observation = load_observation(reference.observation_id)
                manifest_selected_observation_ids.add(observation.observation_id)
                manifest_provenance_observation_ids.add(observation.observation_id)
                offset = 0
                while True:
                    batch = self.repository.list_events(
                        observation_id=observation.observation_id,
                        limit=500,
                        offset=offset,
                    )
                    if not isinstance(batch, tuple) or not all(
                        isinstance(item, Event) for item in batch
                    ):
                        raise PropositionProvenanceError(
                            "repository returned invalid events for a manifest observation"
                        )
                    for event in batch:
                        if event.observation_id != observation.observation_id:
                            raise PropositionProvenanceError(
                                "manifest observation contains an event with mismatched provenance"
                            )
                        manifest_event_ids.add(event.event_id)
                        load_event(event.event_id)
                    if len(batch) < 500:
                        break
                    offset += 500

        # Re-run the same manifest-to-role selection used by construction.
        # This rejects omitted optional roles and stale propositions when a
        # manifest later contains a newly matching event.
        try:
            expected_event_selection = resolve_event_roles(
                semantics,
                construction,
                tuple(events[event_id] for event_id in manifest_event_ids),
            )
        except (TypeError, ValueError) as exc:
            raise PropositionProvenanceError(str(exc)) from exc
        expected_event_roles = dict(expected_event_selection.event_roles)
        actual_event_roles = {
            name: role for name, role in roles.items() if role.role_kind == "event"
        }
        if set(actual_event_roles) != set(expected_event_roles) or any(
            actual_event_roles[name].value_id != event.event_id
            or actual_event_roles[name].occurred_at != event.occurred_at
            for name, event in expected_event_roles.items()
        ):
            raise PropositionProvenanceError(
                "event roles do not match the unique role selection from the evidence manifest"
            )

        for item in proposition.field_provenance:
            for event_id in item.event_ids:
                event = load_event(event_id)
                if event.observation_id not in item.observation_ids:
                    raise PropositionProvenanceError(
                        f"event and observation provenance disagree for field: {item.field_name}"
                    )
                if event_id not in manifest_event_ids:
                    raise PropositionProvenanceError(
                        f"field provenance event is outside the evidence manifest: {item.field_name}"
                    )
            for observation_id in item.observation_ids:
                load_observation(observation_id)
                if observation_id not in manifest_provenance_observation_ids:
                    raise PropositionProvenanceError(
                        f"field provenance observation is outside the evidence manifest: {item.field_name}"
                    )

        for role_name, binding in event_bindings.items():
            definition = semantic_roles.get(role_name)
            if definition is None or definition.role_kind != "event" or definition.event_type != binding.event_type:
                raise PropositionProvenanceError(
                    f"registered event binding conflicts with semantic role: {role_name}"
                )
        for role_name in entity_bindings:
            definition = semantic_roles.get(role_name)
            if definition is None or definition.role_kind != "entity":
                raise PropositionProvenanceError(
                    f"registered entity binding conflicts with semantic role: {role_name}"
                )

        for role_name, binding in event_bindings.items():
            role = roles.get(role_name)
            if role is None:
                continue
            source = provenance.get(role_name)
            if source is None:
                raise PropositionProvenanceError(
                    f"populated event role has no provenance: {role_name}"
                )
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

        # Independently derive the fallback scope from source-verified required roles.
        if semantics.temporal_scope is None:
            required_event_times = [
                load_event(roles[name].value_id).occurred_at
                for name, definition in semantic_roles.items()
                if definition.role_kind == "event" and definition.required
            ]
            if not required_event_times:
                raise PropositionProvenanceError(
                    "implicit temporal scope requires at least one required event role"
                )
            expected_start = min(required_event_times)
            expected_end = max(required_event_times)
            if (
                proposition.temporal_scope.starts_at != expected_start
                or proposition.temporal_scope.ends_at != expected_end
                or not proposition.temporal_scope.start_inclusive
                or not proposition.temporal_scope.end_inclusive
            ):
                raise PropositionProvenanceError(
                    "fallback temporal scope does not match source-verified required event roles"
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
                        f"populated entity role depends on an absent event role: {participant_binding.event_role}"
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
            source = provenance.get(role_name)
            if source is None:
                raise PropositionProvenanceError(
                    f"populated entity role has no provenance: {role_name}"
                )
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
        for role_name in entity_bindings:
            role = roles.get(role_name)
            if role is None:
                continue
            entity = entities_by_id.get(role.value_id)
            if entity is None or entity.entity_type != semantic_roles[role_name].entity_type:
                raise PropositionProvenanceError(
                    f"entity role does not resolve to the declared entity type: {role_name}"
                )

        parameter_definitions = {item.parameter_name: item for item in semantics.parameters}

        def validate_unavailable_optional_parameter(parameter_name: str) -> None:
            definition = parameter_definitions[parameter_name]
            value_present = parameter_name in parameters
            source_present = parameter_name in provenance
            if definition.required:
                raise PropositionProvenanceError(
                    f"required parameter depends on an absent optional role: {parameter_name}"
                )
            if source_present or (
                value_present and (
                    type(parameters[parameter_name]) is not type(definition.default_value)
                    or parameters[parameter_name] != definition.default_value
                )
            ):
                raise PropositionProvenanceError(
                    f"parameter claims a source that is unavailable: {parameter_name}"
                )

        for binding in construction.event_attribute_bindings:
            event_role = roles.get(binding.event_role)
            if event_role is None:
                validate_unavailable_optional_parameter(binding.parameter_name)
                continue
            event = load_event(event_role.value_id)
            attributes = json.loads(event.normalized_data_json)
            if binding.attribute_name not in attributes:
                raise PropositionProvenanceError(
                    f"source event is missing the registered attribute for parameter: {binding.parameter_name}"
                )
            if (
                type(parameters.get(binding.parameter_name)) is not type(attributes[binding.attribute_name])
                or parameters.get(binding.parameter_name) != attributes[binding.attribute_name]
            ):
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
            earlier_role = roles.get(binding.earlier_event_role)
            later_role = roles.get(binding.later_event_role)
            if earlier_role is None or later_role is None:
                validate_unavailable_optional_parameter(binding.parameter_name)
                continue
            earlier = load_event(earlier_role.value_id)
            later = load_event(later_role.value_id)
            seconds = (later.occurred_at - earlier.occurred_at).total_seconds()
            source = provenance[binding.parameter_name]
            expected_observations = tuple(dict.fromkeys(
                (earlier.observation_id, later.observation_id)
            ))
            if (
                not seconds.is_integer()
                or seconds < 0
                or type(parameters.get(binding.parameter_name)) is not int
                or parameters.get(binding.parameter_name) != int(seconds)
                or source.event_ids != (earlier.event_id, later.event_id)
                or source.observation_ids != expected_observations
                or source.relationship_ids
            ):
                raise PropositionProvenanceError(
                    f"temporal parameter value or provenance is inconsistent: {binding.parameter_name}"
                )

        # Resolve every relationship binding against the complete set of
        # explicitly selected observations, not merely the relationship ID
        # already claimed by the proposition.
        selected_relationships: list[Relationship] = []
        for observation_id in manifest_selected_observation_ids:
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
                if any(item.observation_id != observation_id for item in batch):
                    raise PropositionProvenanceError(
                        "selected relationship has mismatched observation provenance"
                    )
                selected_relationships.extend(batch)
                if len(batch) < 500:
                    break
                offset += 500

        for binding in construction.relationship_parameter_bindings:
            source_role = roles.get(binding.source_entity_role)
            target_role = roles.get(binding.target_entity_role)
            if source_role is None or target_role is None:
                continue
            try:
                expected_relationship = resolve_relationship(
                    tuple(selected_relationships),
                    binding,
                    source_role.value_id,
                    target_role.value_id,
                    required=parameter_definitions[binding.parameter_name].required,
                )
            except (TypeError, ValueError) as exc:
                raise PropositionProvenanceError(str(exc)) from exc
            claimed = provenance.get(binding.parameter_name)
            if expected_relationship is None:
                if claimed is not None:
                    raise PropositionProvenanceError(
                        f"relationship provenance claims an unavailable optional binding: {binding.parameter_name}"
                    )
            elif claimed is None or claimed.relationship_ids != (
                expected_relationship.relationship_id,
            ):
                raise PropositionProvenanceError(
                    f"relationship binding is omitted or does not match the unique manifest selection: {binding.parameter_name}"
                )

        for binding in construction.relationship_parameter_bindings:
            source_role = roles.get(binding.source_entity_role)
            target_role = roles.get(binding.target_entity_role)
            if source_role is None or target_role is None:
                validate_unavailable_optional_parameter(binding.parameter_name)
                continue
            source_id = source_role.value_id
            target_id = target_role.value_id
            source = provenance.get(binding.parameter_name)
            if source is None:
                definition = parameter_definitions[binding.parameter_name]
                value_present = binding.parameter_name in parameters
                if definition.required or (
                    value_present and parameters[binding.parameter_name] != definition.default_value
                ):
                    raise PropositionProvenanceError(
                        f"bound relationship parameter has no provenance: {binding.parameter_name}"
                    )
                candidate_matches: list[Relationship] = []
                for observation_id in manifest_selected_observation_ids:
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
                        candidate_matches.extend(
                            item for item in batch
                            if item.source_entity_id == source_id
                            and item.target_entity_id == target_id
                            and item.relationship_type == binding.relationship_type
                        )
                        if len(batch) < 500:
                            break
                        offset += 500
                if candidate_matches:
                    raise PropositionProvenanceError(
                        f"optional relationship parameter omitted despite matching source: {binding.parameter_name}"
                    )
                continue
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
                or relationship.observation_id not in manifest_selected_observation_ids
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
            if (
                type(parameters.get(binding.parameter_name)) is not type(expected_value)
                or parameters.get(binding.parameter_name) != expected_value
            ):
                raise PropositionProvenanceError(
                    f"relationship parameter value does not match its source: {binding.parameter_name}"
                )
