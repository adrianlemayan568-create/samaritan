"""Repository-backed verification of proposition source provenance."""

from __future__ import annotations

import json

from samaritan.contracts import Entity, Event, Observation, Relationship
from samaritan.database import DatabaseIntegrityError
from samaritan.proposition_construction_contracts import PropositionConstructionRegistry
from samaritan.proposition_contracts import Proposition
from samaritan.proposition_semantics import PropositionSemanticsRegistry
from samaritan.repository import Repository
from samaritan.proposition_evidence_selection import (
    resolve_manifest_selection,
    resolve_event_roles,
    resolve_entity_roles,
    resolve_parameter_values,
)


class PropositionProvenanceError(ValueError):
    """Raised when proposition provenance does not match authoritative source records."""


class PropositionProvenanceValidator:
    """Check semantic and source provenance claims against registered definitions.

    The validator also recomputes manifest-level role and relationship selection.
    The manifest is not yet authenticated by persistent EvidenceSet storage.
    """

    def __init__(
        self,
        repository: Repository,
        semantics_registry: PropositionSemanticsRegistry,
        construction_registry: PropositionConstructionRegistry,
    ) -> None:
        if type(repository) is not Repository:
            raise TypeError("repository must be a Repository")
        if type(semantics_registry) is not PropositionSemanticsRegistry:
            raise TypeError("semantics_registry must be a PropositionSemanticsRegistry")
        if type(construction_registry) is not PropositionConstructionRegistry:
            raise TypeError("construction_registry must be a PropositionConstructionRegistry")
        semantics_registry.freeze()
        construction_registry.freeze()
        self.repository = repository
        self.semantics_registry = semantics_registry
        self.construction_registry = construction_registry

    def validate(self, proposition: Proposition) -> None:
        """Raise when any claimed field provenance is not supported by stored records."""
        if type(proposition) is not Proposition:
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
        except ValueError as exc:
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
                if not type(event) is Event or event.event_id != event_id:
                    raise PropositionProvenanceError(
                        f"provenance references a missing or invalid event: {event_id}"
                    )
                events[event_id] = event
            return event

        def load_observation(observation_id: str) -> Observation:
            observation = observations.get(observation_id)
            if observation is None:
                observation = self.repository.get_observation(observation_id)
                if not type(observation) is Observation or observation.observation_id != observation_id:
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

        # Resolve the exact manifest through the shared repository-backed
        # selection path. Event-only references load their parent observations
        # for provenance but do not select observation-level child records.
        try:
            manifest_selection = resolve_manifest_selection(
                self.repository, proposition.evidence_manifest
            )
        except (DatabaseIntegrityError, ValueError) as exc:
            raise PropositionProvenanceError(str(exc)) from exc
        events.update({item.event_id: item for item in manifest_selection.events})
        observations.update({
            item.observation_id: item
            for item in manifest_selection.provenance_observations
        })
        manifest_selected_observation_ids = {
            item.observation_id for item in manifest_selection.selected_observations
        }
        manifest_provenance_observation_ids = {
            item.observation_id for item in manifest_selection.provenance_observations
        }
        manifest_event_ids = set(events)
        try:
            expected_event_selection = resolve_event_roles(
                semantics, construction, tuple(manifest_selection.events)
            )
            expected_entity_role_records = resolve_entity_roles(
                semantics, construction, expected_event_selection
            )
        except (TypeError, ValueError) as exc:
            raise PropositionProvenanceError(str(exc)) from exc
        expected_event_roles = dict(expected_event_selection.event_roles)
        expected_entity_roles = {
            item.role_name: item for item in expected_entity_role_records
        }
        for role_name, expected_event in expected_event_roles.items():
            actual_role = roles.get(role_name)
            if (
                actual_role is None
                or actual_role.value_id != expected_event.event_id
                or actual_role.occurred_at != expected_event.occurred_at
            ):
                raise PropositionProvenanceError(
                    f"event role presence or value does not match manifest selection: {role_name}"
                )
        for role_name, actual_role in roles.items():
            if actual_role.role_kind == "event" and role_name not in expected_event_roles:
                raise PropositionProvenanceError(
                    f"event role is not selected by the evidence manifest: {role_name}"
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

        # Require the proposition's entity-role set to match the shared
        # manifest-derived participant selection, including optional roles.
        for role_name in entity_bindings:
            expected_entity = expected_entity_roles.get(role_name)
            actual_entity = roles.get(role_name)
            if (
                (expected_entity is None and actual_entity is not None)
                or (
                    expected_entity is not None
                    and (
                        actual_entity is None
                        or actual_entity.value_id != expected_entity.entity_id
                    )
                )
            ):
                raise PropositionProvenanceError(
                    f"entity role presence or value does not match manifest selection: {role_name}"
                )

        entity_ids: set[str] = set()
        for role_name in entity_bindings:
            expected_entity = expected_entity_roles.get(role_name)
            role = roles.get(role_name)
            if expected_entity is None:
                continue
            if role is None:
                raise PropositionProvenanceError(
                    f"entity role is missing despite a manifest match: {role_name}"
                )
            source = provenance.get(role_name)
            if source is None:
                raise PropositionProvenanceError(
                    f"populated entity role has no provenance: {role_name}"
                )
            if (
                role.value_id != expected_entity.entity_id
                or source.event_ids != expected_entity.event_ids
                or source.observation_ids != expected_entity.observation_ids
                or source.relationship_ids
            ):
                raise PropositionProvenanceError(
                    f"entity role value or provenance does not match shared selection: {role_name}"
                )
            entity_ids.add(role.value_id)

        entities = self.repository.get_entities_by_ids(tuple(sorted(entity_ids)))
        if not isinstance(entities, tuple) or not all(type(item) is Entity for item in entities):
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

        # Resolve all parameter sources with the same pure selection logic
        # used by construction, then compare the full value/provenance projection.
        selected_relationships: list[Relationship] = []
        if construction.relationship_parameter_bindings:
            for observation_id in manifest_selected_observation_ids:
                offset = 0
                while True:
                    batch = self.repository.list_relationships(
                        observation_id=observation_id, limit=500, offset=offset
                    )
                    if not isinstance(batch, tuple) or not all(
                        type(item) is Relationship for item in batch
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

        try:
            expected_parameters = resolve_parameter_values(
                semantics,
                construction,
                expected_event_selection,
                expected_entity_role_records,
                tuple(selected_relationships),
            )
        except (TypeError, ValueError) as exc:
            raise PropositionProvenanceError(str(exc)) from exc
        expected_parameters_by_name = {
            item.parameter_name: item for item in expected_parameters
        }
        parameter_definitions = {
            item.parameter_name: item for item in semantics.parameters
        }
        for parameter_name, definition in parameter_definitions.items():
            expected = expected_parameters_by_name.get(parameter_name)
            source = provenance.get(parameter_name)
            if expected is not None:
                actual_present = parameter_name in parameters
                actual_value = parameters.get(parameter_name)
                if (
                    not actual_present
                    or type(actual_value) is not type(expected.value)
                    or actual_value != expected.value
                ):
                    raise PropositionProvenanceError(
                        f"parameter value does not match shared source selection: {parameter_name}"
                    )
                if source is None or (
                    source.event_ids != expected.event_ids
                    or source.observation_ids != expected.observation_ids
                    or source.relationship_ids != expected.relationship_ids
                ):
                    raise PropositionProvenanceError(
                        f"parameter provenance does not match shared source selection: {parameter_name}"
                    )
                continue

            if definition.default_value is not None:
                actual_value = parameters.get(parameter_name)
                if (
                    parameter_name not in parameters
                    or type(actual_value) is not type(definition.default_value)
                    or actual_value != definition.default_value
                    or source is not None
                ):
                    raise PropositionProvenanceError(
                        f"unresolved optional parameter must retain only its registered default: {parameter_name}"
                    )
            elif parameter_name in parameters or source is not None:
                raise PropositionProvenanceError(
                    f"parameter claims a source that is unavailable: {parameter_name}"
                )
