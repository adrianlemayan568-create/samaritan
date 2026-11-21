"""Immutable contracts for structured, provenance-preserving propositions."""

from __future__ import annotations

import json
import uuid

from samaritan.analysis_contracts import EvidenceReference
from dataclasses import dataclass
from datetime import datetime

from samaritan.contracts import SCHEMA_VERSION, normalize_timestamp, validate_id, validate_json_object, validate_text
from samaritan.proposition_semantics import PropositionSemantics
from samaritan.proposition_semantics_fingerprint import proposition_semantics_fingerprint


_PROPOSITION_NAMESPACE = uuid.UUID("a6c3d5f2-4c4b-5a87-9c23-0e4c5f6b7a81")
LEGACY_PROPOSITION_IDENTITY_ALGORITHM = "uuid5-v1"
CURRENT_PROPOSITION_IDENTITY_ALGORITHM = "uuid5-v2"
SUPPORTED_PROPOSITION_IDENTITY_ALGORITHMS = frozenset({
    LEGACY_PROPOSITION_IDENTITY_ALGORITHM,
    CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
})
ROLE_VALUE_KINDS = frozenset({"entity", "event"})


@dataclass(frozen=True, slots=True)
class PropositionRoleValue:
    """Bind one semantic role to one concrete entity or event."""

    role_name: str
    role_kind: str
    value_id: str
    occurred_at: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_name", validate_text(self.role_name, "role_name"))
        object.__setattr__(self, "role_kind", validate_text(self.role_kind, "role_kind"))
        if self.role_kind not in ROLE_VALUE_KINDS:
            raise ValueError("unsupported proposition role value kind")
        object.__setattr__(self, "value_id", validate_id(self.value_id, "value_id"))
        if self.role_kind == "entity" and self.occurred_at is not None:
            raise ValueError("entity role values cannot have occurred_at")
        if self.role_kind == "event":
            if self.occurred_at is None:
                raise ValueError("event role values require occurred_at")
            object.__setattr__(self, "occurred_at", normalize_timestamp(self.occurred_at, "occurred_at"))
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposition role value schema version")


@dataclass(frozen=True, slots=True)
class PropositionFieldProvenance:
    """Identify the exact source records used to construct one semantic field."""

    field_name: str
    event_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    relationship_ids: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_name", validate_text(self.field_name, "field_name"))
        for name in ("event_ids", "observation_ids", "relationship_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            normalized = tuple(validate_id(item, name) for item in values)
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must contain unique IDs")
            object.__setattr__(self, name, normalized)
        if not (self.event_ids or self.observation_ids or self.relationship_ids):
            raise ValueError("field provenance must reference at least one source record")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposition field provenance schema version")


@dataclass(frozen=True, slots=True)
class PropositionTemporalScope:
    """Concrete inclusive or exclusive UTC interval for a proposition."""

    starts_at: datetime
    ends_at: datetime
    start_inclusive: bool = True
    end_inclusive: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "starts_at", normalize_timestamp(self.starts_at, "starts_at"))
        object.__setattr__(self, "ends_at", normalize_timestamp(self.ends_at, "ends_at"))
        if self.ends_at < self.starts_at:
            raise ValueError("temporal scope end must not precede its start")
        if type(self.start_inclusive) is not bool or type(self.end_inclusive) is not bool:
            raise TypeError("temporal scope inclusivity flags must be booleans")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposition temporal scope schema version")


@dataclass(frozen=True, slots=True)
class PropositionEvidenceManifest:
    """Preserve the exact immutable evidence selection used for construction."""

    evidence_set_id: str
    evidence: tuple[EvidenceReference, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_set_id",
            validate_id(self.evidence_set_id, "evidence_set_id"),
        )
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence manifest evidence must be a tuple")
        if not all(isinstance(item, EvidenceReference) for item in self.evidence):
            raise TypeError(
                "evidence manifest must contain EvidenceReference objects"
            )
        keys = [(item.observation_id, item.event_id) for item in self.evidence]
        if len(keys) != len(set(keys)):
            raise ValueError("evidence manifest contains duplicate references")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposition evidence manifest schema version")


@dataclass(frozen=True, slots=True)
class Proposition:
    """An atomic proposition with exact semantics, role values, scope and provenance."""

    proposition_id: str
    proposition_type: str
    semantics_version: int
    roles: tuple[PropositionRoleValue, ...]
    temporal_scope: PropositionTemporalScope
    field_provenance: tuple[PropositionFieldProvenance, ...]
    construction_id: str
    construction_version: int
    semantics_fingerprint: str
    identity_algorithm_version: str
    evidence_manifest: PropositionEvidenceManifest
    parameters_json: str = "{}"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposition_id", validate_id(self.proposition_id, "proposition_id"))
        object.__setattr__(self, "proposition_type", validate_text(self.proposition_type, "proposition_type"))
        if type(self.semantics_version) is not int or self.semantics_version < 1:
            raise ValueError("semantics_version must be a positive integer")
        if not isinstance(self.roles, tuple) or not self.roles:
            raise TypeError("roles must be a non-empty tuple")
        if not all(isinstance(item, PropositionRoleValue) for item in self.roles):
            raise TypeError("roles must contain PropositionRoleValue objects")
        names = tuple(item.role_name for item in self.roles)
        if len(names) != len(set(names)):
            raise ValueError("proposition role names must be unique")
        if not isinstance(self.evidence_manifest, PropositionEvidenceManifest):
            raise TypeError("evidence_manifest must be a PropositionEvidenceManifest")
        if not isinstance(self.temporal_scope, PropositionTemporalScope):
            raise TypeError("temporal_scope must be a PropositionTemporalScope")
        if not isinstance(self.field_provenance, tuple) or not self.field_provenance:
            raise TypeError("field_provenance must be a non-empty tuple")
        if not all(isinstance(item, PropositionFieldProvenance) for item in self.field_provenance):
            raise TypeError("field_provenance must contain PropositionFieldProvenance objects")
        provenance_names = tuple(item.field_name for item in self.field_provenance)
        if len(provenance_names) != len(set(provenance_names)):
            raise ValueError("field provenance names must be unique")
        if not set(names).issubset(set(provenance_names)):
            raise ValueError("every proposition role must have field-level provenance")
        provenance_by_name = {item.field_name: item for item in self.field_provenance}
        for role in self.roles:
            source = provenance_by_name[role.role_name]
            if role.role_kind == "event":
                if source.event_ids != (role.value_id,):
                    raise ValueError(
                        f"event role provenance must identify its exact event: {role.role_name}"
                    )
                if len(source.observation_ids) != 1 or source.relationship_ids:
                    raise ValueError(
                        f"event role provenance must identify exactly one observation: {role.role_name}"
                    )
            elif source.relationship_ids or not source.event_ids:
                raise ValueError(
                    f"entity role provenance must identify its supporting events: {role.role_name}"
                )
        object.__setattr__(self, "construction_id", validate_text(self.construction_id, "construction_id"))
        if type(self.construction_version) is not int or self.construction_version < 1:
            raise ValueError("construction_version must be a positive integer")
        object.__setattr__(
            self, "semantics_fingerprint",
            validate_text(self.semantics_fingerprint, "semantics_fingerprint"),
        )
        algorithm = validate_text(
            self.identity_algorithm_version, "identity_algorithm_version"
        )
        if algorithm not in SUPPORTED_PROPOSITION_IDENTITY_ALGORITHMS:
            raise ValueError("unsupported proposition identity algorithm version")
        object.__setattr__(self, "identity_algorithm_version", algorithm)
        if not isinstance(self.parameters_json, str):
            raise TypeError("parameters_json must be a JSON object string")
        validated_parameters = validate_json_object(self.parameters_json, "parameters_json")
        parameters = json.loads(validated_parameters)
        if not isinstance(parameters, dict):
            raise ValueError("parameters_json must contain a JSON object")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposition schema version")

    def validate_against(self, semantics: PropositionSemantics) -> None:
        """Validate role shape, types, temporal constraints and parameters against exact semantics."""
        if not isinstance(semantics, PropositionSemantics):
            raise TypeError("semantics must be PropositionSemantics")
        if (self.proposition_type, self.semantics_version) != (
            semantics.proposition_type, semantics.semantics_version
        ):
            raise ValueError("proposition semantics identity does not match")
        expected_fingerprint = proposition_semantics_fingerprint(semantics)
        if self.semantics_fingerprint != expected_fingerprint:
            raise ValueError("proposition semantics fingerprint does not match exact semantics")
        expected = {item.role_name: item for item in semantics.roles}
        actual = {item.role_name: item for item in self.roles}
        for name, definition in expected.items():
            value = actual.get(name)
            if value is None:
                if definition.required:
                    raise ValueError(f"missing required proposition role: {name}")
                continue
            if value.role_kind != definition.role_kind:
                raise ValueError(f"role kind mismatch for {name}")
        unexpected = set(actual) - set(expected)
        if unexpected:
            raise ValueError(f"undeclared proposition roles: {sorted(unexpected)}")
        times = {item.role_name: item.occurred_at for item in self.roles if item.role_kind == "event"}
        for constraint in semantics.temporal_constraints:
            earlier = times[constraint.earlier_role]
            later = times[constraint.later_role]
            if earlier >= later:
                raise ValueError(
                    f"temporal constraint violated: {constraint.earlier_role} must precede {constraint.later_role}"
                )
            if constraint.max_interval_seconds is not None:
                if (later - earlier).total_seconds() > constraint.max_interval_seconds:
                    raise ValueError(
                        f"temporal interval exceeds {constraint.max_interval_seconds} seconds"
                    )
        scope = self.temporal_scope
        if semantics.temporal_scope is not None:
            start = times[semantics.temporal_scope.start_role]
            end = times[semantics.temporal_scope.end_role]
            expected_start_inclusive = semantics.temporal_scope.start_inclusive
            expected_end_inclusive = semantics.temporal_scope.end_inclusive
        else:
            required_event_role_names = {
                item.role_name for item in semantics.roles
                if item.role_kind == "event" and item.required
            }
            if not required_event_role_names:
                raise ValueError(
                    "implicit temporal scope requires at least one required event role"
                )
            required_event_times = [times[name] for name in required_event_role_names]
            start = min(required_event_times)
            end = max(required_event_times)
            expected_start_inclusive = True
            expected_end_inclusive = True
        if scope.starts_at != start or scope.ends_at != end:
            raise ValueError("proposition temporal scope does not match semantic boundaries")
        if (
            scope.start_inclusive != expected_start_inclusive
            or scope.end_inclusive != expected_end_inclusive
        ):
            raise ValueError("proposition temporal scope inclusivity does not match semantics")
        parameters = json.loads(self.parameters_json)
        definitions = {item.parameter_name: item for item in semantics.parameters}
        if set(parameters) - set(definitions):
            raise ValueError("proposition contains undeclared semantic parameters")
        actual_provenance = {item.field_name for item in self.field_provenance}
        expected_provenance = set(actual)
        for name, definition in definitions.items():
            if name not in parameters:
                if definition.required:
                    raise ValueError(f"missing required semantic parameter: {name}")
                continue
            value = parameters[name]
            valid = (
                isinstance(value, str) if definition.value_type == "string"
                else type(value) is int if definition.value_type == "integer"
                else type(value) is bool
            )
            if not valid:
                raise ValueError(f"semantic parameter type mismatch: {name}")
            if definition.value_type == "string":
                validate_text(value, f"semantic parameter {name}")
            if (
                definition.default_value is None
                or value != definition.default_value
                or name in actual_provenance
            ):
                expected_provenance.add(name)
        if set(item.field_name for item in self.field_provenance) != expected_provenance:
            raise ValueError("every non-default proposition field must have exact field-level provenance")
        expected_id = deterministic_proposition_id(
            self.proposition_type,
            self.semantics_version,
            self.roles,
            self.temporal_scope,
            self.parameters_json,
            semantics_fingerprint=self.semantics_fingerprint,
            identity_algorithm_version=self.identity_algorithm_version,
        )
        if self.proposition_id != expected_id:
            raise ValueError("proposition ID does not match its deterministic semantic identity")


@dataclass(frozen=True, slots=True)
class StructuredHypothesis:
    """A hypothesis whose analytical content is an immutable structured proposition."""

    hypothesis_id: str
    proposition: Proposition
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", validate_id(self.hypothesis_id, "hypothesis_id"))
        if not isinstance(self.proposition, Proposition):
            raise TypeError("proposition must be a Proposition")
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at, "created_at"))
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported structured hypothesis schema version")


def deterministic_proposition_id(
    proposition_type: str,
    semantics_version: int,
    roles: tuple[PropositionRoleValue, ...],
    temporal_scope: PropositionTemporalScope,
    parameters_json: str = "{}",
    *,
    semantics_fingerprint: str | None = None,
    identity_algorithm_version: str = LEGACY_PROPOSITION_IDENTITY_ALGORITHM,
) -> str:
    """Return a versioned UUID5 identity independent of construction history."""
    algorithm = validate_text(identity_algorithm_version, "identity_algorithm_version")
    if algorithm not in SUPPORTED_PROPOSITION_IDENTITY_ALGORITHMS:
        raise ValueError("unsupported proposition identity algorithm version")
    canonical = {
        "type": validate_text(proposition_type, "proposition_type"),
        "semantics_version": semantics_version,
        "roles": [
            {
                "name": role.role_name,
                "kind": role.role_kind,
                "id": role.value_id,
                "occurred_at": role.occurred_at.isoformat() if role.occurred_at else None,
            }
            for role in sorted(roles, key=lambda item: item.role_name)
        ],
        "scope": {
            "start": temporal_scope.starts_at.isoformat(),
            "end": temporal_scope.ends_at.isoformat(),
            "start_inclusive": temporal_scope.start_inclusive,
            "end_inclusive": temporal_scope.end_inclusive,
        },
        "parameters": json.loads(parameters_json),
    }
    if algorithm == CURRENT_PROPOSITION_IDENTITY_ALGORITHM:
        if semantics_fingerprint is None:
            raise ValueError("uuid5-v2 identity requires a semantics fingerprint")
        canonical["identity_algorithm_version"] = algorithm
        canonical["semantics_fingerprint"] = validate_text(
            semantics_fingerprint, "semantics_fingerprint"
        )
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return str(uuid.uuid5(_PROPOSITION_NAMESPACE, encoded))
