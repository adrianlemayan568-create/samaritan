"""Finite, immutable definitions for deterministic proposition construction."""

from __future__ import annotations

from dataclasses import dataclass

from samaritan.contracts import SCHEMA_VERSION, validate_text


@dataclass(frozen=True, slots=True)
class EventRoleBinding:
    """Resolve one required semantic event role from exactly one event type."""

    semantic_role: str
    event_type: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("semantic_role", "event_type"):
            object.__setattr__(self, field_name, validate_text(getattr(self, field_name), field_name))
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported event role binding schema version")


@dataclass(frozen=True, slots=True)
class ParticipantRoleBinding:
    """Identify the participant role used to resolve an entity from one event role."""

    event_role: str
    participant_role: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_role", validate_text(self.event_role, "event_role"))
        object.__setattr__(self, "participant_role", validate_text(self.participant_role, "participant_role"))
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported participant role binding schema version")


@dataclass(frozen=True, slots=True)
class EntityRoleBinding:
    """Resolve one semantic entity role from explicitly mapped event participants."""

    semantic_role: str
    participant_bindings: tuple[ParticipantRoleBinding, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_role", validate_text(self.semantic_role, "semantic_role"))
        if not isinstance(self.participant_bindings, tuple) or not self.participant_bindings:
            raise TypeError("participant_bindings must be a non-empty tuple")
        if not all(isinstance(item, ParticipantRoleBinding) for item in self.participant_bindings):
            raise TypeError("participant_bindings must contain ParticipantRoleBinding objects")
        event_roles = tuple(item.event_role for item in self.participant_bindings)
        if len(event_roles) != len(set(event_roles)):
            raise ValueError("participant event roles must be unique")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported entity role binding schema version")


@dataclass(frozen=True, slots=True)
class EventAttributeParameterBinding:
    """Bind a semantic parameter to one registered top-level event attribute."""

    parameter_name: str
    event_role: str
    attribute_name: str
    meaning: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("parameter_name", "event_role", "attribute_name", "meaning"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        _validate_version(self.schema_version, "event attribute parameter binding")


@dataclass(frozen=True, slots=True)
class TemporalDifferenceParameterBinding:
    """Derive an integer number of seconds between two declared event roles."""

    parameter_name: str
    earlier_event_role: str
    later_event_role: str
    meaning: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("parameter_name", "earlier_event_role", "later_event_role", "meaning"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        if self.earlier_event_role == self.later_event_role:
            raise ValueError("temporal difference event roles must be different")
        _validate_version(self.schema_version, "temporal difference parameter binding")


@dataclass(frozen=True, slots=True)
class RelationshipParameterBinding:
    """Bind a semantic string parameter to one explicit directed relationship."""

    parameter_name: str
    source_entity_role: str
    target_entity_role: str
    relationship_type: str
    relationship_field: str
    meaning: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "parameter_name", "source_entity_role", "target_entity_role",
            "relationship_type", "relationship_field", "meaning",
        ):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        if self.source_entity_role == self.target_entity_role:
            raise ValueError("relationship entity roles must be different")
        if self.relationship_field not in {"relationship_id", "relationship_type", "observed_at"}:
            raise ValueError("unsupported relationship field binding")
        _validate_version(self.schema_version, "relationship parameter binding")


def _validate_version(value: int, name: str) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError(f"unsupported {name} schema version")


@dataclass(frozen=True, slots=True)
class PropositionConstructionDefinition:
    """Bind exact proposition semantics fields to validated source structures."""

    construction_id: str
    construction_version: int
    proposition_type: str
    semantics_version: int
    event_bindings: tuple[EventRoleBinding, ...]
    entity_bindings: tuple[EntityRoleBinding, ...]
    description: str
    event_attribute_bindings: tuple[EventAttributeParameterBinding, ...] = ()
    temporal_parameter_bindings: tuple[TemporalDifferenceParameterBinding, ...] = ()
    relationship_parameter_bindings: tuple[RelationshipParameterBinding, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("construction_id", "proposition_type", "description"):
            object.__setattr__(self, field_name, validate_text(getattr(self, field_name), field_name))
        for field_name in ("construction_version", "semantics_version"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.event_bindings, tuple) or not self.event_bindings:
            raise TypeError("event_bindings must be a non-empty tuple")
        if not all(isinstance(item, EventRoleBinding) for item in self.event_bindings):
            raise TypeError("event_bindings must contain EventRoleBinding objects")
        if not isinstance(self.entity_bindings, tuple):
            raise TypeError("entity_bindings must be a tuple")
        if not all(isinstance(item, EntityRoleBinding) for item in self.entity_bindings):
            raise TypeError("entity_bindings must contain EntityRoleBinding objects")
        semantic_roles = tuple(item.semantic_role for item in self.event_bindings + self.entity_bindings)
        if len(semantic_roles) != len(set(semantic_roles)):
            raise ValueError("each semantic role must have exactly one construction binding")
        event_role_names = {item.semantic_role for item in self.event_bindings}
        for binding in self.entity_bindings:
            if not {item.event_role for item in binding.participant_bindings}.issubset(event_role_names):
                raise ValueError("entity binding references an undeclared event binding")
        parameter_bindings = (
            self.event_attribute_bindings
            + self.temporal_parameter_bindings
            + self.relationship_parameter_bindings
        )
        if not isinstance(self.event_attribute_bindings, tuple) or not all(
            isinstance(item, EventAttributeParameterBinding) for item in self.event_attribute_bindings
        ):
            raise TypeError("event_attribute_bindings must contain EventAttributeParameterBinding objects")
        if not isinstance(self.temporal_parameter_bindings, tuple) or not all(
            isinstance(item, TemporalDifferenceParameterBinding) for item in self.temporal_parameter_bindings
        ):
            raise TypeError("temporal_parameter_bindings must contain TemporalDifferenceParameterBinding objects")
        if not isinstance(self.relationship_parameter_bindings, tuple) or not all(
            isinstance(item, RelationshipParameterBinding) for item in self.relationship_parameter_bindings
        ):
            raise TypeError("relationship_parameter_bindings must contain RelationshipParameterBinding objects")
        parameter_names = tuple(item.parameter_name for item in parameter_bindings)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("semantic parameter bindings must be unique")
        for binding in self.event_attribute_bindings:
            if binding.event_role not in event_role_names:
                raise ValueError("event attribute binding references an undeclared event role")
        for binding in self.temporal_parameter_bindings:
            if not {binding.earlier_event_role, binding.later_event_role}.issubset(event_role_names):
                raise ValueError("temporal parameter binding references an undeclared event role")
        entity_role_names = {item.semantic_role for item in self.entity_bindings}
        for binding in self.relationship_parameter_bindings:
            if not {binding.source_entity_role, binding.target_entity_role}.issubset(entity_role_names):
                raise ValueError("relationship parameter binding references an undeclared entity role")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported proposition construction schema version")


class PropositionConstructionRegistry:
    """Register immutable construction definitions by exact ID and version."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, int], PropositionConstructionDefinition] = {}

    def register(self, definition: PropositionConstructionDefinition) -> None:
        if not isinstance(definition, PropositionConstructionDefinition):
            raise TypeError("definition must be PropositionConstructionDefinition")
        key = (definition.construction_id, definition.construction_version)
        if key in self._definitions:
            raise ValueError("proposition construction version is already registered")
        self._definitions[key] = definition

    def get(self, construction_id: str, construction_version: int) -> PropositionConstructionDefinition:
        construction_id = validate_text(construction_id, "construction_id")
        if type(construction_version) is not int or construction_version < 1:
            raise ValueError("construction_version must be a positive integer")
        try:
            return self._definitions[(construction_id, construction_version)]
        except KeyError as exc:
            raise KeyError(f"unknown proposition construction: {construction_id}@{construction_version}") from exc

    def __len__(self) -> int:
        return len(self._definitions)


def build_default_proposition_construction_registry() -> PropositionConstructionRegistry:
    """Build the initial account-authentication-sequence constructor."""
    registry = PropositionConstructionRegistry()
    registry.register(
        PropositionConstructionDefinition(
            construction_id="account_authentication_sequence",
            construction_version=1,
            proposition_type="account_authentication_sequence",
            semantics_version=1,
            description="Construct an account authentication sequence from explicit event participants.",
            event_bindings=(
                EventRoleBinding("failure_event", "authentication_failure"),
                EventRoleBinding("success_event", "authentication_success"),
                EventRoleBinding("host_activity_event", "host_activity"),
            ),
            entity_bindings=(
                EntityRoleBinding(
                    "account",
                    (
                        ParticipantRoleBinding("failure_event", "target"),
                        ParticipantRoleBinding("success_event", "target"),
                        ParticipantRoleBinding("host_activity_event", "subject"),
                    ),
                ),
            ),
        )
    )
    return registry
