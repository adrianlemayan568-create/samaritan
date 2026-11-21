
"""Immutable semantic definitions for structured propositions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import unicodedata

from samaritan.contracts import SCHEMA_VERSION, validate_identifier, validate_text


ROLE_KINDS = frozenset({"entity", "event"})
ROLE_CARDINALITIES = frozenset({"one"})
TEMPORAL_RELATIONS = frozenset({"precedes"})
PARAMETER_TYPES = frozenset({"string", "integer", "boolean"})


def _validate_nfc_identifier(value: str, field_name: str) -> str:
    """Require identifiers to use one canonical Unicode representation."""
    normalized = unicodedata.normalize("NFC", value)
    if value != normalized:
        raise ValueError(f"{field_name} must be Unicode NFC normalized")
    return validate_identifier(value, field_name)


def _validate_schema_version(value: int, field_name: str) -> None:
    """Require the exact supported schema version and an actual integer."""
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported {field_name}")


@dataclass(frozen=True, slots=True)
class PropositionRoleDefinition:
    """Define a typed role and its meaning within a proposition."""

    role_name: str
    role_kind: str
    cardinality: str
    meaning: str
    entity_type: str | None = None
    event_type: str | None = None
    required: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role_name", _validate_nfc_identifier(
                validate_text(self.role_name, "role_name"), "role_name"
            )
        )
        object.__setattr__(
            self, "role_kind", validate_text(self.role_kind, "role_kind")
        )
        object.__setattr__(
            self, "cardinality", validate_text(self.cardinality, "cardinality")
        )

        if self.role_kind not in ROLE_KINDS:
            raise ValueError("unsupported role kind")

        if self.cardinality not in ROLE_CARDINALITIES:
            raise ValueError("unsupported role cardinality")

        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")

        if self.role_kind == "entity":
            if self.entity_type is None or self.event_type is not None:
                raise ValueError(
                    "entity roles require entity_type and forbid event_type"
                )
            object.__setattr__(
                self,
                "entity_type",
                _validate_nfc_identifier(
                    validate_text(self.entity_type, "entity_type"), "entity_type"
                ),
            )
        else:
            if self.event_type is None or self.entity_type is not None:
                raise ValueError(
                    "event roles require event_type and forbid entity_type"
                )
            object.__setattr__(
                self,
                "event_type",
                _validate_nfc_identifier(
                    validate_text(self.event_type, "event_type"), "event_type"
                ),
            )

        _validate_schema_version(
            self.schema_version, "proposition role schema version"
        )
        object.__setattr__(
            self, "meaning", validate_text(self.meaning, "meaning")
        )


@dataclass(frozen=True, slots=True)
class TemporalConstraintDefinition:
    """Define a temporal relationship between two event roles."""

    earlier_role: str
    later_role: str
    relation: str = "precedes"
    schema_version: int = SCHEMA_VERSION
    max_interval_seconds: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "earlier_role", _validate_nfc_identifier(
                validate_text(self.earlier_role, "earlier_role"), "earlier_role"
            )
        )
        object.__setattr__(
            self, "later_role", _validate_nfc_identifier(
                validate_text(self.later_role, "later_role"), "later_role"
            )
        )
        object.__setattr__(
            self, "relation", validate_text(self.relation, "relation")
        )

        if self.earlier_role == self.later_role:
            raise ValueError("temporal roles must be different")

        if self.relation not in TEMPORAL_RELATIONS:
            raise ValueError("unsupported temporal relation")
        if self.max_interval_seconds is not None:
            if type(self.max_interval_seconds) is not int:
                raise TypeError("max_interval_seconds must be an integer or None")
            if self.max_interval_seconds <= 0:
                raise ValueError("max_interval_seconds must be positive")

        _validate_schema_version(
            self.schema_version, "temporal constraint schema version"
        )


@dataclass(frozen=True, slots=True)
class TemporalScopeDefinition:
    """Define the event roles that establish a proposition's time scope."""

    start_role: str
    end_role: str
    start_inclusive: bool = True
    end_inclusive: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start_role", _validate_nfc_identifier(
                validate_text(self.start_role, "start_role"), "start_role"
            )
        )
        object.__setattr__(
            self, "end_role", _validate_nfc_identifier(
                validate_text(self.end_role, "end_role"), "end_role"
            )
        )

        if self.start_role == self.end_role:
            raise ValueError("temporal scope boundary roles must be different")

        if type(self.start_inclusive) is not bool:
            raise TypeError("start_inclusive must be a boolean")

        if type(self.end_inclusive) is not bool:
            raise TypeError("end_inclusive must be a boolean")

        _validate_schema_version(
            self.schema_version, "temporal scope schema version"
        )


@dataclass(frozen=True, slots=True)
class SemanticParameterDefinition:
    """Define a typed semantic parameter."""

    parameter_name: str
    value_type: str
    required: bool = False
    default_value: str | int | bool | None = None
    schema_version: int = SCHEMA_VERSION
    meaning: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_name",
            _validate_nfc_identifier(
                validate_text(self.parameter_name, "parameter_name"), "parameter_name"
            ),
        )
        object.__setattr__(
            self, "value_type", validate_text(self.value_type, "value_type")
        )

        if self.value_type not in PARAMETER_TYPES:
            raise ValueError("unsupported semantic parameter type")

        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")

        if self.required and self.default_value is not None:
            raise ValueError("required parameters cannot have a default value")

        if not isinstance(self.meaning, str):
            raise TypeError("meaning must be a string")
        if self.meaning:
            object.__setattr__(self, "meaning", validate_text(self.meaning, "meaning"))

        if self.default_value is not None:
            if self.value_type == "string":
                if not isinstance(self.default_value, str):
                    raise TypeError("string parameter default must be a string")
                validate_text(self.default_value, "default_value")
            elif self.value_type == "integer":
                if type(self.default_value) is not int:
                    raise TypeError("integer parameter default must be an integer")
            elif self.value_type == "boolean":
                if type(self.default_value) is not bool:
                    raise TypeError("boolean parameter default must be a boolean")
            else:
                raise ValueError(
                    "defaults for this parameter type are not yet supported"
                )

        _validate_schema_version(
            self.schema_version, "semantic parameter schema version"
        )


@dataclass(frozen=True, slots=True)
class PropositionSemantics:
    """Define the meaning and structural requirements of a proposition type."""

    proposition_type: str
    semantics_version: int
    roles: tuple[PropositionRoleDefinition, ...]
    temporal_constraints: tuple[TemporalConstraintDefinition, ...] = ()
    parameters: tuple[SemanticParameterDefinition, ...] = ()
    description: str = ""
    schema_version: int = SCHEMA_VERSION
    temporal_scope: TemporalScopeDefinition | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposition_type",
            _validate_nfc_identifier(
                validate_text(self.proposition_type, "proposition_type"), "proposition_type"
            ),
        )
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        if self.description:
            object.__setattr__(
                self, "description", validate_text(self.description, "description")
            )

        if type(self.semantics_version) is not int:
            raise TypeError("semantics_version must be an integer")
        if self.semantics_version <= 0:
            raise ValueError("semantics_version must be a positive integer")

        if not isinstance(self.roles, tuple) or not self.roles:
            raise TypeError("roles must be a non-empty tuple")

        if not all(
            type(item) is PropositionRoleDefinition for item in self.roles
        ):
            raise TypeError(
                "roles must contain PropositionRoleDefinition objects"
            )

        if not isinstance(self.temporal_constraints, tuple):
            raise TypeError("temporal_constraints must be a tuple")

        if not all(
            type(item) is TemporalConstraintDefinition
            for item in self.temporal_constraints
        ):
            raise TypeError(
                "temporal_constraints must contain "
                "TemporalConstraintDefinition objects"
            )

        if (
            self.temporal_scope is not None
            and not type(self.temporal_scope) is TemporalScopeDefinition
        ):
            raise TypeError(
                "temporal_scope must be a TemporalScopeDefinition or None"
            )

        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple")

        if not all(
            type(item) is SemanticParameterDefinition
            for item in self.parameters
        ):
            raise TypeError(
                "parameters must contain SemanticParameterDefinition objects"
            )

        role_names = [item.role_name for item in self.roles]
        if len(role_names) != len(set(role_names)):
            raise ValueError("proposition role names must be unique")

        event_roles = {
            item.role_name: item
            for item in self.roles
            if item.role_kind == "event"
        }
        event_role_names = set(event_roles)
        required_event_role_names = {
            name for name, role in event_roles.items() if role.required
        }
        if self.temporal_scope is None and not required_event_role_names:
            raise ValueError(
                "semantics without an explicit temporal scope must declare at least one required event role"
            )

        constraint_pairs = set()
        for constraint in self.temporal_constraints:
            if (
                constraint.earlier_role not in event_role_names
                or constraint.later_role not in event_role_names
            ):
                raise ValueError(
                    "temporal constraints must reference declared event roles"
                )
            if (
                not event_roles[constraint.earlier_role].required
                or not event_roles[constraint.later_role].required
            ):
                raise ValueError(
                    "temporal constraints must reference required event roles"
                )

            pair = (constraint.earlier_role, constraint.later_role)
            if pair in constraint_pairs:
                raise ValueError("duplicate temporal constraint")
            constraint_pairs.add(pair)

        # Temporal precedence must form a directed acyclic graph.
        adjacency = {name: set() for name in event_role_names}
        in_degree = {name: 0 for name in event_role_names}

        for constraint in self.temporal_constraints:
            earlier = constraint.earlier_role
            later = constraint.later_role
            if later not in adjacency[earlier]:
                adjacency[earlier].add(later)
                in_degree[later] += 1

        ready = [
            name for name, degree in in_degree.items()
            if degree == 0
        ]
        visited_count = 0

        while ready:
            current = ready.pop()
            visited_count += 1

            for later in adjacency[current]:
                in_degree[later] -= 1
                if in_degree[later] == 0:
                    ready.append(later)

        if visited_count != len(event_role_names):
            raise ValueError("temporal constraints must not contain cycles")

        if self.temporal_scope is not None:
            start_role = self.temporal_scope.start_role
            end_role = self.temporal_scope.end_role

            if (
                start_role not in event_role_names
                or end_role not in event_role_names
            ):
                raise ValueError(
                    "temporal scope must reference declared event roles"
                )
            if (
                not event_roles[start_role].required
                or not event_roles[end_role].required
            ):
                raise ValueError(
                    "temporal scope must reference required event roles"
                )

            # Require a directed temporal path from start to end.
            reachable = set()
            pending = [start_role]

            while pending:
                current = pending.pop()

                if current in reachable:
                    continue

                reachable.add(current)
                pending.extend(adjacency[current] - reachable)

            if end_role not in reachable:
                raise ValueError(
                    "temporal scope start role must precede end role"
                )

        parameter_names = [item.parameter_name for item in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("semantic parameter names must be unique")
        if set(role_names) & set(parameter_names):
            raise ValueError("semantic role and parameter names must be distinct")

        _validate_schema_version(
            self.schema_version, "proposition semantics schema version"
        )


class PropositionSemanticsRegistry:
    """Store immutable proposition semantics by exact type and version."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, int], PropositionSemantics] | Mapping[
            tuple[str, int], PropositionSemantics
        ] = {}
        self._frozen = False

    def register(self, definition: PropositionSemantics) -> None:
        if type(definition) is not PropositionSemantics:
            raise TypeError("definition must be exactly PropositionSemantics")
        if self._frozen:
            raise RuntimeError("proposition semantics registry is frozen")

        key = (definition.proposition_type, definition.semantics_version)

        if key in self._definitions:
            raise ValueError("proposition semantics version is already registered")

        self._definitions[key] = definition

    def freeze(self) -> None:
        """Prevent further registration and expose an immutable definition map."""
        if not self._frozen:
            self._definitions = MappingProxyType(dict(self._definitions))
            self._frozen = True

    @property
    def definitions(self) -> Mapping[tuple[str, int], PropositionSemantics]:
        """Return a read-only view of registered semantics definitions."""
        return MappingProxyType(dict(self._definitions))

    def get(
        self,
        proposition_type: str,
        semantics_version: int,
    ) -> PropositionSemantics:
        proposition_type = validate_text(proposition_type, "proposition_type")

        if type(semantics_version) is not int:
            raise TypeError("semantics_version must be an integer")
        if semantics_version <= 0:
            raise ValueError("semantics_version must be a positive integer")

        try:
            return self._definitions[(proposition_type, semantics_version)]
        except KeyError as exc:
            raise KeyError(
                f"unknown proposition semantics: "
                f"{proposition_type}@{semantics_version}"
            ) from exc

    def contains(self, proposition_type: str, semantics_version: int) -> bool:
        try:
            self.get(proposition_type, semantics_version)
        except KeyError:
            return False
        return True

    def __len__(self) -> int:
        return len(self._definitions)


def build_default_proposition_semantics_registry() -> PropositionSemanticsRegistry:
    """Build the initial registry of supported proposition semantics."""

    registry = PropositionSemanticsRegistry()

    registry.register(
        PropositionSemantics(
            proposition_type="account_authentication_sequence",
            semantics_version=1,
            description=(
                "A specified user account is associated with an authentication "
                "failure, a subsequent successful authentication, and subsequent "
                "host activity, in the required temporal order."
            ),
            roles=(
                PropositionRoleDefinition(
                    role_name="account",
                    role_kind="entity",
                    cardinality="one",
                    entity_type="user_account",
                    meaning=(
                        "The user account associated with the authentication events."
                    ),
                ),
                PropositionRoleDefinition(
                    role_name="failure_event",
                    role_kind="event",
                    cardinality="one",
                    event_type="authentication_failure",
                    meaning=(
                        "An authentication failure associated with the specified account."
                    ),
                ),
                PropositionRoleDefinition(
                    role_name="success_event",
                    role_kind="event",
                    cardinality="one",
                    event_type="authentication_success",
                    meaning=(
                        "A subsequent successful authentication associated with that account."
                    ),
                ),
                PropositionRoleDefinition(
                    role_name="host_activity_event",
                    role_kind="event",
                    cardinality="one",
                    event_type="host_activity",
                    meaning=(
                        "Subsequent host activity associated with that account."
                    ),
                ),
            ),
            temporal_constraints=(
                TemporalConstraintDefinition(
                    earlier_role="failure_event",
                    later_role="success_event",
                    max_interval_seconds=300,
                ),
                TemporalConstraintDefinition(
                    earlier_role="success_event",
                    later_role="host_activity_event",
                    max_interval_seconds=300,
                ),
            ),
            temporal_scope=TemporalScopeDefinition(
                start_role="failure_event",
                end_role="host_activity_event",
                start_inclusive=True,
                end_inclusive=True,
            ),
        )
    )

    return registry
