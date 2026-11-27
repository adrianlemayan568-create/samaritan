"""Immutable contracts for evidence-driven hypothesis generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from samaritan.analysis_contracts import EvidenceReference, Hypothesis
from samaritan.contracts import (
    SCHEMA_VERSION,
    normalize_timestamp,
    validate_id,
    validate_text,
)


@dataclass(frozen=True, slots=True)
class PatternEventRole:
    """Define one named event position in a hypothesis pattern."""

    role_name: str
    event_type: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role_name", validate_text(self.role_name, "role_name")
        )
        object.__setattr__(
            self, "event_type", validate_text(self.event_type, "event_type")
        )


@dataclass(frozen=True, slots=True)
class PatternCorrelationConstraint:
    """Require a registered correlation rule between two event positions."""

    first_event_role: str
    second_event_role: str
    correlation_rule_id: str

    def __post_init__(self) -> None:
        for field_name in ("first_event_role", "second_event_role"):
            object.__setattr__(
                self,
                field_name,
                validate_text(getattr(self, field_name), field_name),
            )
        if self.first_event_role == self.second_event_role:
            raise ValueError(
                "correlation constraint requires two distinct event roles"
            )
        object.__setattr__(
            self,
            "correlation_rule_id",
            validate_text(self.correlation_rule_id, "correlation_rule_id"),
        )


@dataclass(frozen=True, slots=True)
class PatternEntityBinding:
    """Bind a named identity to a participant role at an event position."""

    binding_name: str
    event_role: str
    entity_type: str
    participant_role: str

    def __post_init__(self) -> None:
        for field_name in (
            "binding_name",
            "event_role",
            "entity_type",
            "participant_role",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_text(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class PatternTemporalConstraint:
    """Constrain the temporal relationship between two event positions."""

    earlier_event_role: str
    later_event_role: str
    strict: bool = True
    maximum_interval_seconds: int | float | None = None

    def __post_init__(self) -> None:
        for field_name in ("earlier_event_role", "later_event_role"):
            object.__setattr__(
                self,
                field_name,
                validate_text(getattr(self, field_name), field_name),
            )
        if self.earlier_event_role == self.later_event_role:
            raise ValueError(
                "temporal constraint requires two distinct event roles"
            )
        if type(self.strict) is not bool:
            raise TypeError("strict must be a boolean")

        interval = self.maximum_interval_seconds
        if interval is not None:
            if type(interval) not in (int, float):
                raise TypeError(
                    "maximum_interval_seconds must be numeric or None"
                )
            if not math.isfinite(interval) or interval < 0:
                raise ValueError(
                    "maximum_interval_seconds must be finite and non-negative"
                )
            if self.strict and interval == 0:
                raise ValueError(
                    "a strict temporal constraint cannot have a zero maximum interval"
                )


@dataclass(frozen=True, slots=True)
class HypothesisPattern:
    """Define a complete, versioned, declarative event pattern.

    Each named event role has cardinality exactly one. Event IDs may be
    reused by separate matches, but one event cannot fill two roles in
    the same match. Competing assignments that differ in exactly one
    role, or permute the same event set across roles, are rejected as
    ambiguous.
    """

    pattern_id: str
    version: int
    event_roles: tuple[PatternEventRole, ...]
    correlation_constraints: tuple[PatternCorrelationConstraint, ...]
    entity_bindings: tuple[PatternEntityBinding, ...]
    temporal_constraints: tuple[PatternTemporalConstraint, ...]
    event_cardinality: str = "exactly_one_per_role"
    event_reuse_policy: str = "across_matches_only"
    ambiguity_policy: str = "reject_competing_assignments"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pattern_id", validate_text(self.pattern_id, "pattern_id")
        )
        if type(self.version) is not int or self.version < 1:
            raise ValueError("pattern version must be a positive integer")

        object.__setattr__(
            self, "event_cardinality",
            validate_text(self.event_cardinality, "event_cardinality"),
        )
        object.__setattr__(
            self, "event_reuse_policy",
            validate_text(self.event_reuse_policy, "event_reuse_policy"),
        )
        object.__setattr__(
            self, "ambiguity_policy",
            validate_text(self.ambiguity_policy, "ambiguity_policy"),
        )
        if self.event_cardinality != "exactly_one_per_role":
            raise ValueError("unsupported pattern event cardinality policy")
        if self.event_reuse_policy != "across_matches_only":
            raise ValueError("unsupported pattern event reuse policy")
        if self.ambiguity_policy != "reject_competing_assignments":
            raise ValueError("unsupported pattern ambiguity policy")

        collections = (
            ("event_roles", self.event_roles, PatternEventRole),
            (
                "correlation_constraints",
                self.correlation_constraints,
                PatternCorrelationConstraint,
            ),
            ("entity_bindings", self.entity_bindings, PatternEntityBinding),
            (
                "temporal_constraints",
                self.temporal_constraints,
                PatternTemporalConstraint,
            ),
        )
        for name, value, expected_type in collections:
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be a tuple")
            if not all(isinstance(item, expected_type) for item in value):
                raise TypeError(
                    f"{name} must contain {expected_type.__name__} objects"
                )

        if len(self.event_roles) < 2:
            raise ValueError("a pattern requires at least two event roles")
        if not self.correlation_constraints:
            raise ValueError(
                "a pattern requires at least one correlation constraint"
            )
        if not self.entity_bindings:
            raise ValueError("a pattern requires entity bindings")

        role_names = tuple(item.role_name for item in self.event_roles)
        if len(role_names) != len(set(role_names)):
            raise ValueError("event role names must be unique")
        known_roles = set(role_names)

        adjacency = {role_name: set() for role_name in role_names}
        correlation_keys = []
        for constraint in self.correlation_constraints:
            endpoints = {
                constraint.first_event_role,
                constraint.second_event_role,
            }
            if not endpoints.issubset(known_roles):
                raise ValueError(
                    "correlation constraint references an unknown event role"
                )
            first, second = (
                constraint.first_event_role,
                constraint.second_event_role,
            )
            adjacency[first].add(second)
            adjacency[second].add(first)
            correlation_keys.append(
                (frozenset(endpoints), constraint.correlation_rule_id)
            )
        if len(correlation_keys) != len(set(correlation_keys)):
            raise ValueError("duplicate pattern correlation constraints")

        reached = set()
        pending = [role_names[0]]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(adjacency[current] - reached)
        if reached != known_roles:
            raise ValueError(
                "all event roles must belong to one connected correlation pattern"
            )

        binding_types: dict[str, str] = {}
        binding_positions: set[tuple[str, str]] = set()
        bindings_by_role: dict[str, set[str]] = {
            role_name: set() for role_name in role_names
        }
        for binding in self.entity_bindings:
            if binding.event_role not in known_roles:
                raise ValueError(
                    "entity binding references an unknown event role"
                )
            key = (binding.binding_name, binding.event_role)
            if key in binding_positions:
                raise ValueError(
                    "duplicate entity binding for the same event role"
                )
            binding_positions.add(key)
            bindings_by_role[binding.event_role].add(binding.binding_name)

            existing_type = binding_types.get(binding.binding_name)
            if existing_type is not None and existing_type != binding.entity_type:
                raise ValueError(
                    "a named entity binding must use one consistent entity type"
                )
            binding_types[binding.binding_name] = binding.entity_type

        for binding_name in binding_types:
            positions = {
                event_role
                for name, event_role in binding_positions
                if name == binding_name
            }
            if len(positions) < 2:
                raise ValueError(
                    "each entity binding must connect at least two event roles"
                )

        for constraint in self.correlation_constraints:
            common_bindings = (
                bindings_by_role[constraint.first_event_role]
                & bindings_by_role[constraint.second_event_role]
            )
            if not common_bindings:
                raise ValueError(
                    "correlated event roles must share a named entity binding"
                )

        temporal_keys = []
        temporal_edges: dict[str, list[tuple[str, bool]]] = {
            role_name: [] for role_name in role_names
        }
        for constraint in self.temporal_constraints:
            endpoints = {
                constraint.earlier_event_role,
                constraint.later_event_role,
            }
            if not endpoints.issubset(known_roles):
                raise ValueError(
                    "temporal constraint references an unknown event role"
                )
            temporal_keys.append(
                (
                    constraint.earlier_event_role,
                    constraint.later_event_role,
                )
            )
            temporal_edges[constraint.earlier_event_role].append(
                (constraint.later_event_role, constraint.strict)
            )
        if len(temporal_keys) != len(set(temporal_keys)):
            raise ValueError("duplicate temporal constraints")

        # A directed temporal cycle is impossible if that cycle contains
        # at least one strict edge. Cycles consisting only of non-strict
        # edges can consistently describe equal timestamps.
        for origin in role_names:
            for target, strict in temporal_edges[origin]:
                if not strict:
                    continue
                pending = [target]
                visited = set()
                while pending:
                    current = pending.pop()
                    if current == origin:
                        raise ValueError(
                            "temporal constraints contain an impossible strict cycle"
                        )
                    if current in visited:
                        continue
                    visited.add(current)
                    pending.extend(
                        next_role
                        for next_role, _ in temporal_edges[current]
                    )

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported hypothesis pattern schema version")


@dataclass(frozen=True, slots=True)
class HypothesisGenerationRule:
    """Associate a hypothesis statement with a declarative event pattern."""

    rule_id: str
    pattern: HypothesisPattern
    statement: str
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rule_id", validate_text(self.rule_id, "rule_id")
        )
        if not isinstance(self.pattern, HypothesisPattern):
            raise TypeError("pattern must be a HypothesisPattern")
        object.__setattr__(
            self, "statement", validate_text(self.statement, "statement")
        )
        object.__setattr__(
            self, "description", validate_text(self.description, "description")
        )
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported hypothesis generation rule schema version"
            )

@dataclass(frozen=True, slots=True)
class HypothesisGeneration:
    """Record how one hypothesis was generated from observed evidence."""

    generation_id: str
    hypothesis_id: str
    rule_id: str
    correlation_ids: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation_id",
            validate_id(self.generation_id, "generation_id"),
        )
        object.__setattr__(
            self,
            "hypothesis_id",
            validate_id(self.hypothesis_id, "hypothesis_id"),
        )
        object.__setattr__(
            self,
            "rule_id",
            validate_text(self.rule_id, "rule_id"),
        )

        if not isinstance(self.correlation_ids, tuple):
            raise TypeError("correlation_ids must be a tuple")

        normalized_correlation_ids = tuple(
            validate_id(correlation_id, "correlation_id")
            for correlation_id in self.correlation_ids
        )

        if not normalized_correlation_ids:
            raise ValueError(
                "at least one correlation ID is required"
            )

        if (
            len(set(normalized_correlation_ids))
            != len(normalized_correlation_ids)
        ):
            raise ValueError("correlation IDs must be unique")

        object.__setattr__(
            self,
            "correlation_ids",
            normalized_correlation_ids,
        )

        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")

        if not self.evidence:
            raise ValueError("at least one evidence reference is required")

        if not all(
            isinstance(item, EvidenceReference)
            for item in self.evidence
        ):
            raise TypeError(
                "evidence must contain EvidenceReference objects"
            )

        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("evidence references must be unique")

        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, "created_at"),
        )

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported hypothesis generation schema version"
            )


@dataclass(frozen=True, slots=True)
class HypothesisGenerationResult:
    """Return generated hypotheses together with their provenance records."""

    hypotheses: tuple[Hypothesis, ...]
    generations: tuple[HypothesisGeneration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses, tuple):
            raise TypeError("hypotheses must be a tuple")

        if not all(
            isinstance(item, Hypothesis)
            for item in self.hypotheses
        ):
            raise TypeError("hypotheses must contain Hypothesis objects")

        if not isinstance(self.generations, tuple):
            raise TypeError("generations must be a tuple")

        if not all(
            isinstance(item, HypothesisGeneration)
            for item in self.generations
        ):
            raise TypeError(
                "generations must contain HypothesisGeneration objects"
            )

        hypothesis_ids = tuple(
            hypothesis.hypothesis_id
            for hypothesis in self.hypotheses
        )

        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypothesis IDs must be unique")

        generation_ids = tuple(
            generation.generation_id
            for generation in self.generations
        )

        if len(set(generation_ids)) != len(generation_ids):
            raise ValueError("generation IDs must be unique")

        generated_hypothesis_ids = tuple(
            generation.hypothesis_id
            for generation in self.generations
        )

        if len(set(generated_hypothesis_ids)) != len(
            generated_hypothesis_ids
        ):
            raise ValueError(
                "each hypothesis must have exactly one generation record"
            )

        if set(hypothesis_ids) != set(generated_hypothesis_ids):
            raise ValueError(
                "hypotheses and generations must reference each other "
                "exactly"
            )
