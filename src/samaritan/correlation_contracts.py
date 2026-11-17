"""Immutable contracts for evidence correlation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from samaritan.contracts import (
    SCHEMA_VERSION,
    normalize_timestamp,
    validate_id,
    validate_text,
)


@dataclass(frozen=True, slots=True)
class CorrelationRule:
    """Define the conditions under which two events may be correlated."""

    rule_id: str
    event_type_pairs: tuple[tuple[str, str], ...]
    shared_entity_types: tuple[str, ...]
    max_time_delta_seconds: int | float
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rule_id", validate_text(self.rule_id, "rule_id")
        )
        object.__setattr__(
            self, "description",
            validate_text(self.description, "description"),
        )

        if not isinstance(self.event_type_pairs, tuple):
            raise TypeError("event_type_pairs must be a tuple")

        if not self.event_type_pairs:
            raise ValueError("event_type_pairs cannot be empty")

        normalized_pairs = []

        for index, pair in enumerate(self.event_type_pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    f"event_type_pairs[{index}] must be a tuple of two strings"
                )

            first = validate_text(pair[0], f"event_type_pairs[{index}][0]")
            second = validate_text(pair[1], f"event_type_pairs[{index}][1]")

            normalized_pairs.append((first, second))

        canonical_pairs = {
            frozenset(pair) for pair in normalized_pairs
        }

        if len(canonical_pairs) != len(normalized_pairs):
            raise ValueError("duplicate event type pairs")

        object.__setattr__(
            self, "event_type_pairs", tuple(normalized_pairs)
        )

        if not isinstance(self.shared_entity_types, tuple):
            raise TypeError("shared_entity_types must be a tuple")

        if not self.shared_entity_types:
            raise ValueError("shared_entity_types cannot be empty")

        entity_types = tuple(
            validate_text(item, "shared_entity_type")
            for item in self.shared_entity_types
        )

        if len(entity_types) != len(set(entity_types)):
            raise ValueError("duplicate shared entity types")

        object.__setattr__(
            self, "shared_entity_types", entity_types
        )

        delta = self.max_time_delta_seconds

        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise TypeError("max_time_delta_seconds must be numeric")

        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(
                "max_time_delta_seconds must be positive and finite"
            )

        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported correlation rule schema version")


@dataclass(frozen=True, slots=True)
class Correlation:
    """Record why two events were correlated."""

    correlation_id: str
    event_ids: tuple[str, str]
    observation_ids: tuple[str, str]
    shared_entity_ids: tuple[str, ...]
    matched_rules: tuple[CorrelationRule, ...]
    time_delta_seconds: int | float
    explanation: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correlation_id",
            validate_id(self.correlation_id, "correlation_id"),
        )

        if not isinstance(self.event_ids, tuple) or len(self.event_ids) != 2:
            raise TypeError("event_ids must be a tuple containing two IDs")

        event_ids = tuple(
            validate_id(item, "event_id") for item in self.event_ids
        )

        if event_ids[0] == event_ids[1]:
            raise ValueError("a correlation requires two distinct events")

        object.__setattr__(self, "event_ids", event_ids)

        if (
            not isinstance(self.observation_ids, tuple)
            or len(self.observation_ids) != 2
        ):
            raise TypeError(
                "observation_ids must be a tuple containing two IDs"
            )

        observation_ids = tuple(
            validate_id(item, "observation_id")
            for item in self.observation_ids
        )
        object.__setattr__(self, "observation_ids", observation_ids)

        if not isinstance(self.shared_entity_ids, tuple):
            raise TypeError("shared_entity_ids must be a tuple")

        if not self.shared_entity_ids:
            raise ValueError("shared_entity_ids cannot be empty")

        entity_ids = tuple(
            validate_id(item, "shared_entity_id")
            for item in self.shared_entity_ids
        )

        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate shared entity IDs")

        object.__setattr__(self, "shared_entity_ids", entity_ids)

        if not isinstance(self.matched_rules, tuple):
            raise TypeError("matched_rules must be a tuple")

        if not self.matched_rules:
            raise ValueError("matched_rules cannot be empty")

        if not all(
            isinstance(rule, CorrelationRule)
            for rule in self.matched_rules
        ):
            raise TypeError(
                "matched_rules must contain CorrelationRule objects"
            )

        rule_ids = [rule.rule_id for rule in self.matched_rules]

        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("duplicate matched rule IDs")

        delta = self.time_delta_seconds

        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise TypeError("time_delta_seconds must be numeric")

        if not math.isfinite(delta) or delta < 0:
            raise ValueError(
                "time_delta_seconds must be non-negative and finite"
            )

        for rule in self.matched_rules:
            if delta > rule.max_time_delta_seconds:
                raise ValueError(
                    "time_delta_seconds exceeds a matched rule's "
                    "maximum time delta"
                )

        object.__setattr__(
            self,
            "explanation",
            validate_text(self.explanation, "explanation"),
        )
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, "created_at"),
        )

        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported correlation schema version")
