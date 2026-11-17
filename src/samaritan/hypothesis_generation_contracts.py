"""Immutable contracts for evidence-driven hypothesis generation."""

from __future__ import annotations

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
class HypothesisGenerationRule:
    """Define an observed correlation pattern that can generate a hypothesis."""

    rule_id: str
    required_correlation_rule_ids: tuple[str, ...]
    minimum_correlations: int
    statement: str
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule_id",
            validate_text(self.rule_id, "rule_id"),
        )

        if not isinstance(self.required_correlation_rule_ids, tuple):
            raise TypeError(
                "required_correlation_rule_ids must be a tuple"
            )

        normalized_rule_ids = tuple(
            validate_text(rule_id, "correlation_rule_id")
            for rule_id in self.required_correlation_rule_ids
        )

        if not normalized_rule_ids:
            raise ValueError(
                "at least one correlation rule ID is required"
            )

        if len(set(normalized_rule_ids)) != len(normalized_rule_ids):
            raise ValueError(
                "required correlation rule IDs must be unique"
            )

        object.__setattr__(
            self,
            "required_correlation_rule_ids",
            normalized_rule_ids,
        )

        if type(self.minimum_correlations) is not int:
            raise TypeError("minimum_correlations must be an integer")

        if self.minimum_correlations < 1:
            raise ValueError("minimum_correlations must be positive")

        object.__setattr__(
            self,
            "statement",
            validate_text(self.statement, "statement"),
        )
        object.__setattr__(
            self,
            "description",
            validate_text(self.description, "description"),
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
