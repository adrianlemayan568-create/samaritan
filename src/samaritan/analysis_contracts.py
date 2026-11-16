
"""Immutable contracts for evidence-based intelligence analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from samaritan.contracts import (
    SCHEMA_VERSION,
    normalize_timestamp,
    validate_id,
    validate_text,
)


CONFIDENCE_LEVELS = frozenset({
    "low",
    "moderate",
    "high",
})

EVIDENCE_STANCES = frozenset({
    "supports",
    "contradicts",
    "neutral",
})

ASSESSMENT_OUTCOMES = frozenset({
    "supported",
    "inconclusive",
    "not_supported",
})


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Reference an existing observation or event."""

    observation_id: str | None = None
    event_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (self.observation_id is None) == (self.event_id is None):
            raise ValueError(
                "exactly one of observation_id or event_id must be provided"
            )

        if self.observation_id is not None:
            object.__setattr__(
                self,
                "observation_id",
                validate_id(self.observation_id, "observation_id"),
            )

        if self.event_id is not None:
            object.__setattr__(
                self,
                "event_id",
                validate_id(self.event_id, "event_id"),
            )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported evidence reference schema version")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A proposed explanation that requires evidential evaluation."""

    hypothesis_id: str
    statement: str
    description: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hypothesis_id",
            validate_id(self.hypothesis_id, "hypothesis_id"),
        )
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
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, "created_at"),
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported hypothesis schema version")


@dataclass(frozen=True, slots=True)
class EvidenceEvaluation:
    """Record how a specific piece of evidence relates to a hypothesis."""

    evaluation_id: str
    evidence: EvidenceReference
    stance: str
    rationale: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_id",
            validate_id(self.evaluation_id, "evaluation_id"),
        )

        if not isinstance(self.evidence, EvidenceReference):
            raise TypeError("evidence must be an EvidenceReference")

        object.__setattr__(
            self,
            "stance",
            validate_text(self.stance, "stance"),
        )

        if self.stance not in EVIDENCE_STANCES:
            raise ValueError("unsupported evidence stance")

        object.__setattr__(
            self,
            "rationale",
            validate_text(self.rationale, "rationale"),
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported evidence evaluation schema version")


@dataclass(frozen=True, slots=True)
class AlternativeExplanation:
    """A competing explanation for the observed evidence."""

    explanation_id: str
    statement: str
    rationale: str
    evidence: tuple[EvidenceReference, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "explanation_id",
            validate_id(self.explanation_id, "explanation_id"),
        )
        object.__setattr__(
            self,
            "statement",
            validate_text(self.statement, "statement"),
        )
        object.__setattr__(
            self,
            "rationale",
            validate_text(self.rationale, "rationale"),
        )

        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")

        if not all(
            isinstance(item, EvidenceReference)
            for item in self.evidence
        ):
            raise TypeError(
                "evidence must contain EvidenceReference objects"
            )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "unsupported alternative explanation schema version"
            )


@dataclass(frozen=True, slots=True)
class Assessment:
    """An evidence-backed evaluation of a specific hypothesis."""

    assessment_id: str
    hypothesis_id: str
    outcome: str
    confidence: str
    summary: str
    assessed_at: datetime
    evaluations: tuple[EvidenceEvaluation, ...] = ()
    alternatives: tuple[AlternativeExplanation, ...] = ()
    missing_information: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessment_id",
            validate_id(self.assessment_id, "assessment_id"),
        )
        object.__setattr__(
            self,
            "hypothesis_id",
            validate_id(self.hypothesis_id, "hypothesis_id"),
        )

        object.__setattr__(
            self,
            "outcome",
            validate_text(self.outcome, "outcome"),
        )

        if self.outcome not in ASSESSMENT_OUTCOMES:
            raise ValueError("unsupported assessment outcome")

        object.__setattr__(
            self,
            "confidence",
            validate_text(self.confidence, "confidence"),
        )

        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError("unsupported confidence level")

        object.__setattr__(
            self,
            "summary",
            validate_text(self.summary, "summary"),
        )
        object.__setattr__(
            self,
            "assessed_at",
            normalize_timestamp(self.assessed_at, "assessed_at"),
        )

        if not isinstance(self.evaluations, tuple):
            raise TypeError("evaluations must be a tuple")

        if not all(
            isinstance(item, EvidenceEvaluation)
            for item in self.evaluations
        ):
            raise TypeError(
                "evaluations must contain EvidenceEvaluation objects"
            )

        evaluation_ids = [
            item.evaluation_id for item in self.evaluations
        ]

        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("duplicate evaluation IDs")

        if not isinstance(self.alternatives, tuple):
            raise TypeError("alternatives must be a tuple")

        if not all(
            isinstance(item, AlternativeExplanation)
            for item in self.alternatives
        ):
            raise TypeError(
                "alternatives must contain AlternativeExplanation objects"
            )

        alternative_ids = [
            item.explanation_id for item in self.alternatives
        ]

        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("duplicate alternative explanation IDs")

        if not isinstance(self.missing_information, tuple):
            raise TypeError("missing_information must be a tuple")

        for item in self.missing_information:
            validate_text(item, "missing_information item")

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported assessment schema version")
