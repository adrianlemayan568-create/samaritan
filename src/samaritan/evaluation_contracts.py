
"""Immutable contracts for deterministic hypothesis evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from samaritan.analysis_contracts import (
    EVIDENCE_STANCES,
    EvidenceReference,
    Hypothesis,
)
from samaritan.contracts import (
    SCHEMA_VERSION,
    normalize_timestamp,
    validate_id,
    validate_text,
)


CRITERION_STATUSES = frozenset({
    "matched",
    "not_matched",
    "insufficient",
})


@dataclass(frozen=True, slots=True)
class Proposition:
    """Represent one atomic, machine-readable analytical claim."""

    proposition_type: str
    subject_entity_id: str
    object_entity_id: str | None = None
    target_event_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposition_type",
            validate_text(self.proposition_type, "proposition_type"),
        )
        object.__setattr__(
            self,
            "subject_entity_id",
            validate_id(self.subject_entity_id, "subject_entity_id"),
        )

        if self.object_entity_id is not None:
            object.__setattr__(
                self,
                "object_entity_id",
                validate_id(self.object_entity_id, "object_entity_id"),
            )

        if self.target_event_id is not None:
            object.__setattr__(
                self,
                "target_event_id",
                validate_id(self.target_event_id, "target_event_id"),
            )

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported proposition schema version")


@dataclass(frozen=True, slots=True)
class EvaluationCriterion:
    """Define one independently testable evaluation condition."""

    criterion_id: str
    criterion_type: str
    match_stance: str
    non_match_stance: str
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            validate_text(self.criterion_id, "criterion_id"),
        )
        object.__setattr__(
            self,
            "criterion_type",
            validate_text(self.criterion_type, "criterion_type"),
        )
        object.__setattr__(
            self,
            "match_stance",
            validate_text(self.match_stance, "match_stance"),
        )
        object.__setattr__(
            self,
            "non_match_stance",
            validate_text(self.non_match_stance, "non_match_stance"),
        )
        object.__setattr__(
            self,
            "description",
            validate_text(self.description, "description"),
        )

        if self.match_stance not in EVIDENCE_STANCES:
            raise ValueError("unsupported match stance")

        if self.non_match_stance not in EVIDENCE_STANCES:
            raise ValueError("unsupported non-match stance")

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported evaluation criterion schema version")


@dataclass(frozen=True, slots=True)
class HypothesisEvaluationRule:
    """Group independent criteria for evaluating a generated hypothesis."""

    rule_id: str
    generation_rule_id: str
    criteria: tuple[EvaluationCriterion, ...]
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule_id",
            validate_text(self.rule_id, "rule_id"),
        )
        object.__setattr__(
            self,
            "generation_rule_id",
            validate_text(self.generation_rule_id, "generation_rule_id"),
        )

        if not isinstance(self.criteria, tuple):
            raise TypeError("criteria must be a tuple")

        if not self.criteria:
            raise ValueError("criteria must not be empty")

        if not all(
            isinstance(item, EvaluationCriterion)
            for item in self.criteria
        ):
            raise TypeError(
                "criteria must contain EvaluationCriterion objects"
            )

        criterion_ids = [
            item.criterion_id for item in self.criteria
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("duplicate criterion IDs")

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
                "unsupported hypothesis evaluation rule schema version"
            )


@dataclass(frozen=True, slots=True)
class CriterionEvaluationResult:
    """Preserve the immutable result of evaluating one criterion."""

    criterion: EvaluationCriterion
    status: str
    stance: str | None
    evidence: tuple[EvidenceReference, ...]
    rationale: str
    missing_information: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, EvaluationCriterion):
            raise TypeError("criterion must be an EvaluationCriterion")

        object.__setattr__(
            self,
            "status",
            validate_text(self.status, "status"),
        )

        if self.status not in CRITERION_STATUSES:
            raise ValueError("unsupported criterion evaluation status")

        if self.status == "insufficient":
            if self.stance is not None:
                raise ValueError(
                    "insufficient results must not have a stance"
                )
        else:
            if self.stance is None:
                raise ValueError(
                    "matched and not_matched results require a stance"
                )

            object.__setattr__(
                self,
                "stance",
                validate_text(self.stance, "stance"),
            )

            expected_stance = (
                self.criterion.match_stance
                if self.status == "matched"
                else self.criterion.non_match_stance
            )

            if self.stance != expected_stance:
                raise ValueError(
                    "stance does not match the criterion definition"
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

        evidence_keys = [
            (item.observation_id, item.event_id)
            for item in self.evidence
        ]

        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("duplicate evidence references")

        if self.status != "insufficient" and not self.evidence:
            raise ValueError(
                "evaluated results must reference at least one piece of evidence"
            )

        object.__setattr__(
            self,
            "rationale",
            validate_text(self.rationale, "rationale"),
        )

        if not isinstance(self.missing_information, tuple):
            raise TypeError("missing_information must be a tuple")

        for item in self.missing_information:
            validate_text(item, "missing_information item")

        if self.status == "insufficient":
            if not self.missing_information:
                raise ValueError(
                    "insufficient results must specify missing information"
                )
        elif self.missing_information:
            raise ValueError(
                "evaluated results must not specify missing information"
            )

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported criterion evaluation result schema version"
            )



@dataclass(frozen=True, slots=True)
class HypothesisAlternative:
    """Represent one machine-readable competing explanation."""

    alternative_id: str
    proposition: Proposition
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alternative_id",
            validate_id(self.alternative_id, "alternative_id"),
        )

        if not isinstance(self.proposition, Proposition):
            raise TypeError("proposition must be a Proposition")

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
                "unsupported hypothesis alternative schema version"
            )


@dataclass(frozen=True, slots=True)
class HypothesisSpecification:
    """Define a hypothesis and its deterministic evaluation framework."""

    hypothesis: Hypothesis
    evaluation_rule: HypothesisEvaluationRule
    alternatives: tuple[HypothesisAlternative, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis")

        if not isinstance(
            self.evaluation_rule,
            HypothesisEvaluationRule,
        ):
            raise TypeError(
                "evaluation_rule must be a HypothesisEvaluationRule"
            )

        if not isinstance(self.alternatives, tuple):
            raise TypeError("alternatives must be a tuple")

        if not self.alternatives:
            raise ValueError("alternatives must not be empty")

        if not all(
            isinstance(item, HypothesisAlternative)
            for item in self.alternatives
        ):
            raise TypeError(
                "alternatives must contain HypothesisAlternative objects"
            )

        alternative_ids = [
            item.alternative_id for item in self.alternatives
        ]

        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("duplicate alternative IDs")

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported hypothesis specification schema version"
            )


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Preserve one immutable selection of evidence for evaluation."""

    evidence_set_id: str
    evidence: tuple[EvidenceReference, ...]
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_set_id",
            validate_id(self.evidence_set_id, "evidence_set_id"),
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

        evidence_keys = [
            (item.observation_id, item.event_id)
            for item in self.evidence
        ]

        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("duplicate evidence references")

        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, "created_at"),
        )

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported evidence set schema version")


@dataclass(frozen=True, slots=True)
class HypothesisEvaluation:
    """Preserve the immutable record of one completed hypothesis evaluation."""

    evaluation_id: str
    hypothesis_id: str
    evaluation_rule_id: str
    evidence_set_id: str
    assessment_id: str
    evaluated_at: datetime
    criterion_results: tuple[CriterionEvaluationResult, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_rule_id",
            validate_text(self.evaluation_rule_id, "evaluation_rule_id"),
        )

        for field_name in (
            "evaluation_id",
            "hypothesis_id",
            "evidence_set_id",
            "assessment_id",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_id(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "evaluated_at",
            normalize_timestamp(self.evaluated_at, "evaluated_at"),
        )

        if not isinstance(self.criterion_results, tuple):
            raise TypeError("criterion_results must be a tuple")

        if not self.criterion_results:
            raise ValueError("criterion_results must not be empty")

        if not all(
            isinstance(item, CriterionEvaluationResult)
            for item in self.criterion_results
        ):
            raise TypeError(
                "criterion_results must contain "
                "CriterionEvaluationResult objects"
            )

        criterion_ids = [
            item.criterion.criterion_id
            for item in self.criterion_results
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("duplicate criterion results")

        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported hypothesis evaluation schema version"
            )
