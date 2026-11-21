"""End-to-end deterministic hypothesis evaluation orchestration."""

from __future__ import annotations

from datetime import datetime, timezone

from samaritan.criterion_evaluation import CriterionEvaluationEngine
from samaritan.contracts import new_id
from samaritan.diagnosticity import DiagnosticitySynthesizer
from samaritan.evaluation_contracts import EvidenceSet
from samaritan.reasoning_contracts import (
    CriterionEvaluatorRegistry,
    CriterionOutcomeRegistry,
    HypothesisEvaluationRecord,
    HypothesisSpecification,
)
from samaritan.repository import Repository
from samaritan.proposition_contracts import (
    CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
)
from samaritan.proposition_semantics import (
    PropositionSemanticsRegistry,
    build_default_proposition_semantics_registry,
)


class HypothesisEvaluationEngine:
    """Evaluate every declared criterion, synthesize alternatives, and preserve the exact rule identity."""

    def __init__(
        self,
        repository: Repository,
        evaluator_registry: CriterionEvaluatorRegistry,
        outcome_registry: CriterionOutcomeRegistry,
        semantics_registry: PropositionSemanticsRegistry | None = None,
    ) -> None:
        self.criterion_engine = CriterionEvaluationEngine(
            repository,
            evaluator_registry,
            outcome_registry,
        )
        if semantics_registry is None:
            semantics_registry = build_default_proposition_semantics_registry()
        if not isinstance(semantics_registry, PropositionSemanticsRegistry):
            raise TypeError("semantics_registry must be a PropositionSemanticsRegistry")
        self.semantics_registry = semantics_registry
        self.outcome_registry = outcome_registry
        self.synthesizer = DiagnosticitySynthesizer()

    def evaluate(
        self,
        specification: HypothesisSpecification,
        evidence_set: EvidenceSet,
        *,
        production: bool = True,
    ) -> HypothesisEvaluationRecord:
        """Return an immutable evaluation record without persisting analytical state."""
        if not isinstance(specification, HypothesisSpecification):
            raise TypeError("specification must be a HypothesisSpecification")
        if not isinstance(evidence_set, EvidenceSet):
            raise TypeError("evidence_set must be an EvidenceSet")

        # A fingerprint supplied by a Proposition is only a claim until
        # checked against the authoritative semantics registry. Evaluation
        # accepts only the current fingerprint-bound identity algorithm.
        propositions = (
            specification.hypothesis.proposition,
            *(item.proposition for item in specification.alternatives),
        )
        for proposition in propositions:
            if proposition.identity_algorithm_version != CURRENT_PROPOSITION_IDENTITY_ALGORITHM:
                raise ValueError(
                    "hypothesis evaluation requires the current fingerprint-bound proposition identity"
                )
            semantics = self.semantics_registry.get(
                proposition.proposition_type, proposition.semantics_version
            )
            proposition.validate_against(semantics)

        rule = specification.evaluation_rule
        rule.validate_outcome_coverage(
            self.criterion_engine.evaluator_registry,
            self.outcome_registry,
            tuple(item.alternative_id for item in specification.alternatives),
        )
        rule.validate_for(
            specification.hypothesis.proposition,
            production=production,
        )
        findings = tuple(
            self.criterion_engine.evaluate(
                criterion,
                specification.hypothesis.proposition,
                evidence_set,
            )
            for criterion in rule.criterion_definitions
        )
        synthesis = self.synthesizer.synthesize(
            specification,
            findings,
            self.outcome_registry,
            production=production,
        )
        record = HypothesisEvaluationRecord(
            evaluation_id=new_id(),
            hypothesis_id=specification.hypothesis.hypothesis_id,
            evaluation_rule_id=rule.evaluation_rule_id,
            evaluation_rule_version=rule.rule_version,
            evidence_set_id=evidence_set.evidence_set_id,
            assessment_id=new_id(),
            evaluated_at=datetime.now(timezone.utc),
            findings=findings,
            synthesis=synthesis,
        )
        record.validate_against(specification)
        return record
