"""End-to-end deterministic hypothesis evaluation orchestration."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable

from samaritan.criterion_evaluation import CriterionEvaluationEngine
from samaritan.contracts import normalize_timestamp
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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.criterion_engine = CriterionEvaluationEngine(
            repository,
            evaluator_registry,
            outcome_registry,
            clock=self.clock,
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
        identity_payload = json.dumps(
            {
                "hypothesis_id": specification.hypothesis.hypothesis_id,
                "evaluation_rule_id": rule.evaluation_rule_id,
                "evaluation_rule_version": rule.rule_version,
                "evidence_set_id": evidence_set.evidence_set_id,
                "finding_ids": sorted(item.finding_id for item in findings),
                "outcome": synthesis.outcome,
                "confidence_ceiling": synthesis.confidence_ceiling,
                "alternative_standings": [
                    {
                        "alternative_id": item.alternative_id,
                        "standing": item.standing,
                        "supporting": sorted(item.supporting_finding_ids),
                        "opposing": sorted(item.opposing_finding_ids),
                        "neutral": sorted(item.non_discriminating_finding_ids),
                    }
                    for item in synthesis.alternative_standings
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        evaluation_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            "samaritan:hypothesis-evaluation:v1:" + identity_payload,
        ))
        assessment_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            "samaritan:assessment:v1:" + evaluation_id,
        ))
        record = HypothesisEvaluationRecord(
            evaluation_id=evaluation_id,
            hypothesis_id=specification.hypothesis.hypothesis_id,
            evaluation_rule_id=rule.evaluation_rule_id,
            evaluation_rule_version=rule.rule_version,
            evidence_set_id=evidence_set.evidence_set_id,
            assessment_id=assessment_id,
            evaluated_at=normalize_timestamp(self.clock(), "clock timestamp"),
            findings=findings,
            synthesis=synthesis,
        )
        record.validate_against(specification)
        return record
