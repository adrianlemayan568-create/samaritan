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
    HypothesisEvaluationRuleRegistry,
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
    """Evaluate registered propositions and preserve reproducible rule/evidence identity.

    When a custom proposition type is used, its exact semantics registry must be
    supplied explicitly; the default registry covers only built-in semantics.
    The optional clock controls evaluation timestamps, while record identities
    remain deterministic for identical analytical inputs.
    """

    def __init__(
        self,
        repository: Repository,
        evaluator_registry: CriterionEvaluatorRegistry,
        outcome_registry: CriterionOutcomeRegistry,
        rule_registry: HypothesisEvaluationRuleRegistry,
        semantics_registry: PropositionSemanticsRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if type(rule_registry) is not HypothesisEvaluationRuleRegistry:
            raise TypeError("rule_registry must be a HypothesisEvaluationRuleRegistry")
        if semantics_registry is None:
            semantics_registry = build_default_proposition_semantics_registry()
        if type(semantics_registry) is not PropositionSemanticsRegistry:
            raise TypeError("semantics_registry must be a PropositionSemanticsRegistry")
        self.clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        self.repository = repository
        self.criterion_engine = CriterionEvaluationEngine(
            repository,
            evaluator_registry,
            outcome_registry,
            clock=self.clock,
        )
        semantics_registry.freeze()
        self.semantics_registry = semantics_registry
        self.outcome_registry = outcome_registry
        rule_registry.freeze()
        self.rule_registry = rule_registry
        self.synthesizer = DiagnosticitySynthesizer()

    def evaluate(
        self,
        specification: HypothesisSpecification,
        evidence_set: EvidenceSet,
        *,
        production: bool = True,
    ) -> HypothesisEvaluationRecord:
        """Return an immutable evaluation record without persisting analytical state."""
        if type(specification) is not HypothesisSpecification:
            raise TypeError("specification must be a HypothesisSpecification")
        if type(evidence_set) is not EvidenceSet:
            raise TypeError("evidence_set must be an EvidenceSet")

        focal_proposition = specification.hypothesis.proposition
        if (
            evidence_set.evidence_set_id
            != focal_proposition.evidence_manifest.evidence_set_id
            or evidence_set.evidence != focal_proposition.evidence_manifest.evidence
        ):
            raise ValueError(
                "evaluation EvidenceSet must exactly match the focal proposition evidence manifest"
            )

        persisted_evidence_set = self.repository.get_evidence_set(
            focal_proposition.evidence_manifest.evidence_set_id
        )
        if (
            persisted_evidence_set is None
            or persisted_evidence_set.evidence_set_id != evidence_set.evidence_set_id
            or persisted_evidence_set.evidence != evidence_set.evidence
        ):
            raise ValueError(
                "evaluation EvidenceSet must match the persisted historical selection"
            )

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

        supplied_rule = specification.evaluation_rule
        try:
            registered_rule = self.rule_registry.get(
                supplied_rule.evaluation_rule_id, supplied_rule.rule_version
            )
        except KeyError as exc:
            raise ValueError(
                "evaluation rule version is not registered"
            ) from exc
        if supplied_rule != registered_rule:
            raise ValueError(
                "supplied evaluation rule does not match its registered immutable definition"
            )
        rule = registered_rule
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
                "focal_proposition_id": focal_proposition.proposition_id,
                "alternative_proposition_ids": sorted(
                    item.proposition.proposition_id
                    for item in specification.alternatives
                ),
                "evaluation_rule_id": rule.evaluation_rule_id,
                "evaluation_rule_version": rule.rule_version,
                "evidence_set_id": evidence_set.evidence_set_id,
                "finding_ids": sorted(item.finding_id for item in findings),
                "outcome": synthesis.outcome,
                "confidence_ceiling": synthesis.confidence_ceiling,
                "reason_codes": synthesis.reason_codes,
                "summary": synthesis.summary,
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
