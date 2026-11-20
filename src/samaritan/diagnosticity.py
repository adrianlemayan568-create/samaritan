"""Deterministic, ordinal synthesis across declared alternatives."""

from __future__ import annotations

from collections.abc import Iterable

from samaritan.reasoning_contracts import (
    AlternativeStanding,
    CriterionFinding,
    CriterionOutcomeRegistry,
    DiagnosticitySynthesis,
    HypothesisSpecification,
    RESIDUAL_ALTERNATIVE_ID,
)


_JUDGMENT_RANK = {
    "incompatible": 0,
    "unexpected": 1,
    "compatible": 2,
    "expected": 3,
}


class DiagnosticitySynthesizer:
    """Synthesize criterion findings without scores, probabilities or hidden weights."""

    def synthesize(
        self,
        specification: HypothesisSpecification,
        findings: Iterable[CriterionFinding],
        outcome_registry: CriterionOutcomeRegistry,
        *,
        production: bool = True,
    ) -> DiagnosticitySynthesis:
        if not isinstance(specification, HypothesisSpecification):
            raise TypeError("specification must be a HypothesisSpecification")
        if not isinstance(outcome_registry, CriterionOutcomeRegistry):
            raise TypeError("outcome_registry must be a CriterionOutcomeRegistry")
        if isinstance(findings, (str, bytes)):
            raise TypeError("findings must be an iterable of CriterionFinding objects")
        try:
            supplied = tuple(findings)
        except TypeError as exc:
            raise TypeError("findings must be iterable") from exc
        if not all(isinstance(item, CriterionFinding) for item in supplied):
            raise TypeError("findings must contain CriterionFinding objects")

        rule = specification.evaluation_rule
        rule.validate_for(specification.hypothesis.proposition, production=production)
        by_criterion = {item.criterion_id: item for item in rule.criterion_definitions}
        findings_by_criterion: dict[str, CriterionFinding] = {}
        for finding in supplied:
            if finding.criterion_id not in by_criterion:
                raise ValueError(f"finding references an undeclared criterion: {finding.criterion_id}")
            if finding.criterion_id in findings_by_criterion:
                raise ValueError(f"duplicate finding for criterion: {finding.criterion_id}")
            finding.validate_against(by_criterion[finding.criterion_id], outcome_registry)
            findings_by_criterion[finding.criterion_id] = finding

        alternatives = {item.alternative_id: item for item in specification.alternatives}
        expected_alternative_ids = set(alternatives) | {RESIDUAL_ALTERNATIVE_ID}
        bearing_index = {
            (item.criterion_id, item.outcome_code, item.alternative_id): item
            for item in rule.bearing_definitions
        }

        directional: dict[str, dict[str, list[str]]] = {
            alternative_id: {
                "support": [],
                "oppose": [],
                "neutral": [],
            }
            for alternative_id in alternatives
        }
        reason_codes: list[str] = []
        required_incomplete = False
        complete_findings = 0
        focal_refuted = False
        residual_discriminator = False

        for criterion in rule.criterion_definitions:
            finding = findings_by_criterion.get(criterion.criterion_id)
            if finding is None:
                reason_codes.append(f"missing_finding:{criterion.criterion_id}")
                if any(
                    bearing.criterion_id == criterion.criterion_id
                    and bearing.evaluability_role == "required"
                    for bearing in rule.bearing_definitions
                ):
                    required_incomplete = True
                continue
            if finding.status == "insufficient":
                reason_codes.append(f"insufficient_finding:{criterion.criterion_id}")
                if any(
                    bearing.criterion_id == criterion.criterion_id
                    and bearing.evaluability_role == "required"
                    for bearing in rule.bearing_definitions
                ):
                    required_incomplete = True
                continue

            complete_findings += 1
            outcome_registry.get(criterion.criterion_type, criterion.outcome_vocabulary_version, finding.outcome_code)
            for alternative_id in expected_alternative_ids:
                bearing = bearing_index.get((
                    criterion.criterion_id,
                    finding.outcome_code,
                    alternative_id,
                ))
                if bearing is None:
                    raise ValueError(
                        "missing bearing definition for "
                        f"{criterion.criterion_id}:{finding.outcome_code}:{alternative_id}"
                    )
                if production and bearing.review_status != "approved":
                    raise ValueError("production synthesis requires approved bearing definitions")
                focal_rank = _JUDGMENT_RANK[bearing.focal_judgment]
                alternative_rank = _JUDGMENT_RANK[bearing.alternative_judgment]
                if alternative_id == RESIDUAL_ALTERNATIVE_ID:
                    if bearing.alternative_judgment != "compatible":
                        raise ValueError("residual compatibility is immutable")
                    if focal_rank > alternative_rank:
                        residual_discriminator = True
                    continue

                if focal_rank > alternative_rank:
                    directional[alternative_id]["support"].append(finding.finding_id)
                elif focal_rank < alternative_rank:
                    directional[alternative_id]["oppose"].append(finding.finding_id)
                else:
                    directional[alternative_id]["neutral"].append(finding.finding_id)

                if (
                    bearing.focal_judgment == "incompatible"
                    and not criterion.attacker_influenceable
                ):
                    focal_refuted = True

        standings = []
        for alternative_id in sorted(alternatives):
            values = directional[alternative_id]
            if values["support"] and values["oppose"]:
                standing = "conflicting"
            elif values["support"]:
                standing = "favors_focal"
            elif values["oppose"]:
                standing = "favors_alternative"
            else:
                standing = "non_discriminating"
            standings.append(
                AlternativeStanding(
                    alternative_id=alternative_id,
                    standing=standing,
                    supporting_finding_ids=tuple(values["support"]),
                    opposing_finding_ids=tuple(values["oppose"]),
                    non_discriminating_finding_ids=tuple(values["neutral"]),
                )
            )

        if required_incomplete:
            reason_codes.append("required_criteria_incomplete")

        if complete_findings == 0:
            reason_codes.append("no_complete_findings")
            outcome = "inconclusive"
        elif focal_refuted:
            reason_codes.append("complete_non_influenceable_focal_contradiction")
            outcome = "not_supported"
        elif required_incomplete:
            outcome = "inconclusive"
        elif all(item.standing == "favors_focal" for item in standings):
            outcome = "supported"
            reason_codes.append("focal_favored_over_every_declared_alternative")
        else:
            outcome = "inconclusive"
            reason_codes.append("alternatives_not_eliminated")

        confidence_ceiling = "moderate" if residual_discriminator else "low"
        if not residual_discriminator:
            reason_codes.append("no_outcome_discriminates_focal_from_residual")

        summary = {
            "supported": (
                "The focal proposition is favored over every declared alternative "
                "by the registered ordinal findings; this is not proof of truth."
            ),
            "not_supported": (
                "At least one complete, non-attacker-influenceable finding is "
                "incompatible with the focal proposition."
            ),
            "inconclusive": (
                "The available findings do not eliminate every declared alternative "
                "or required evaluation information is incomplete."
            ),
        }[outcome]
        return DiagnosticitySynthesis(
            outcome=outcome,
            confidence_ceiling=confidence_ceiling,
            alternative_standings=tuple(standings),
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            summary=summary,
        )
