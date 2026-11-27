from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from samaritan.analysis_contracts import EvidenceReference
from samaritan.diagnosticity import DiagnosticitySynthesizer
from samaritan.proposition_contracts import (
    Proposition,
    PropositionEvidenceManifest,
    PropositionFieldProvenance,
    PropositionRoleValue,
    PropositionTemporalScope,
    StructuredHypothesis,
    deterministic_proposition_id,
)
from samaritan.proposition_semantics import build_default_proposition_semantics_registry
from samaritan.proposition_semantics_fingerprint import proposition_semantics_fingerprint
from samaritan.reasoning_contracts import (
    AlternativeEvaluation,
    BearingDefinition,
    CriterionDefinition,
    CriterionFinding,
    CriterionOutcomeDefinition,
    CriterionOutcomeRegistry,
    CriterionParameterDefinition,
    HypothesisAlternative,
    HypothesisEvaluationRule,
    HypothesisEvaluationRuleRegistry,
    HypothesisEvaluationRecord,
    HypothesisSpecification,
    RESIDUAL_ALTERNATIVE_ID,
)


def make_proposition(account_id=None):
    account_id = account_id or str(uuid4())
    timestamp = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    event_ids = tuple(str(uuid4()) for _ in range(3))
    times = (timestamp, timestamp.replace(minute=1), timestamp.replace(minute=2))
    roles = (
        PropositionRoleValue("account", "entity", account_id),
        PropositionRoleValue("failure_event", "event", event_ids[0], times[0]),
        PropositionRoleValue("success_event", "event", event_ids[1], times[1]),
        PropositionRoleValue("host_activity_event", "event", event_ids[2], times[2]),
    )
    scope = PropositionTemporalScope(times[0], times[2])
    provenance = tuple(
        PropositionFieldProvenance(
            role.role_name,
            (role.value_id,) if role.role_kind == "event" else event_ids,
            tuple(str(uuid4()) for _ in range(1 if role.role_kind == "event" else 3)),
        )
        for role in roles
    )
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    semantics_fingerprint = proposition_semantics_fingerprint(semantics)
    identity_algorithm_version = "uuid5-v2"
    return Proposition(
        proposition_id=deterministic_proposition_id(
            "account_authentication_sequence", 1, roles, scope,
            semantics_fingerprint=semantics_fingerprint,
            identity_algorithm_version=identity_algorithm_version,
        ),
        proposition_type="account_authentication_sequence",
        semantics_version=1,
        roles=roles,
        temporal_scope=scope,
        field_provenance=provenance,
        construction_id="account_authentication_sequence",
        construction_version=1,
        semantics_fingerprint=semantics_fingerprint,
        identity_algorithm_version=identity_algorithm_version,
        evidence_manifest=PropositionEvidenceManifest(
            evidence_set_id=str(uuid4()),
            evidence=tuple(EvidenceReference(event_id=event_id) for event_id in event_ids),
        ),
    )


def make_specification(focal=None, outcome="observed"):
    focal = focal or make_proposition()
    alternative = HypothesisAlternative(
        alternative_id=str(uuid4()),
        proposition=make_proposition(str(uuid4())),
        description="The account activity was authorized.",
    )
    criterion = CriterionDefinition(
        criterion_id="independent-confirmation",
        criterion_type="out_of_band_confirmation",
        parameters_json="{}",
        description="The account holder independently confirms the activity.",
        independence_assertion_id="account-holder-confirmation",
    )
    bearings = (
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code=outcome,
            alternative_id=alternative.alternative_id,
            focal_judgment="expected",
            alternative_judgment="unexpected",
            evaluability_role="required",
            rationale="The confirmation is expected under the focal proposition.",
            basis="Independent account-holder confirmation.",
            author="analyst",
            review_status="approved",
        ),
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code=outcome,
            alternative_id=RESIDUAL_ALTERNATIVE_ID,
            focal_judgment="expected",
            alternative_judgment="compatible",
            evaluability_role="contextual",
            rationale="The finding is more expected under the focal proposition.",
            basis="Residual remains broadly compatible.",
            author="analyst",
            review_status="approved",
        ),
    )
    rule = HypothesisEvaluationRule(
        evaluation_rule_id="account-authentication-evaluation-v1",
        rule_version=1,
        proposition_type=focal.proposition_type,
        semantics_version=focal.semantics_version,
        semantics_fingerprint=focal.semantics_fingerprint,
        criterion_definitions=(criterion,),
        bearing_definitions=bearings,
        description="Evaluate account authentication sequence hypotheses.",
        review_status="approved",
    )
    return HypothesisSpecification(
        hypothesis=StructuredHypothesis(
            hypothesis_id=str(uuid4()),
            proposition=focal,
            created_at=datetime.now(timezone.utc),
        ),
        alternatives=(alternative,),
        evaluation_rule=rule,
    ), criterion, alternative

def make_alternative_evaluation(specification, criterion):
    alternative = specification.alternatives[0]
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="insufficient",
        outcome_code=None,
        evidence=(),
        rationale="Alternative-specific evidence is insufficient for this test fixture.",
        missing_information=("Complete alternative-specific evidence",),
    )
    return AlternativeEvaluation(
        alternative_id=alternative.alternative_id,
        proposition_id=alternative.proposition.proposition_id,
        evidence_set_id=alternative.proposition.evidence_manifest.evidence_set_id,
        findings=(finding,),
    )


def test_insufficient_finding_has_no_outcome_code_or_bearing():
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id="inventory",
        status="insufficient",
        outcome_code=None,
        evidence=(),
        rationale="Inventory coverage is incomplete.",
        missing_information=("Complete device inventory",),
    )
    assert finding.status == "insufficient"
    assert finding.outcome_code is None


def test_finding_rejects_stance_field_by_design():
    with pytest.raises(TypeError):
        CriterionFinding(
            finding_id=str(uuid4()),
            criterion_id="inventory",
            status="matched",
            outcome_code="present",
            evidence=(EvidenceReference(event_id=str(uuid4())),),
            rationale="A factual finding.",
            stance="supports",
        )


def test_supported_synthesis_is_ordinal_and_residual_does_not_block():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        vocabulary_version=1,
        meaning="Independent account-holder confirmation was obtained.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="The account holder independently confirmed the activity.",
    )
    result = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes
    )
    assert result.outcome == "supported"
    assert result.confidence_ceiling == "moderate"
    assert result.alternative_standings[0].standing == "favors_focal"


def test_missing_required_finding_is_inconclusive():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent confirmation.",
    ))
    result = DiagnosticitySynthesizer().synthesize(
        specification, (), outcomes
    )
    assert result.outcome == "inconclusive"
    assert "required_criteria_incomplete" in result.reason_codes


def test_unapproved_rule_cannot_be_used_in_production():
    specification, criterion, alternative = make_specification()
    rule = specification.evaluation_rule
    unapproved = HypothesisEvaluationRule(
        evaluation_rule_id=rule.evaluation_rule_id,
        rule_version=rule.rule_version,
        proposition_type=rule.proposition_type,
        semantics_version=rule.semantics_version,
        semantics_fingerprint=rule.semantics_fingerprint,
        criterion_definitions=rule.criterion_definitions,
        bearing_definitions=tuple(
            BearingDefinition(
                criterion_id=item.criterion_id,
                outcome_code=item.outcome_code,
                alternative_id=item.alternative_id,
                focal_judgment=item.focal_judgment,
                alternative_judgment=item.alternative_judgment,
                evaluability_role=item.evaluability_role,
                rationale=item.rationale,
                basis=item.basis,
                author=item.author,
                review_status="reviewed",
            )
            for item in rule.bearing_definitions
        ),
        description=rule.description,
        review_status="reviewed",
    )
    with pytest.raises(ValueError, match="approved"):
        unapproved.validate_for(specification.hypothesis.proposition, production=True)


def test_criterion_parameters_are_typed_and_defaults_are_materialized():
    criterion = CriterionDefinition(
        criterion_id="threshold-check",
        criterion_type="threshold_check",
        parameters_json='{"minimum": 3}',
        description="Check a typed threshold.",
        independence_assertion_id="threshold-source",
        parameter_definitions=(
            CriterionParameterDefinition("minimum", "integer", required=True),
            CriterionParameterDefinition("enabled", "boolean", default_value=True),
        ),
    )
    assert criterion.parameters_json == '{"enabled":true,"minimum":3}'


def test_criterion_parameter_type_mismatch_is_rejected():
    with pytest.raises(ValueError, match="type mismatch"):
        CriterionDefinition(
            criterion_id="threshold-check",
            criterion_type="threshold_check",
            parameters_json='{"minimum": true}',
            description="Check a typed threshold.",
            independence_assertion_id="threshold-source",
            parameter_definitions=(
                CriterionParameterDefinition("minimum", "integer", required=True),
            ),
        )


def test_outcome_vocabularies_are_versioned_and_immutable():
    registry = CriterionOutcomeRegistry()
    v1 = CriterionOutcomeDefinition("auth_context", "present", "Context exists.", vocabulary_version=1)
    v2 = CriterionOutcomeDefinition("auth_context", "present", "Context exists under revised semantics.", vocabulary_version=2)
    registry.register(v1)
    registry.register(v2)
    assert registry.get("auth_context", 1, "present") is v1
    assert registry.get("auth_context", 2, "present") is v2
    with pytest.raises(ValueError, match="already registered"):
        registry.register(v1)


def test_evaluation_rule_registry_preserves_exact_immutable_version():
    specification, _, _ = make_specification()
    registry = HypothesisEvaluationRuleRegistry()
    registry.register(specification.evaluation_rule)
    assert registry.get(
        specification.evaluation_rule.evaluation_rule_id,
        specification.evaluation_rule.rule_version,
    ) is specification.evaluation_rule
    with pytest.raises(ValueError, match="already registered"):
        registry.register(specification.evaluation_rule)



def test_complete_non_influenceable_contradiction_produces_not_supported():
    specification, criterion, alternative = make_specification()
    original = specification.evaluation_rule
    bearings = tuple(
        BearingDefinition(
            criterion_id=item.criterion_id,
            outcome_code=item.outcome_code,
            alternative_id=item.alternative_id,
            focal_judgment="incompatible",
            alternative_judgment=(
                "compatible"
                if item.alternative_id == RESIDUAL_ALTERNATIVE_ID
                else "expected"
            ),
            evaluability_role=item.evaluability_role,
            rationale="The complete independent evidence contradicts the focal claim.",
            basis="Validated independent source.",
            author="analyst",
            review_status="approved",
        )
        for item in original.bearing_definitions
    )
    rule = HypothesisEvaluationRule(
        evaluation_rule_id=original.evaluation_rule_id,
        rule_version=original.rule_version,
        proposition_type=original.proposition_type,
        semantics_version=original.semantics_version,
        semantics_fingerprint=original.semantics_fingerprint,
        criterion_definitions=original.criterion_definitions,
        bearing_definitions=bearings,
        description=original.description,
        review_status="approved",
    )
    specification = HypothesisSpecification(
        hypothesis=specification.hypothesis,
        alternatives=specification.alternatives,
        evaluation_rule=rule,
    )
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="Complete independent evidence contradicts the focal proposition.",
    )
    result = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes
    )
    assert result.outcome == "not_supported"


def test_hypothesis_evaluation_record_preserves_exact_rule_identity():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent account-holder confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="The account holder confirmed the activity.",
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes
    )
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(finding,),
        synthesis=synthesis,
        alternative_evaluations=(
            AlternativeEvaluation(
                alternative_id=specification.alternatives[0].alternative_id,
                proposition_id=specification.alternatives[0].proposition.proposition_id,
                evidence_set_id=(
                    specification.alternatives[0].proposition.evidence_manifest.evidence_set_id
                ),
                findings=(finding,),
            ),
        ),
    )
    record.validate_against(specification, outcomes, production=True)
    assert record.evaluation_rule_id == specification.evaluation_rule.evaluation_rule_id
    assert record.evaluation_rule_version == specification.evaluation_rule.rule_version



def test_evaluation_record_requires_one_result_for_each_declared_alternative():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent account-holder confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="A registered criterion result.",
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes
    )
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(finding,),
        synthesis=synthesis,
    )
    with pytest.raises(ValueError, match="exactly one ordered result"):
        record.validate_against(specification, outcomes, production=True)


def test_alternative_evaluation_rejects_misattributed_proposition():
    specification, criterion, alternative = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent account-holder confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="A registered criterion result.",
    )
    result = AlternativeEvaluation(
        alternative_id=alternative.alternative_id,
        proposition_id=str(uuid4()),
        evidence_set_id=alternative.proposition.evidence_manifest.evidence_set_id,
        findings=(finding,),
    )
    with pytest.raises(ValueError, match="proposition ID"):
        result.validate_against(
            alternative, specification.evaluation_rule.criterion_definitions, outcomes
        )


def test_evaluation_record_rejects_duplicate_alternative_results():
    specification, criterion, alternative = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent account-holder confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="A registered criterion result.",
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes
    )
    result = AlternativeEvaluation(
        alternative_id=alternative.alternative_id,
        proposition_id=alternative.proposition.proposition_id,
        evidence_set_id=alternative.proposition.evidence_manifest.evidence_set_id,
        findings=(finding,),
    )
    with pytest.raises(ValueError, match="unique alternative IDs"):
        HypothesisEvaluationRecord(
            evaluation_id=str(uuid4()),
            hypothesis_id=specification.hypothesis.hypothesis_id,
            evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
            evaluation_rule_version=specification.evaluation_rule.rule_version,
            evidence_set_id=str(uuid4()),
            assessment_id=str(uuid4()),
            evaluated_at=datetime.now(timezone.utc),
            findings=(finding,),
            synthesis=synthesis,
            alternative_evaluations=(result, result),
        )


def test_opposing_findings_across_independent_criteria_aggregate_as_conflicting():
    specification, first_criterion, alternative = make_specification()
    second_criterion = CriterionDefinition(
        criterion_id="independent-network-context",
        criterion_type="network_context",
        parameters_json="{}",
        description="An independent network-context condition.",
        independence_assertion_id="network-source",
    )
    first_bearings = specification.evaluation_rule.bearing_definitions
    second_bearings = (
        BearingDefinition(
            criterion_id=second_criterion.criterion_id,
            outcome_code="observed",
            alternative_id=alternative.alternative_id,
            focal_judgment="unexpected",
            alternative_judgment="expected",
            evaluability_role="required",
            rationale="This outcome is more expected under the declared alternative.",
            basis="Independent network evidence.",
            author="analyst",
            review_status="approved",
        ),
        BearingDefinition(
            criterion_id=second_criterion.criterion_id,
            outcome_code="observed",
            alternative_id=RESIDUAL_ALTERNATIVE_ID,
            focal_judgment="unexpected",
            alternative_judgment="compatible",
            evaluability_role="contextual",
            rationale="The residual remains compatible.",
            basis="Broad residual explanation.",
            author="analyst",
            review_status="approved",
        ),
    )
    original = specification.evaluation_rule
    rule = HypothesisEvaluationRule(
        evaluation_rule_id="account-authentication-conflicting-v1",
        rule_version=1,
        proposition_type=original.proposition_type,
        semantics_version=original.semantics_version,
        semantics_fingerprint=original.semantics_fingerprint,
        criterion_definitions=(first_criterion, second_criterion),
        bearing_definitions=first_bearings + second_bearings,
        description="Synthesize independent findings that may conflict.",
        review_status="approved",
    )
    specification = HypothesisSpecification(
        hypothesis=specification.hypothesis,
        alternatives=specification.alternatives,
        evaluation_rule=rule,
    )
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=first_criterion.criterion_type,
        outcome_code="observed",
        meaning="The first independent outcome.",
    ))
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=second_criterion.criterion_type,
        outcome_code="observed",
        meaning="The second independent outcome.",
    ))
    findings = (
        CriterionFinding(
            finding_id=str(uuid4()),
            criterion_id=first_criterion.criterion_id,
            status="matched",
            outcome_code="observed",
            evidence=(EvidenceReference(event_id=str(uuid4())),),
            rationale="The first outcome favors the focal proposition.",
        ),
        CriterionFinding(
            finding_id=str(uuid4()),
            criterion_id=second_criterion.criterion_id,
            status="matched",
            outcome_code="observed",
            evidence=(EvidenceReference(event_id=str(uuid4())),),
            rationale="The second outcome favors the alternative.",
        ),
    )
    result = DiagnosticitySynthesizer().synthesize(
        specification, findings, outcomes
    )
    assert result.outcome == "inconclusive"
    assert result.alternative_standings[0].standing == "conflicting"


def make_two_criterion_specification(
    *,
    second_required=True,
    first_focal_judgment="expected",
):
    specification, first_criterion, alternative = make_specification()
    original = specification.evaluation_rule

    second_criterion = CriterionDefinition(
        criterion_id="independent-network-context",
        criterion_type="network_context",
        parameters_json="{}",
        description="An independent network-context condition.",
        independence_assertion_id="network-source",
    )

    first_bearings = tuple(
        BearingDefinition(
            criterion_id=item.criterion_id,
            outcome_code=item.outcome_code,
            alternative_id=item.alternative_id,
            focal_judgment=first_focal_judgment,
            alternative_judgment=(
                "compatible"
                if item.alternative_id == RESIDUAL_ALTERNATIVE_ID
                else (
                    "expected"
                    if first_focal_judgment == "incompatible"
                    else "unexpected"
                )
            ),
            evaluability_role=item.evaluability_role,
            rationale="The first independent criterion's registered interpretation.",
            basis="Independent first source.",
            author="analyst",
            review_status="approved",
        )
        for item in original.bearing_definitions
    )

    second_bearings = (
        BearingDefinition(
            criterion_id=second_criterion.criterion_id,
            outcome_code="observed",
            alternative_id=alternative.alternative_id,
            focal_judgment="expected",
            alternative_judgment="unexpected",
            evaluability_role="required" if second_required else "contextual",
            rationale="The second criterion favors the focal proposition.",
            basis="Independent network evidence.",
            author="analyst",
            review_status="approved",
        ),
        BearingDefinition(
            criterion_id=second_criterion.criterion_id,
            outcome_code="observed",
            alternative_id=RESIDUAL_ALTERNATIVE_ID,
            focal_judgment="expected",
            alternative_judgment="compatible",
            evaluability_role="contextual",
            rationale="The residual remains compatible.",
            basis="Broad residual explanation.",
            author="analyst",
            review_status="approved",
        ),
    )

    rule = HypothesisEvaluationRule(
        evaluation_rule_id="account-authentication-adversarial-v1",
        rule_version=1,
        proposition_type=original.proposition_type,
        semantics_version=original.semantics_version,
        semantics_fingerprint=original.semantics_fingerprint,
        criterion_definitions=(first_criterion, second_criterion),
        bearing_definitions=first_bearings + second_bearings,
        description="Adversarial tests for criterion completeness.",
        review_status="approved",
    )

    specification = HypothesisSpecification(
        hypothesis=specification.hypothesis,
        alternatives=specification.alternatives,
        evaluation_rule=rule,
    )
    return specification, first_criterion, second_criterion, alternative


def test_multiple_missing_required_criteria_report_incompleteness_once():
    specification, _, _, _ = make_two_criterion_specification()

    result = DiagnosticitySynthesizer().synthesize(
        specification,
        (),
        CriterionOutcomeRegistry(),
    )

    assert result.outcome == "inconclusive"
    assert result.reason_codes.count("required_criteria_incomplete") == 1
    assert "missing_finding:independent-confirmation" in result.reason_codes
    assert "missing_finding:independent-network-context" in result.reason_codes


def test_insufficient_required_finding_marks_evaluation_incomplete():
    specification, first_criterion, _, _ = make_two_criterion_specification(
        second_required=False,
    )
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=first_criterion.criterion_id,
        status="insufficient",
        outcome_code=None,
        evidence=(),
        rationale="The source record is incomplete.",
        missing_information=("Complete account-holder confirmation",),
    )

    result = DiagnosticitySynthesizer().synthesize(
        specification,
        (finding,),
        CriterionOutcomeRegistry(),
    )

    assert result.outcome == "inconclusive"
    assert f"insufficient_finding:{first_criterion.criterion_id}" in result.reason_codes
    assert "required_criteria_incomplete" in result.reason_codes


def test_missing_contextual_criterion_does_not_block_supported_result():
    specification, first_criterion, second_criterion, _ = (
        make_two_criterion_specification(second_required=False)
    )
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=first_criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent confirmation was obtained.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=first_criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="The account holder independently confirmed the activity.",
    )

    result = DiagnosticitySynthesizer().synthesize(
        specification,
        (finding,),
        outcomes,
    )

    assert result.outcome == "supported"
    assert f"missing_finding:{second_criterion.criterion_id}" in result.reason_codes
    assert "required_criteria_incomplete" not in result.reason_codes


def test_complete_contradiction_takes_precedence_over_missing_required_criterion():
    specification, first_criterion, second_criterion, _ = (
        make_two_criterion_specification(
            second_required=True,
            first_focal_judgment="incompatible",
        )
    )
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=first_criterion.criterion_type,
        outcome_code="observed",
        meaning="Complete independent evidence contradicts the focal proposition.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=first_criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="Complete non-influenceable evidence contradicts the focal proposition.",
    )

    result = DiagnosticitySynthesizer().synthesize(
        specification,
        (finding,),
        outcomes,
    )

    assert result.outcome == "not_supported"
    assert "complete_non_influenceable_focal_contradiction" in result.reason_codes
    assert "required_criteria_incomplete" in result.reason_codes
    assert f"missing_finding:{second_criterion.criterion_id}" in result.reason_codes


def test_any_required_bearing_makes_criterion_required_for_completeness():
    specification, first_criterion, _, _ = make_two_criterion_specification(
        second_required=False,
    )

    # The first criterion has a required declared-alternative bearing
    # and a contextual residual bearing.
    first_bearings = tuple(
        bearing
        for bearing in specification.evaluation_rule.bearing_definitions
        if bearing.criterion_id == first_criterion.criterion_id
    )

    assert {bearing.evaluability_role for bearing in first_bearings} == {
        "required",
        "contextual",
    }

    result = DiagnosticitySynthesizer().synthesize(
        specification,
        (),
        CriterionOutcomeRegistry(),
    )

    assert result.outcome == "inconclusive"
    assert "required_criteria_incomplete" in result.reason_codes
    assert f"missing_finding:{first_criterion.criterion_id}" in result.reason_codes



def test_rule_rejects_conflicting_focal_judgments_for_same_criterion_outcome():
    specification, _, _ = make_specification()
    original = specification.evaluation_rule
    first, second = original.bearing_definitions
    contradictory = BearingDefinition(
        criterion_id=second.criterion_id,
        outcome_code=second.outcome_code,
        alternative_id="another-explicit-alternative",
        focal_judgment="incompatible",
        alternative_judgment=second.alternative_judgment,
        evaluability_role=second.evaluability_role,
        rationale=second.rationale,
        basis=second.basis,
        author=second.author,
        review_status=second.review_status,
    )

    with pytest.raises(ValueError, match="focal judgment must be consistent"):
        HypothesisEvaluationRule(
            evaluation_rule_id=original.evaluation_rule_id,
            rule_version=original.rule_version,
            proposition_type=original.proposition_type,
            semantics_version=original.semantics_version,
            semantics_fingerprint=original.semantics_fingerprint,
            criterion_definitions=original.criterion_definitions,
            bearing_definitions=(first, contradictory),
            description=original.description,
            review_status=original.review_status,
        )


def test_evaluation_rule_is_pinned_to_exact_semantics_fingerprint():
    specification, _, _ = make_specification()
    rule = specification.evaluation_rule
    other_fingerprint = "sha256-v1:" + ("0" * 64)
    mismatched_rule = replace(rule, semantics_fingerprint=other_fingerprint)

    with pytest.raises(ValueError, match="exact proposition semantics"):
        mismatched_rule.validate_for(
            specification.hypothesis.proposition,
            production=False,
        )


def test_hypothesis_specification_rejects_alternative_with_different_fingerprint():
    specification, _, alternative = make_specification()
    other_fingerprint = "sha256-v1:" + ("0" * 64)
    mismatched_proposition = replace(
        alternative.proposition,
        semantics_fingerprint=other_fingerprint,
    )
    mismatched_alternative = replace(
        alternative,
        proposition=mismatched_proposition,
    )

    with pytest.raises(ValueError, match="focal proposition semantics"):
        HypothesisSpecification(
            hypothesis=specification.hypothesis,
            alternatives=(mismatched_alternative,),
            evaluation_rule=specification.evaluation_rule,
        )


def test_rule_registry_allows_multiple_immutable_versions_of_one_rule_identity():
    specification, _, _ = make_specification()
    first = replace(
        specification.evaluation_rule,
        evaluation_rule_id="account-authentication-evaluation",
    )
    second = replace(first, rule_version=first.rule_version + 1)
    registry = HypothesisEvaluationRuleRegistry()

    registry.register(first)
    registry.register(second)

    assert registry.get(first.evaluation_rule_id, first.rule_version) is first
    assert registry.get(second.evaluation_rule_id, second.rule_version) is second
    registry.freeze()
    with pytest.raises(RuntimeError, match="registry is frozen"):
        registry.register(replace(second, rule_version=second.rule_version + 1))


def test_residual_bearing_must_share_focal_judgment_with_declared_alternative():
    specification, _, _ = make_specification()
    rule = specification.evaluation_rule
    residual = rule.bearing_definitions[1]
    forged_residual = replace(residual, focal_judgment="incompatible")

    with pytest.raises(ValueError, match="focal judgment must be consistent"):
        replace(
            rule,
            bearing_definitions=(rule.bearing_definitions[0], forged_residual),
        )


def test_diagnosticity_synthesis_rejects_uncalibrated_high_confidence_ceiling():
    specification, _, _ = make_specification()
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (), CriterionOutcomeRegistry(), production=False
    )

    with pytest.raises(ValueError, match="reserved for a future calibrated milestone"):
        replace(synthesis, confidence_ceiling="high")


def test_evaluation_record_rejects_forged_zero_finding_supported_synthesis():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (), outcomes, production=False
    )
    forged_synthesis = replace(synthesis, outcome="supported")
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(),
        alternative_evaluations=(make_alternative_evaluation(specification, criterion),),
        synthesis=forged_synthesis,
    )

    with pytest.raises(ValueError, match="does not match the result derived"):
        record.validate_against(specification, outcomes, production=False)


def test_evaluation_record_rejects_synthesis_forged_against_valid_finding():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent account-holder confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="The account holder independently confirmed the activity.",
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes, production=True
    )
    forged_standing = replace(
        synthesis.alternative_standings[0],
        supporting_finding_ids=(
            *synthesis.alternative_standings[0].supporting_finding_ids,
            str(uuid4()),
        ),
    )
    forged_synthesis = replace(
        synthesis,
        outcome="not_supported",
        reason_codes=("made_up_reason",),
        alternative_standings=(forged_standing,),
    )
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(finding,),
        alternative_evaluations=(make_alternative_evaluation(specification, criterion),),
        synthesis=forged_synthesis,
    )

    with pytest.raises(ValueError, match="does not match the result derived"):
        record.validate_against(specification, outcomes, production=True)


def test_evaluation_record_validation_uses_explicit_experimental_mode():
    specification, criterion, _ = make_specification()
    reviewed_bearings = tuple(
        replace(item, review_status="reviewed")
        for item in specification.evaluation_rule.bearing_definitions
    )
    reviewed_rule = replace(
        specification.evaluation_rule,
        bearing_definitions=reviewed_bearings,
        review_status="reviewed",
    )
    specification = replace(specification, evaluation_rule=reviewed_rule)
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="observed",
        meaning="Independent account-holder confirmation.",
    ))
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="observed",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="The account holder independently confirmed the activity.",
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes, production=False
    )
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(finding,),
        alternative_evaluations=(make_alternative_evaluation(specification, criterion),),
        synthesis=synthesis,
    )

    record.validate_against(specification, outcomes, production=False)
    with pytest.raises(ValueError, match="production"):
        record.validate_against(specification, outcomes, production=True)


def test_evaluation_record_rejects_finding_with_unregistered_outcome():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="matched",
        outcome_code="made_up",
        evidence=(EvidenceReference(event_id=str(uuid4())),),
        rationale="A finding with an unregistered outcome.",
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification,
        (),
        CriterionOutcomeRegistry(),
        production=False,
    )
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(finding,),
        alternative_evaluations=(make_alternative_evaluation(specification, criterion),),
        synthesis=synthesis,
    )

    with pytest.raises(KeyError, match="unregistered criterion outcome"):
        record.validate_against(specification, outcomes, production=False)



def test_evaluation_record_rejects_supported_synthesis_from_only_insufficient_findings():
    specification, criterion, _ = make_specification()
    outcomes = CriterionOutcomeRegistry()
    finding = CriterionFinding(
        finding_id=str(uuid4()),
        criterion_id=criterion.criterion_id,
        status="insufficient",
        outcome_code=None,
        evidence=(),
        rationale="The source is incomplete.",
        missing_information=("A complete independent confirmation",),
    )
    synthesis = DiagnosticitySynthesizer().synthesize(
        specification, (finding,), outcomes, production=True
    )
    forged_synthesis = replace(synthesis, outcome="supported")
    record = HypothesisEvaluationRecord(
        evaluation_id=str(uuid4()),
        hypothesis_id=specification.hypothesis.hypothesis_id,
        evaluation_rule_id=specification.evaluation_rule.evaluation_rule_id,
        evaluation_rule_version=specification.evaluation_rule.rule_version,
        evidence_set_id=str(uuid4()),
        assessment_id=str(uuid4()),
        evaluated_at=datetime.now(timezone.utc),
        findings=(finding,),
        alternative_evaluations=(make_alternative_evaluation(specification, criterion),),
        synthesis=forged_synthesis,
    )

    with pytest.raises(ValueError, match="does not match the result derived"):
        record.validate_against(specification, outcomes, production=True)

def test_evaluation_rule_requires_distinct_independence_assertion_ids():
    specification, _, _, _ = make_two_criterion_specification()
    rule = specification.evaluation_rule
    first, second = rule.criterion_definitions

    assert first.independence_assertion_id != second.independence_assertion_id

    duplicate = replace(
        second,
        independence_assertion_id=first.independence_assertion_id,
    )
    with pytest.raises(
        ValueError,
        match="distinct independence assertion identifiers",
    ):
        HypothesisEvaluationRule(
            evaluation_rule_id=rule.evaluation_rule_id,
            rule_version=rule.rule_version,
            proposition_type=rule.proposition_type,
            semantics_version=rule.semantics_version,
            semantics_fingerprint=rule.semantics_fingerprint,
            criterion_definitions=(first, duplicate),
            bearing_definitions=rule.bearing_definitions,
            description=rule.description,
            review_status=rule.review_status,
        )
