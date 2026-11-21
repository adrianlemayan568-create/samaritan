from datetime import datetime, timezone

from samaritan.analysis_contracts import EvidenceReference
from samaritan.criterion_evaluation import CriterionEvaluationEngine
from samaritan.hypothesis_evaluation import HypothesisEvaluationEngine
from samaritan.contracts import (
    Event,
    Observation,
    Source,
    new_id,
    payload_checksum,
)
from samaritan.database import Database
from samaritan.evaluation_contracts import EvidenceSet
from samaritan.proposition_contracts import (
    Proposition,
    PropositionEvidenceManifest,
    PropositionFieldProvenance,
    PropositionRoleValue,
    PropositionTemporalScope,
    CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
    StructuredHypothesis,
    deterministic_proposition_id,
)
from samaritan.reasoning_contracts import (
    CriterionDefinition,
    CriterionEvaluatorDefinition,
    CriterionEvaluatorRegistry,
    CriterionOutcomeDefinition,
    CriterionOutcomeRegistry,
    CriterionParameterDefinition,
    BearingDefinition,
    HypothesisAlternative,
    HypothesisEvaluationRule,
    HypothesisSpecification,
    RESIDUAL_ALTERNATIVE_ID,
)
from samaritan.repository import Repository
from samaritan.proposition_semantics import (
    PropositionRoleDefinition,
    PropositionSemantics,
    PropositionSemanticsRegistry,
)
from samaritan.proposition_semantics_fingerprint import proposition_semantics_fingerprint


_TEST_SEMANTICS_FINGERPRINT = "test-semantics-fingerprint-v1"


def make_evaluation_context(tmp_path):
    database = Database(tmp_path / "criterion.db")
    database.initialize()
    repository = Repository(database)
    now = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="criterion evaluator test",
        created_at=now,
    )
    repository.register_source(source)
    observation = Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=now,
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    event = Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type="authentication_success",
        occurred_at=now,
        original_timestamp="2026-11-20T10:00:00Z",
        normalized_data_json='{"result":"success"}',
        participants=(),
    )
    repository.persist_observation(observation, events=(event,))
    semantics = PropositionSemantics(
        proposition_type="test_authentication",
        semantics_version=1,
        roles=(
            PropositionRoleDefinition(
                role_name="success_event",
                role_kind="event",
                cardinality="one",
                event_type="authentication_success",
                meaning="The successful authentication event.",
            ),
        ),
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(semantics)
    semantics_fingerprint = proposition_semantics_fingerprint(semantics)
    evidence_set_id = new_id()
    proposition = Proposition(
        proposition_id=deterministic_proposition_id(
            "test_authentication",
            1,
            (PropositionRoleValue(
                "success_event", "event", event.event_id, event.occurred_at
            ),),
            PropositionTemporalScope(now, now),
            semantics_fingerprint=semantics_fingerprint,
            identity_algorithm_version=CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
        ),
        proposition_type="test_authentication",
        semantics_version=1,
        roles=(
            PropositionRoleValue(
                "success_event", "event", event.event_id, event.occurred_at
            ),
        ),
        temporal_scope=PropositionTemporalScope(now, now),
        field_provenance=(
            PropositionFieldProvenance(
                "success_event",
                (event.event_id,),
                (observation.observation_id,),
            ),
        ),
        construction_id="test-construction",
        construction_version=1,
        semantics_fingerprint=semantics_fingerprint,
        identity_algorithm_version=CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
        evidence_manifest=PropositionEvidenceManifest(
            evidence_set_id=evidence_set_id,
            evidence=(EvidenceReference(event_id=event.event_id),),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=evidence_set_id,
        evidence=(EvidenceReference(event_id=event.event_id),),
        created_at=now,
    )
    return repository, proposition, evidence_set, event, semantics_registry


def test_registered_event_attribute_evaluator_returns_stance_neutral_finding(tmp_path):
    repository, proposition, evidence_set, event, _ = make_evaluation_context(tmp_path)
    parameters = (
        CriterionParameterDefinition("event_role", "string", required=True),
        CriterionParameterDefinition("attribute_name", "string", required=True),
        CriterionParameterDefinition("expected_value", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="authentication-result",
        criterion_type="authentication_result",
        parameters_json='{"event_role":"success_event","attribute_name":"result","expected_value":"success"}',
        description="The normalized authentication result equals the registered expected value.",
        independence_group="authentication-event",
        parameter_definitions=parameters,
    )
    evaluators = CriterionEvaluatorRegistry()
    evaluators.register(
        CriterionEvaluatorDefinition(
            criterion_type=criterion.criterion_type,
            evaluator_version=1,
            evaluator_type="event_attribute_equals",
            parameter_definitions=parameters,
            description="Compare one registered normalized event attribute to a typed expected value.",
        )
    )
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="equal",
        meaning="The event attribute equals the expected value.",
    ))
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="not_equal",
        meaning="The event attribute differs from the expected value or is absent.",
    ))
    fixed_time = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    engine = CriterionEvaluationEngine(
        repository, evaluators, outcomes, clock=lambda: fixed_time
    )
    finding = engine.evaluate(criterion, proposition, evidence_set)
    repeated = engine.evaluate(criterion, proposition, evidence_set)
    assert finding.finding_id == repeated.finding_id
    assert finding.created_at == repeated.created_at == fixed_time
    assert finding.status == "matched"
    assert finding.outcome_code == "equal"
    assert finding.evidence == (
        EvidenceReference(event_id=event.event_id),
        EvidenceReference(observation_id=event.observation_id),
    )
    assert not hasattr(finding, "stance")


def test_evaluator_registry_rejects_arbitrary_evaluator_callbacks():
    import pytest

    with pytest.raises(ValueError, match="unsupported built-in"):
        CriterionEvaluatorDefinition(
            criterion_type="custom",
            evaluator_version=1,
            evaluator_type="arbitrary_python_callback",
            parameter_definitions=(),
            description="Must not accept arbitrary callback execution.",
        )



def test_hypothesis_evaluation_engine_runs_findings_and_synthesis_end_to_end(tmp_path):
    repository, proposition, evidence_set, event, semantics_registry = make_evaluation_context(tmp_path)
    parameter_definitions = (
        CriterionParameterDefinition("event_role", "string", required=True),
        CriterionParameterDefinition("attribute_name", "string", required=True),
        CriterionParameterDefinition("expected_value", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="authentication-result",
        criterion_type="authentication_result",
        parameters_json='{"event_role":"success_event","attribute_name":"result","expected_value":"success"}',
        description="The normalized authentication result equals the registered expected value.",
        independence_group="authentication-event",
        parameter_definitions=parameter_definitions,
    )
    evaluators = CriterionEvaluatorRegistry()
    evaluators.register(CriterionEvaluatorDefinition(
        criterion_type=criterion.criterion_type,
        evaluator_version=1,
        evaluator_type="event_attribute_equals",
        parameter_definitions=parameter_definitions,
        description="Compare a registered normalized event attribute with its expected value.",
    ))
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="equal",
        meaning="The event attribute equals the expected value.",
    ))
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="not_equal",
        meaning="The event attribute differs from the expected value.",
    ))

    alternative_event_id = new_id()
    alternative_observation_id = new_id()
    alternative_roles = (
        PropositionRoleValue(
            "success_event", "event", alternative_event_id, event.occurred_at
        ),
    )
    alternative_scope = proposition.temporal_scope
    alternative_provenance = (
        PropositionFieldProvenance(
            "success_event",
            (alternative_event_id,),
            (alternative_observation_id,),
        ),
    )
    alternative_proposition = Proposition(
        proposition_id=deterministic_proposition_id(
            proposition.proposition_type,
            proposition.semantics_version,
            alternative_roles,
            alternative_scope,
            semantics_fingerprint=proposition.semantics_fingerprint,
            identity_algorithm_version=proposition.identity_algorithm_version,
        ),
        proposition_type=proposition.proposition_type,
        semantics_version=proposition.semantics_version,
        roles=alternative_roles,
        temporal_scope=alternative_scope,
        field_provenance=alternative_provenance,
        construction_id=proposition.construction_id,
        construction_version=proposition.construction_version,
        semantics_fingerprint=proposition.semantics_fingerprint,
        identity_algorithm_version=proposition.identity_algorithm_version,
        evidence_manifest=PropositionEvidenceManifest(
            evidence_set_id=evidence_set.evidence_set_id,
            evidence=(EvidenceReference(event_id=alternative_event_id),),
        ),
    )
    alternative = HypothesisAlternative(
        alternative_id=new_id(),
        proposition=alternative_proposition,
        description="The successful authentication had a different recorded outcome.",
    )
    bearings = (
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code="equal",
            alternative_id=alternative.alternative_id,
            focal_judgment="expected",
            alternative_judgment="unexpected",
            evaluability_role="required",
            rationale="The observed value is expected under the focal proposition.",
            basis="Validated normalized event attribute.",
            author="analyst",
            review_status="approved",
        ),
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code="equal",
            alternative_id=RESIDUAL_ALTERNATIVE_ID,
            focal_judgment="expected",
            alternative_judgment="compatible",
            evaluability_role="contextual",
            rationale="The residual remains broadly compatible.",
            basis="Residual explanation is not excluded.",
            author="analyst",
            review_status="approved",
        ),
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code="not_equal",
            alternative_id=alternative.alternative_id,
            focal_judgment="incompatible",
            alternative_judgment="expected",
            evaluability_role="required",
            rationale="The observed value contradicts the focal proposition.",
            basis="Validated normalized event attribute.",
            author="analyst",
            review_status="approved",
        ),
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code="not_equal",
            alternative_id=RESIDUAL_ALTERNATIVE_ID,
            focal_judgment="unexpected",
            alternative_judgment="compatible",
            evaluability_role="contextual",
            rationale="The residual remains broadly compatible.",
            basis="Residual explanation is not excluded.",
            author="analyst",
            review_status="approved",
        ),
    )
    rule = HypothesisEvaluationRule(
        evaluation_rule_id="test-authentication-rule-v1",
        rule_version=1,
        proposition_type=proposition.proposition_type,
        semantics_version=proposition.semantics_version,
        semantics_fingerprint=proposition.semantics_fingerprint,
        criterion_definitions=(criterion,),
        bearing_definitions=bearings,
        description="Evaluate the normalized authentication result.",
        review_status="approved",
    )
    specification = HypothesisSpecification(
        hypothesis=StructuredHypothesis(
            hypothesis_id=new_id(),
            proposition=proposition,
            created_at=datetime.now(timezone.utc),
        ),
        alternatives=(alternative,),
        evaluation_rule=rule,
    )
    fixed_time = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    engine = HypothesisEvaluationEngine(
        repository,
        evaluators,
        outcomes,
        semantics_registry=semantics_registry,
        clock=lambda: fixed_time,
    )
    record = engine.evaluate(specification, evidence_set)
    repeated = engine.evaluate(specification, evidence_set)
    assert record.evaluation_id == repeated.evaluation_id
    assert record.assessment_id == repeated.assessment_id
    assert record.evaluated_at == repeated.evaluated_at == fixed_time
    assert record.findings[0].finding_id == repeated.findings[0].finding_id
    assert record.evaluation_rule_id == rule.evaluation_rule_id
    assert record.evidence_set_id == evidence_set.evidence_set_id
    assert record.findings[0].status == "matched"
    assert record.synthesis.outcome == "supported"



def test_missing_registered_attribute_is_insufficient_not_a_negative_finding(tmp_path):
    import pytest

    repository, proposition, evidence_set, _, _ = make_evaluation_context(tmp_path)
    parameters = (
        CriterionParameterDefinition("event_role", "string", required=True),
        CriterionParameterDefinition("attribute_name", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="missing-context-attribute",
        criterion_type="context_attribute",
        parameters_json='{"event_role":"success_event","attribute_name":"device_fingerprint"}',
        description="Check whether the normalized event records a device fingerprint.",
        independence_group="authentication-event",
        parameter_definitions=parameters,
    )
    evaluators = CriterionEvaluatorRegistry()
    evaluators.register(CriterionEvaluatorDefinition(
        criterion_type=criterion.criterion_type,
        evaluator_version=1,
        evaluator_type="event_attribute_present",
        parameter_definitions=parameters,
        description="Check for a registered normalized event attribute.",
    ))
    finding = CriterionEvaluationEngine(
        repository, evaluators, CriterionOutcomeRegistry()
    ).evaluate(criterion, proposition, evidence_set)

    assert finding.status == "insufficient"
    assert finding.outcome_code is None
    assert "coverage" in finding.missing_information[0].lower()
