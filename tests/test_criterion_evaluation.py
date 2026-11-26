from dataclasses import replace

import pytest
from datetime import datetime, timedelta, timezone

from samaritan.analysis_contracts import EvidenceReference
from samaritan.criterion_evaluation import (
    CriterionEvaluationEngine,
    CriterionEvaluationError,
)
from samaritan.hypothesis_evaluation import HypothesisEvaluationEngine
from samaritan.contracts import (
    Event,
    Entity,
    EventParticipant,
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
    LEGACY_PROPOSITION_IDENTITY_ALGORITHM,
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
    HypothesisEvaluationRuleRegistry,
    HypothesisSpecification,
    RESIDUAL_ALTERNATIVE_ID,
)
from samaritan.repository import Repository
from samaritan.proposition_semantics import (
    PropositionRoleDefinition,
    PropositionSemantics,
    PropositionSemanticsRegistry,
    SemanticParameterDefinition,
)
from samaritan.proposition_semantics_fingerprint import proposition_semantics_fingerprint
from samaritan.proposition_construction import PropositionConstructor
from samaritan.proposition_construction_contracts import (
    EventAttributeParameterBinding,
    EventRoleBinding,
    EntityRoleBinding,
    ParticipantRoleBinding,
    PropositionConstructionDefinition,
    PropositionConstructionRegistry,
)
from samaritan.proposition_validation import PropositionProvenanceValidator




def _test_construction_registry():
    registry = PropositionConstructionRegistry()
    registry.register(
        PropositionConstructionDefinition(
            construction_id="test-construction",
            construction_version=1,
            proposition_type="test_authentication",
            semantics_version=1,
            event_bindings=(
                EventRoleBinding("success_event", "authentication_success"),
            ),
            entity_bindings=(),
            description="Construct the test authentication proposition from its source event.",
        )
    )
    return registry


def _test_provenance_validator(repository, semantics_registry):
    return PropositionProvenanceValidator(
        repository, semantics_registry, _test_construction_registry()
    )


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
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="criterion-test-account",
        created_at=now,
    )
    event = Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type="authentication_success",
        occurred_at=now,
        original_timestamp="2026-11-20T10:00:00Z",
        normalized_data_json='{"result":"success"}',
        participants=(EventParticipant(account.entity_id, "target"),),
    )
    repository.persist_observation(observation, entities=(account,), events=(event,))
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
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(EvidenceReference(event_id=event.event_id),),
        created_at=now,
    )
    repository.persist_evidence_set(evidence_set)
    constructor = PropositionConstructor(
        repository, semantics_registry, _test_construction_registry()
    )
    proposition = constructor.construct(evidence_set, "test-construction", 1)
    return repository, proposition, evidence_set, event, semantics_registry

def test_registered_event_attribute_evaluator_returns_stance_neutral_finding(tmp_path):
    repository, proposition, evidence_set, event, semantics_registry = make_evaluation_context(tmp_path)
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
        independence_assertion_id="authentication-event",
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
        repository, evaluators, outcomes,
        provenance_validator=_test_provenance_validator(repository, semantics_registry),
        clock=lambda: fixed_time,
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

    with pytest.raises(ValueError, match="unsupported built-in"):
        CriterionEvaluatorDefinition(
            criterion_type="custom",
            evaluator_version=1,
            evaluator_type="arbitrary_python_callback",
            parameter_definitions=(),
            description="Must not accept arbitrary callback execution.",
        )



def test_hypothesis_evaluation_engine_runs_findings_and_synthesis_end_to_end(tmp_path, monkeypatch):
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
        independence_assertion_id="authentication-event",
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

    alternative_source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="alternative proposition construction evidence",
        created_at=event.occurred_at,
    )
    repository.register_source(alternative_source)
    alternative_observation = Observation(
        observation_id=new_id(),
        source_id=alternative_source.source_id,
        received_at=event.occurred_at,
        raw_payload='{"result":"failure"}',
        payload_sha256=payload_checksum('{"result":"failure"}'),
    )
    alternative_event = Event(
        event_id=new_id(),
        observation_id=alternative_observation.observation_id,
        event_type="authentication_success",
        occurred_at=event.occurred_at,
        original_timestamp=event.original_timestamp,
        normalized_data_json='{"result":"failure"}',
        participants=(),
    )
    repository.persist_observation(
        alternative_observation, events=(alternative_event,)
    )
    alternative_evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(EvidenceReference(event_id=alternative_event.event_id),),
        created_at=event.occurred_at,
    )
    repository.persist_evidence_set(alternative_evidence_set)
    alternative_proposition = PropositionConstructor(
        repository, semantics_registry, _test_construction_registry()
    ).construct(alternative_evidence_set, "test-construction", 1)
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
            focal_judgment="incompatible",
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
    rule_registry = HypothesisEvaluationRuleRegistry()
    rule_registry.register(rule)
    engine = HypothesisEvaluationEngine(
        repository,
        evaluators,
        outcomes,
        rule_registry,
        construction_registry=_test_construction_registry(),
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

    forged_alternative = replace(
        alternative,
        proposition=replace(alternative_proposition, construction_version=2),
    )
    forged_alternative_specification = replace(
        specification, alternatives=(forged_alternative,)
    )

    def evaluator_must_not_run(_evidence):
        raise AssertionError("criterion evidence resolution ran before alternative authentication")

    monkeypatch.setattr(
        engine.criterion_engine, "_resolve_evidence", evaluator_must_not_run
    )
    with pytest.raises(ValueError, match="unknown proposition construction"):
        engine.evaluate(forged_alternative_specification, evidence_set)

    # An alternative's construction history is authenticated independently
    # from the focal EvidenceSet used for comparative criterion evaluation.
    missing_alternative_set = replace(
        alternative.proposition.evidence_manifest,
        evidence_set_id=new_id(),
    )
    missing_alternative = replace(
        alternative,
        proposition=replace(
            alternative.proposition,
            evidence_manifest=missing_alternative_set,
        ),
    )
    missing_alternative_specification = replace(
        specification, alternatives=(missing_alternative,)
    )
    with pytest.raises(
        ValueError,
        match="evidence manifest references an EvidenceSet that is not persisted",
    ):
        engine.evaluate(missing_alternative_specification, evidence_set)

    mismatched_alternative_manifest = replace(
        alternative.proposition.evidence_manifest,
        evidence=(EvidenceReference(event_id=event.event_id),),
    )
    mismatched_alternative = replace(
        alternative,
        proposition=replace(
            alternative.proposition,
            evidence_manifest=mismatched_alternative_manifest,
        ),
    )
    mismatched_alternative_specification = replace(
        specification, alternatives=(mismatched_alternative,)
    )
    with pytest.raises(
        ValueError,
        match="evidence manifest does not match the persisted EvidenceSet selection",
    ):
        engine.evaluate(mismatched_alternative_specification, evidence_set)

    altered_rule = replace(rule, description="Unregistered altered definition.")
    altered_specification = replace(
        specification, evaluation_rule=altered_rule
    )
    with pytest.raises(ValueError, match="does not match its registered immutable definition"):
        engine.evaluate(altered_specification, evidence_set)

    empty_rule_registry_engine = HypothesisEvaluationEngine(
        repository,
        evaluators,
        outcomes,
        HypothesisEvaluationRuleRegistry(),
        construction_registry=_test_construction_registry(),
        semantics_registry=semantics_registry,
        clock=lambda: fixed_time,
    )
    with pytest.raises(ValueError, match="evaluation rule version is not registered"):
        empty_rule_registry_engine.evaluate(specification, evidence_set)

    mismatched_evidence = replace(evidence_set, evidence=())
    with pytest.raises(ValueError, match="must exactly match the focal proposition evidence manifest"):
        engine.evaluate(specification, mismatched_evidence)

    incomplete_rule = replace(
        rule,
        bearing_definitions=tuple(
            item for item in rule.bearing_definitions
            if not (
                item.outcome_code == "not_equal"
                and item.alternative_id == alternative.alternative_id
            )
        ),
    )
    with pytest.raises(ValueError, match="incomplete bearing coverage"):
        incomplete_rule.validate_outcome_coverage(
            evaluators, outcomes, (alternative.alternative_id,)
        )

    forged_fingerprint = "sha256-v1:" + ("0" * 64)
    forged_focal = replace(
        proposition,
        semantics_fingerprint=forged_fingerprint,
        proposition_id=deterministic_proposition_id(
            proposition.proposition_type,
            proposition.semantics_version,
            proposition.roles,
            proposition.temporal_scope,
            proposition.parameters_json,
            semantics_fingerprint=forged_fingerprint,
            identity_algorithm_version=proposition.identity_algorithm_version,
        ),
    )
    forged_alternative = replace(
        alternative.proposition,
        semantics_fingerprint=forged_fingerprint,
        proposition_id=deterministic_proposition_id(
            alternative.proposition.proposition_type,
            alternative.proposition.semantics_version,
            alternative.proposition.roles,
            alternative.proposition.temporal_scope,
            alternative.proposition.parameters_json,
            semantics_fingerprint=forged_fingerprint,
            identity_algorithm_version=alternative.proposition.identity_algorithm_version,
        ),
    )
    forged_rule = replace(rule, semantics_fingerprint=forged_fingerprint)
    forged_specification = HypothesisSpecification(
        hypothesis=replace(specification.hypothesis, proposition=forged_focal),
        alternatives=(replace(alternative, proposition=forged_alternative),),
        evaluation_rule=forged_rule,
    )
    with pytest.raises(ValueError, match="fingerprint does not match exact semantics"):
        engine.evaluate(forged_specification, evidence_set)

    legacy_focal = replace(
        proposition,
        identity_algorithm_version=LEGACY_PROPOSITION_IDENTITY_ALGORITHM,
        proposition_id=deterministic_proposition_id(
            proposition.proposition_type,
            proposition.semantics_version,
            proposition.roles,
            proposition.temporal_scope,
            proposition.parameters_json,
            identity_algorithm_version=LEGACY_PROPOSITION_IDENTITY_ALGORITHM,
        ),
    )
    legacy_specification = HypothesisSpecification(
        hypothesis=replace(specification.hypothesis, proposition=legacy_focal),
        alternatives=specification.alternatives,
        evaluation_rule=rule,
    )
    with pytest.raises(ValueError, match="current fingerprint-bound proposition identity"):
        engine.evaluate(legacy_specification, evidence_set)

    monkeypatch.setattr(repository, "get_evidence_set", lambda _evidence_set_id: None)
    with pytest.raises(
        ValueError,
        match="persisted historical selection",
    ):
        engine.evaluate(specification, evidence_set)



def test_missing_registered_attribute_is_insufficient_not_a_negative_finding(tmp_path):

    repository, proposition, evidence_set, _, semantics_registry = make_evaluation_context(tmp_path)
    parameters = (
        CriterionParameterDefinition("event_role", "string", required=True),
        CriterionParameterDefinition("attribute_name", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="missing-context-attribute",
        criterion_type="context_attribute",
        parameters_json='{"event_role":"success_event","attribute_name":"device_fingerprint"}',
        description="Check whether the normalized event records a device fingerprint.",
        independence_assertion_id="authentication-event",
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
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        vocabulary_version=criterion.outcome_vocabulary_version,
        outcome_code="present",
        meaning="The registered normalized event attribute is present.",
    ))
    finding = CriterionEvaluationEngine(
        repository, evaluators, outcomes,
        provenance_validator=_test_provenance_validator(repository, semantics_registry),
    ).evaluate(criterion, proposition, evidence_set)

    assert finding.status == "insufficient"
    assert finding.outcome_code is None
    assert "coverage" in finding.missing_information[0].lower()


def test_direct_criterion_engine_rejects_missing_evaluator_outcomes_before_execution(tmp_path, monkeypatch):
    repository, proposition, evidence_set, _, semantics_registry = make_evaluation_context(tmp_path)
    parameters = (
        CriterionParameterDefinition("event_role", "string", required=True),
        CriterionParameterDefinition("attribute_name", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="preflight-outcome-coverage",
        criterion_type="context_attribute",
        parameters_json='{"event_role":"success_event","attribute_name":"device_fingerprint"}',
        description="Check whether the normalized event records a device fingerprint.",
        independence_assertion_id="authentication-event",
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
    engine = CriterionEvaluationEngine(
        repository, evaluators, CriterionOutcomeRegistry(),
        provenance_validator=_test_provenance_validator(repository, semantics_registry),
    )

    def unexpected_evidence_resolution(_evidence):
        raise AssertionError("evidence resolution must not run before outcome preflight")

    monkeypatch.setattr(engine, "_resolve_evidence", unexpected_evidence_resolution)
    with pytest.raises(
        CriterionEvaluationError,
        match="criterion evaluator outcomes are not fully registered",
    ):
        engine.evaluate(criterion, proposition, evidence_set)


def test_temporal_gap_reports_ordering_violation_separately(tmp_path):
    repository, _, _, original_event, _ = make_evaluation_context(tmp_path)
    now = original_event.occurred_at
    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="temporal ordering test",
        created_at=now,
    )
    later_observation = Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=now + timedelta(seconds=30),
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    later_event = Event(
        event_id=new_id(),
        observation_id=later_observation.observation_id,
        event_type="authentication_failure",
        occurred_at=now + timedelta(seconds=30),
        original_timestamp=(now + timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
        normalized_data_json="{}",
        participants=(),
    )
    repository.register_source(source)
    repository.persist_observation(later_observation, events=(later_event,))

    semantics = PropositionSemantics(
        proposition_type="temporal_ordering_test",
        semantics_version=1,
        roles=(
            PropositionRoleDefinition(
                role_name="earlier_event",
                role_kind="event",
                cardinality="one",
                event_type="authentication_failure",
                meaning="The event assigned to the earlier semantic role.",
            ),
            PropositionRoleDefinition(
                role_name="later_event",
                role_kind="event",
                cardinality="one",
                event_type="authentication_success",
                meaning="The event assigned to the later semantic role.",
            ),
        ),
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(semantics)
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(
        PropositionConstructionDefinition(
            construction_id="temporal-ordering-test",
            construction_version=1,
            proposition_type="temporal_ordering_test",
            semantics_version=1,
            event_bindings=(
                EventRoleBinding("earlier_event", "authentication_failure"),
                EventRoleBinding("later_event", "authentication_success"),
            ),
            entity_bindings=(),
            description="Bind event roles to distinct source event types.",
        )
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(
            EvidenceReference(event_id=original_event.event_id),
            EvidenceReference(event_id=later_event.event_id),
        ),
        created_at=now + timedelta(minutes=1),
    )
    repository.persist_evidence_set(evidence_set)
    proposition = PropositionConstructor(
        repository, semantics_registry, construction_registry
    ).construct(evidence_set, "temporal-ordering-test", 1)

    parameters = (
        CriterionParameterDefinition("earlier_event_role", "string", required=True),
        CriterionParameterDefinition("later_event_role", "string", required=True),
        CriterionParameterDefinition("maximum_seconds", "integer", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="temporal-ordering",
        criterion_type="temporal_gap",
        parameters_json='{"earlier_event_role":"earlier_event","later_event_role":"later_event","maximum_seconds":60}',
        description="Check strict event ordering and maximum elapsed seconds.",
        independence_assertion_id="temporal-event-order",
        parameter_definitions=parameters,
    )
    evaluators = CriterionEvaluatorRegistry()
    evaluators.register(CriterionEvaluatorDefinition(
        criterion_type=criterion.criterion_type,
        evaluator_version=1,
        evaluator_type="temporal_gap_within",
        parameter_definitions=parameters,
        description="Check strict event ordering and maximum elapsed seconds.",
    ))
    outcomes = CriterionOutcomeRegistry()
    for code in ("within_limit", "exceeds_limit", "ordering_violation"):
        outcomes.register(CriterionOutcomeDefinition(
            criterion_type=criterion.criterion_type,
            outcome_code=code,
            meaning=f"Temporal outcome {code}.",
        ))

    finding = CriterionEvaluationEngine(
        repository, evaluators, outcomes,
        provenance_validator=PropositionProvenanceValidator(
            repository, semantics_registry, construction_registry
        ),
    ).evaluate(criterion, proposition, evidence_set)

    assert finding.status == "not_matched"
    assert finding.outcome_code == "ordering_violation"

def test_event_type_count_is_scoped_to_entity_and_proposition_temporal_window(tmp_path):
    repository, base_proposition, _, success_event, _ = make_evaluation_context(tmp_path)
    now = success_event.occurred_at
    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="entity-scoped event count test",
        created_at=now,
    )
    repository.register_source(source)
    alice = repository.get_entities_by_ids(
        (success_event.participants[0].entity_id,)
    )[0]
    bob = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="bob-count-scope", created_at=now,
    )

    selected_events = []
    for entity, offset in (
        (alice, 0),
        (alice, 0),
        (bob, 0),
        (alice, 30),
    ):
        occurred_at = now + timedelta(seconds=offset)
        observation = Observation(
            observation_id=new_id(),
            source_id=source.source_id,
            received_at=occurred_at,
            raw_payload="{}",
            payload_sha256=payload_checksum("{}"),
        )
        event = Event(
            event_id=new_id(),
            observation_id=observation.observation_id,
            event_type="authentication_failure",
            occurred_at=occurred_at,
            original_timestamp=occurred_at.isoformat().replace("+00:00", "Z"),
            normalized_data_json="{}",
            participants=(EventParticipant(entity.entity_id, "target"),),
        )
        repository.persist_observation(
            observation,
            entities=(entity,) if repository.get_entities_by_ids((entity.entity_id,)) == () else (),
            events=(event,),
        )
        selected_events.append(event)

    semantics = PropositionSemantics(
        proposition_type="entity_scoped_authentication",
        semantics_version=1,
        roles=(
            PropositionRoleDefinition(
                role_name="account",
                role_kind="entity",
                cardinality="one",
                entity_type="user_account",
                meaning="The account whose activity is being evaluated.",
            ),
            PropositionRoleDefinition(
                role_name="success_event",
                role_kind="event",
                cardinality="one",
                event_type="authentication_success",
                meaning="The proposition's successful authentication event.",
            ),
        ),
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(semantics)
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(
        PropositionConstructionDefinition(
            construction_id="entity-scoped-test",
            construction_version=1,
            proposition_type=semantics.proposition_type,
            semantics_version=semantics.semantics_version,
            event_bindings=(
                EventRoleBinding("success_event", "authentication_success"),
            ),
            entity_bindings=(
                EntityRoleBinding(
                    "account",
                    (ParticipantRoleBinding("success_event", "target"),),
                ),
            ),
            description="Bind the account to the target participant of the success event.",
        )
    )
    fingerprint = proposition_semantics_fingerprint(semantics)
    roles = (
        PropositionRoleValue("account", "entity", alice.entity_id),
        PropositionRoleValue(
            "success_event", "event", success_event.event_id, success_event.occurred_at
        ),
    )
    scope = PropositionTemporalScope(now, now)
    provenance = (
        PropositionFieldProvenance(
            "account", (success_event.event_id,), (success_event.observation_id,)
        ),
        PropositionFieldProvenance(
            "success_event", (success_event.event_id,), (success_event.observation_id,)
        ),
    )
    proposition = Proposition(
        proposition_id=deterministic_proposition_id(
            semantics.proposition_type,
            semantics.semantics_version,
            roles,
            scope,
            semantics_fingerprint=fingerprint,
            identity_algorithm_version=CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
        ),
        proposition_type=semantics.proposition_type,
        semantics_version=semantics.semantics_version,
        roles=roles,
        temporal_scope=scope,
        field_provenance=provenance,
        construction_id="entity-scoped-test",
        construction_version=1,
        semantics_fingerprint=fingerprint,
        identity_algorithm_version=CURRENT_PROPOSITION_IDENTITY_ALGORITHM,
        evidence_manifest=PropositionEvidenceManifest(
            evidence_set_id=new_id(),
            evidence=(
                EvidenceReference(event_id=success_event.event_id),
                *(EvidenceReference(event_id=item.event_id) for item in selected_events),
            ),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=proposition.evidence_manifest.evidence_set_id,
        evidence=proposition.evidence_manifest.evidence,
        created_at=now + timedelta(minutes=1),
    )
    repository.persist_evidence_set(evidence_set)
    parameters = (
        CriterionParameterDefinition("event_type", "string", required=True),
        CriterionParameterDefinition("minimum_count", "integer", required=True),
        CriterionParameterDefinition("entity_role", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="alice-authentication-failure-count",
        criterion_type="account_failure_count",
        parameters_json='{"event_type":"authentication_failure","minimum_count":2,"entity_role":"account"}',
        description="Count failures for the focal account within the proposition time scope.",
        independence_assertion_id="account-event-source",
        parameter_definitions=parameters,
    )
    evaluators = CriterionEvaluatorRegistry()
    evaluators.register(CriterionEvaluatorDefinition(
        criterion_type=criterion.criterion_type,
        evaluator_version=1,
        evaluator_type="event_type_count",
        parameter_definitions=parameters,
        description="Count matching event types for one proposition entity within its temporal scope.",
    ))
    outcomes = CriterionOutcomeRegistry()
    for code in ("threshold_met", "threshold_not_met"):
        outcomes.register(CriterionOutcomeDefinition(
            criterion_type=criterion.criterion_type,
            outcome_code=code,
            meaning=f"Event-count outcome {code}.",
        ))

    finding = CriterionEvaluationEngine(
        repository, evaluators, outcomes,
        provenance_validator=PropositionProvenanceValidator(
            repository, semantics_registry, construction_registry
        ),
    ).evaluate(criterion, proposition, evidence_set)

    assert finding.status == "matched"
    assert finding.outcome_code == "threshold_met"
    assert "Observed 2 events" in finding.rationale


def _make_manifest_authentication_engine(repository, provenance_validator):
    """Build an evaluator whose registry must never be reached on bad manifests."""
    evaluators = CriterionEvaluatorRegistry()
    outcomes = CriterionOutcomeRegistry()
    engine = CriterionEvaluationEngine(
        repository, evaluators, outcomes,
        provenance_validator=provenance_validator,
    )
    criterion = CriterionDefinition(
        criterion_id="manifest-authentication",
        criterion_type="manifest_authentication",
        parameters_json="{}",
        description="Exercise the direct EvidenceSet authentication boundary.",
        independence_assertion_id="manifest-authentication",
    )
    return engine, criterion


@pytest.mark.parametrize(
    "tamper",
    (
        "wrong_id",
        "missing_reference",
        "additional_reference",
    ),
)
def test_direct_criterion_evaluation_rejects_supplied_manifest_mismatch(
    tmp_path, monkeypatch, tamper
):
    repository, proposition, evidence_set, event, semantics_registry = make_evaluation_context(tmp_path)
    engine, criterion = _make_manifest_authentication_engine(
        repository, _test_provenance_validator(repository, semantics_registry)
    )
    if tamper == "wrong_id":
        supplied = replace(evidence_set, evidence_set_id=new_id())
    elif tamper == "missing_reference":
        supplied = replace(evidence_set, evidence=())
    else:
        supplied = replace(
            evidence_set,
            evidence=evidence_set.evidence
            + (EvidenceReference(observation_id=event.observation_id),),
        )

    def resolution_must_not_run(*args, **kwargs):
        raise AssertionError("evidence resolution ran before manifest authentication")

    monkeypatch.setattr(engine, "_resolve_evidence", resolution_must_not_run)
    with pytest.raises(
        ValueError,
        match="must exactly match the proposition evidence manifest",
    ):
        engine.evaluate(criterion, proposition, supplied)


def test_direct_criterion_evaluation_rejects_missing_persisted_evidence_set(
    tmp_path, monkeypatch
):
    repository, proposition, evidence_set, _, semantics_registry = make_evaluation_context(tmp_path)
    engine, criterion = _make_manifest_authentication_engine(
        repository, _test_provenance_validator(repository, semantics_registry)
    )
    monkeypatch.setattr(repository, "get_evidence_set", lambda evidence_set_id: None)

    def resolution_must_not_run(*args, **kwargs):
        raise AssertionError("evidence resolution ran before persisted manifest authentication")

    monkeypatch.setattr(engine, "_resolve_evidence", resolution_must_not_run)
    with pytest.raises(
        ValueError,
        match="must match its persisted historical selection",
    ):
        engine.evaluate(criterion, proposition, evidence_set)


def test_direct_criterion_evaluation_rejects_persisted_evidence_set_mismatch(
    tmp_path, monkeypatch
):
    repository, proposition, evidence_set, _, semantics_registry = make_evaluation_context(tmp_path)
    engine, criterion = _make_manifest_authentication_engine(
        repository, _test_provenance_validator(repository, semantics_registry)
    )
    altered_persisted = replace(evidence_set, evidence=())
    monkeypatch.setattr(
        repository,
        "get_evidence_set",
        lambda evidence_set_id: altered_persisted,
    )

    def resolution_must_not_run(*args, **kwargs):
        raise AssertionError("evidence resolution ran before persisted manifest authentication")

    monkeypatch.setattr(engine, "_resolve_evidence", resolution_must_not_run)
    with pytest.raises(
        ValueError,
        match="must match its persisted historical selection",
    ):
        engine.evaluate(criterion, proposition, evidence_set)



@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("event_role", "event role presence or value does not match manifest selection"),
        ("construction_version", "unknown proposition construction"),
        ("semantics_fingerprint", "fingerprint does not match"),
        ("temporal_scope", "temporal scope does not match"),
    ),
)
def test_direct_criterion_evaluation_rejects_forged_proposition_provenance(
    tmp_path, monkeypatch, tamper, message
):
    repository, proposition, evidence_set, event, semantics_registry = make_evaluation_context(tmp_path)
    if tamper == "event_role":
        forged_role = PropositionRoleValue(
            "success_event", "event", new_id(), event.occurred_at
        )
        candidate = replace(
            proposition,
            roles=(forged_role,),
            field_provenance=(
                PropositionFieldProvenance(
                    "success_event", (forged_role.value_id,), (event.observation_id,)
                ),
            ),
        )
    elif tamper == "construction_version":
        candidate = replace(proposition, construction_version=2)
    elif tamper == "semantics_fingerprint":
        candidate = replace(proposition, semantics_fingerprint="forged-semantics-fingerprint")
    else:
        candidate = replace(
            proposition,
            temporal_scope=PropositionTemporalScope(
                event.occurred_at - timedelta(seconds=1), event.occurred_at
            ),
        )

    candidate = replace(
        candidate,
        proposition_id=deterministic_proposition_id(
            candidate.proposition_type,
            candidate.semantics_version,
            candidate.roles,
            candidate.temporal_scope,
            candidate.parameters_json,
            semantics_fingerprint=candidate.semantics_fingerprint,
            identity_algorithm_version=candidate.identity_algorithm_version,
        ),
    )
    engine, criterion = _make_manifest_authentication_engine(
        repository, _test_provenance_validator(repository, semantics_registry)
    )

    def resolution_must_not_run(*args, **kwargs):
        raise AssertionError("evidence resolution ran before proposition provenance authentication")

    monkeypatch.setattr(engine, "_resolve_evidence", resolution_must_not_run)
    with pytest.raises(ValueError, match=message):
        engine.evaluate(criterion, candidate, evidence_set)



@pytest.mark.parametrize("tamper", ("parameter_value", "field_provenance"))
def test_direct_criterion_evaluation_rejects_forged_parameter_provenance(
    tmp_path, tamper
):
    repository, _, evidence_set, event, _ = make_evaluation_context(tmp_path)
    semantics = PropositionSemantics(
        proposition_type="parameter_provenance_test",
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
        parameters=(
            SemanticParameterDefinition(
                parameter_name="result",
                value_type="string",
                required=True,
                meaning="The normalized result recorded by the source event.",
            ),
        ),
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(semantics)
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(
        PropositionConstructionDefinition(
            construction_id="parameter-provenance-test",
            construction_version=1,
            proposition_type=semantics.proposition_type,
            semantics_version=semantics.semantics_version,
            event_bindings=(
                EventRoleBinding("success_event", "authentication_success"),
            ),
            entity_bindings=(),
            event_attribute_bindings=(
                EventAttributeParameterBinding(
                    "result",
                    "success_event",
                    "result",
                    "Bind the result parameter to the normalized event attribute.",
                ),
            ),
            description="Construct a result parameter from the source event.",
        )
    )
    proposition = PropositionConstructor(
        repository, semantics_registry, construction_registry
    ).construct(evidence_set, "parameter-provenance-test", 1)

    if tamper == "parameter_value":
        candidate = replace(proposition, parameters_json='{"result":"failure"}')
        expected_message = "parameter value does not match shared source selection"
    else:
        altered_provenance = replace(
            proposition.field_provenance[0],
            observation_ids=(new_id(),),
        )
        candidate = replace(
            proposition,
            field_provenance=(altered_provenance, *proposition.field_provenance[1:]),
        )
        expected_message = "event and observation provenance disagree"

    candidate = replace(
        candidate,
        proposition_id=deterministic_proposition_id(
            candidate.proposition_type,
            candidate.semantics_version,
            candidate.roles,
            candidate.temporal_scope,
            candidate.parameters_json,
            semantics_fingerprint=candidate.semantics_fingerprint,
            identity_algorithm_version=candidate.identity_algorithm_version,
        ),
    )
    engine, criterion = _make_manifest_authentication_engine(
        repository,
        PropositionProvenanceValidator(
            repository, semantics_registry, construction_registry
        ),
    )
    with pytest.raises(ValueError, match=expected_message):
        engine.evaluate(criterion, candidate, evidence_set)
