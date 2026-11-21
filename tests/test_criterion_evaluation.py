from dataclasses import replace
from datetime import datetime, timedelta, timezone

from samaritan.analysis_contracts import EvidenceReference
from samaritan.criterion_evaluation import CriterionEvaluationEngine
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

    mismatched_evidence = replace(evidence_set, evidence=())
    import pytest
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
    import pytest
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
    import pytest
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


def test_temporal_gap_reports_ordering_violation_separately(tmp_path):
    from samaritan.proposition_contracts import PropositionEvidenceManifest

    repository, proposition, _, original_event, _ = make_evaluation_context(tmp_path)
    now = original_event.occurred_at
    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="temporal ordering test",
        created_at=now,
    )
    repository.register_source(source)
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
        event_type="authentication_success",
        occurred_at=now + timedelta(seconds=30),
        original_timestamp=(now + timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
        normalized_data_json="{}",
        participants=(),
    )
    repository.persist_observation(later_observation, events=(later_event,))
    roles = (
        PropositionRoleValue("earlier_event", "event", later_event.event_id, later_event.occurred_at),
        PropositionRoleValue("later_event", "event", original_event.event_id, original_event.occurred_at),
    )
    scope = PropositionTemporalScope(now, later_event.occurred_at)
    provenance = (
        PropositionFieldProvenance(
            "earlier_event", (later_event.event_id,), (later_observation.observation_id,)
        ),
        PropositionFieldProvenance(
            "later_event", (original_event.event_id,), (original_event.observation_id,)
        ),
    )
    evidence = (
        EvidenceReference(event_id=original_event.event_id),
        EvidenceReference(event_id=later_event.event_id),
    )
    proposition = Proposition(
        proposition_id=deterministic_proposition_id(
            proposition.proposition_type,
            proposition.semantics_version,
            roles,
            scope,
            semantics_fingerprint=proposition.semantics_fingerprint,
            identity_algorithm_version=proposition.identity_algorithm_version,
        ),
        proposition_type=proposition.proposition_type,
        semantics_version=proposition.semantics_version,
        roles=roles,
        temporal_scope=scope,
        field_provenance=provenance,
        construction_id=proposition.construction_id,
        construction_version=proposition.construction_version,
        semantics_fingerprint=proposition.semantics_fingerprint,
        identity_algorithm_version=proposition.identity_algorithm_version,
        evidence_manifest=PropositionEvidenceManifest(
            evidence_set_id=new_id(), evidence=evidence
        ),
    )
    selected = EvidenceSet(
        evidence_set_id=proposition.evidence_manifest.evidence_set_id,
        evidence=evidence,
        created_at=now + timedelta(minutes=1),
    )
    parameters = (
        CriterionParameterDefinition("earlier_event_role", "string", required=True),
        CriterionParameterDefinition("later_event_role", "string", required=True),
        CriterionParameterDefinition("maximum_seconds", "integer", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="strict-temporal-order",
        criterion_type="temporal_order",
        parameters_json='{"earlier_event_role":"earlier_event","later_event_role":"later_event","maximum_seconds":60}',
        description="The earlier event must strictly precede the later event.",
        independence_group="temporal-order-source",
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

    finding = CriterionEvaluationEngine(repository, evaluators, outcomes).evaluate(
        criterion, proposition, selected
    )

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
    alice = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="alice-count-scope", created_at=now,
    )
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
        independence_group="account-event-source",
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

    finding = CriterionEvaluationEngine(repository, evaluators, outcomes).evaluate(
        criterion, proposition, evidence_set
    )

    assert finding.status == "matched"
    assert finding.outcome_code == "threshold_met"
    assert "Observed 2 events" in finding.rationale
