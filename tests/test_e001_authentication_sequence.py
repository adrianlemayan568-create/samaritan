"""E001: narrow empirical validation of account authentication sequences.

Synthetic-only experiment. The registered neutral bearings are reviewed
experimental policy, not evidence and not a production-approved rule.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from time import perf_counter

import pytest

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import Source, new_id
from samaritan.correlation import CorrelationEngine
from samaritan.correlation_contracts import CorrelationRule
from samaritan.database import Database
from samaritan.evaluation_contracts import EvidenceSet
from samaritan.hypothesis_evaluation import HypothesisEvaluationEngine
from samaritan.hypothesis_generation import HypothesisGenerationEngine
from samaritan.hypothesis_generation_contracts import (
    HypothesisGenerationRule,
    HypothesisPattern,
    PatternCorrelationConstraint,
    PatternEntityBinding,
    PatternEventRole,
    PatternTemporalConstraint,
)
from samaritan.ingestion import IngestionService
from samaritan.proposition_construction import (
    PropositionConstructionError,
    PropositionConstructor,
)
from samaritan.proposition_construction_contracts import (
    build_default_proposition_construction_registry,
)
from samaritan.proposition_contracts import StructuredHypothesis
from samaritan.proposition_semantics import (
    build_default_proposition_semantics_registry,
)
from samaritan.reasoning_contracts import (
    BearingDefinition,
    CriterionDefinition,
    CriterionEvaluatorRegistry,
    CriterionEvaluatorDefinition,
    CriterionOutcomeDefinition,
    CriterionOutcomeRegistry,
    CriterionParameterDefinition,
    HypothesisAlternative,
    HypothesisEvaluationRule,
    HypothesisEvaluationRuleRegistry,
    HypothesisSpecification,
    RESIDUAL_ALTERNATIVE_ID,
)
from samaritan.repository import Repository


NOW = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)
FIXED_EVALUATION_TIME = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def _context(tmp_path):
    database = Database(str(tmp_path / "e001.db"))
    database.initialize()
    repository = Repository(database)
    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="E001 synthetic authentication telemetry",
        created_at=NOW,
    )
    repository.register_source(source)
    return repository, source, IngestionService(repository)


def _ingest_specs(service, source, specs):
    """Ingest one strict JSON envelope and retain exact event/observation IDs."""
    entity_ids = {}
    for spec in specs:
        participants = [
            ("user_account", spec["account"], "target" if spec["event_type"] != "host_activity" else "subject"),
        ]
        if spec.get("ip") is not None:
            participants.append(("ip_address", spec["ip"], "source"))
        if spec["event_type"] == "host_activity":
            participants.append(("hostname", "synthetic-host.example.test", "target"))
        for entity_type, identifier, _ in participants:
            entity_ids.setdefault((entity_type, identifier), new_id())

    entities = [
        {
            "entity_id": entity_id,
            "entity_type": key[0],
            "canonical_identifier": key[1],
        }
        for key, entity_id in entity_ids.items()
    ]
    event_ids_by_type = {}
    events = []
    for spec in specs:
        event_id = new_id()
        event_ids_by_type.setdefault(spec["event_type"], []).append(event_id)
        participants = [
            {
                "entity_id": entity_ids[(entity_type, identifier)],
                "role": role,
            }
            for entity_type, identifier, role in (
                [("user_account", spec["account"], "target" if spec["event_type"] != "host_activity" else "subject")]
                + ([("ip_address", spec["ip"], "source")] if spec.get("ip") is not None else [])
                + ([("hostname", "synthetic-host.example.test", "target")] if spec["event_type"] == "host_activity" else [])
            )
        ]
        occurred_at = NOW + timedelta(minutes=spec["minute"])
        events.append({
            "event_id": event_id,
            "event_type": spec["event_type"],
            "original_timestamp": occurred_at.isoformat().replace("+00:00", "Z"),
            "normalized_data": spec["data"],
            "participants": participants,
        })

    envelope = {
        "format_version": 1,
        "entities": entities,
        "events": events,
        "relationships": [],
    }
    raw_payload = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    received_at = NOW + timedelta(minutes=max(spec["minute"] for spec in specs) + 1)
    result = service.ingest(source.source_id, raw_payload, received_at)
    assert result.accepted is True, result.rejection_reason
    return event_ids_by_type, result.observation_id


def _sequence(account="account-A", ip="192.0.2.10", failure_minutes=(0,), success_minute=2, host_minute=4, success_data=None):
    if success_data is None:
        success_data = {"result": "success"}
    specs = [
        {
            "event_type": "authentication_failure",
            "minute": minute,
            "account": account,
            "ip": ip,
            "data": {"result": "failure"},
        }
        for minute in failure_minutes
    ]
    specs.extend((
        {
            "event_type": "authentication_success",
            "minute": success_minute,
            "account": account,
            "ip": ip,
            "data": success_data,
        },
        {
            "event_type": "host_activity",
            "minute": host_minute,
            "account": account,
            "ip": ip,
            "data": {"activity": "interactive_session_started"},
        },
    ))
    return specs


def _correlation_rules():
    return (
        CorrelationRule(
            rule_id="e001-failure-success",
            event_type_pairs=(("authentication_failure", "authentication_success"),),
            shared_entity_types=("user_account", "ip_address"),
            max_time_delta_seconds=300,
            description="Failure and success share an account or network identity within five minutes.",
        ),
        CorrelationRule(
            rule_id="e001-success-host",
            event_type_pairs=(("authentication_success", "host_activity"),),
            shared_entity_types=("user_account", "ip_address"),
            max_time_delta_seconds=300,
            description="Success and host activity share an account or network identity within five minutes.",
        ),
    )


def _authentication_pattern():
    return HypothesisPattern(
        pattern_id="account_authentication_sequence",
        version=1,
        event_roles=(
            PatternEventRole("failure_event", "authentication_failure"),
            PatternEventRole("success_event", "authentication_success"),
            PatternEventRole("host_activity_event", "host_activity"),
        ),
        correlation_constraints=(
            PatternCorrelationConstraint(
                "failure_event", "success_event", "e001-failure-success"
            ),
            PatternCorrelationConstraint(
                "success_event", "host_activity_event", "e001-success-host"
            ),
        ),
        entity_bindings=(
            PatternEntityBinding(
                "account", "failure_event", "user_account", "target"
            ),
            PatternEntityBinding(
                "account", "success_event", "user_account", "target"
            ),
            PatternEntityBinding(
                "account", "host_activity_event", "user_account", "subject"
            ),
        ),
        temporal_constraints=(
            PatternTemporalConstraint(
                "failure_event", "success_event", True, 300
            ),
            PatternTemporalConstraint(
                "success_event", "host_activity_event", True, 300
            ),
        ),
    )


def _generation_rule():
    return HypothesisGenerationRule(
        rule_id="e001-authentication-sequence",
        pattern=_authentication_pattern(),
        statement="The selected events form a candidate account authentication sequence.",
        description="A failure, success, and host activity share an account identity in the configured time windows.",
    )


def _pipeline(repository, event_ids_by_type, record_property=None):
    event_ids = tuple(
        event_id
        for values in event_ids_by_type.values()
        for event_id in values
    )
    started = perf_counter()
    correlations = CorrelationEngine(repository, _correlation_rules()).correlate_events(event_ids)
    correlation_seconds = perf_counter() - started

    generation_rule = _generation_rule()
    started = perf_counter()
    generated = HypothesisGenerationEngine(
        repository,
        (generation_rule,),
        correlation_rules=_correlation_rules(),
    ).generate(correlations)
    generation_seconds = perf_counter() - started

    if record_property is not None:
        record_property("e001.correlation_seconds", correlation_seconds)
        record_property("e001.hypothesis_generation_seconds", generation_seconds)
        record_property("e001.correlation_count", len(correlations))
        record_property("e001.generated_hypothesis_count", len(generated.hypotheses))
    return correlations, generated


def _evidence_set(event_ids, observation_id=None):
    refs = [EvidenceReference(event_id=item) for item in event_ids]
    if observation_id is not None:
        refs.append(EvidenceReference(observation_id=observation_id))
    return EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(refs),
        created_at=NOW,
    )


def _constructor(repository):
    semantics = build_default_proposition_semantics_registry()
    construction = build_default_proposition_construction_registry()
    return PropositionConstructor(repository, semantics, construction), semantics


def _all_event_ids(event_ids_by_type):
    return tuple(
        event_id
        for event_ids in event_ids_by_type.values()
        for event_id in event_ids
    )


def _evaluate_neutrally(repository, semantics, focal, focal_set, alternative, generated_hypothesis):
    params = (
        CriterionParameterDefinition("event_role", "string", required=True),
        CriterionParameterDefinition("attribute_name", "string", required=True),
        CriterionParameterDefinition("expected_value", "string", required=True),
    )
    criterion = CriterionDefinition(
        criterion_id="e001-authentication-result",
        criterion_type="e001_authentication_result",
        parameters_json='{"event_role":"success_event","attribute_name":"result","expected_value":"success"}',
        description="The normalized result attribute on the selected success event equals success.",
        independence_assertion_id="e001-authentication-result-assertion",
        parameter_definitions=params,
    )
    evaluators = CriterionEvaluatorRegistry()
    evaluators.register(CriterionEvaluatorDefinition(
        criterion_type=criterion.criterion_type,
        evaluator_version=1,
        evaluator_type="event_attribute_equals",
        parameter_definitions=params,
        description="Compare one registered normalized event attribute with its expected value.",
    ))
    outcomes = CriterionOutcomeRegistry()
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="equal",
        meaning="The normalized attribute equals the expected value.",
    ))
    outcomes.register(CriterionOutcomeDefinition(
        criterion_type=criterion.criterion_type,
        outcome_code="not_equal",
        meaning="The normalized attribute differs from the expected value.",
    ))

    bearings = tuple(
        BearingDefinition(
            criterion_id=criterion.criterion_id,
            outcome_code=outcome,
            alternative_id=alternative_id,
            focal_judgment="compatible",
            alternative_judgment="compatible",
            evaluability_role="required",
            rationale="This experimental bearing deliberately does not discriminate.",
            basis="Neutral E001 policy: C-05A records alternative-specific findings but does not yet interpret them comparatively in synthesis.",
            author="E001 experimental protocol",
            review_status="reviewed",
        )
        for outcome in ("equal", "not_equal")
        for alternative_id in (alternative.alternative_id, RESIDUAL_ALTERNATIVE_ID)
    )
    rule = HypothesisEvaluationRule(
        evaluation_rule_id="e001-neutral-authentication-rule",
        rule_version=1,
        proposition_type=focal.proposition_type,
        semantics_version=focal.semantics_version,
        semantics_fingerprint=focal.semantics_fingerprint,
        criterion_definitions=(criterion,),
        bearing_definitions=bearings,
        description="Experimental authentication-result evaluation with neutral comparative bearings.",
        review_status="reviewed",
    )
    registry = HypothesisEvaluationRuleRegistry()
    registry.register(rule)
    specification = HypothesisSpecification(
        hypothesis=StructuredHypothesis(
            hypothesis_id=generated_hypothesis.hypothesis_id,
            proposition=focal,
            created_at=generated_hypothesis.created_at,
        ),
        alternatives=(alternative,),
        evaluation_rule=rule,
    )
    engine = HypothesisEvaluationEngine(
        repository,
        evaluators,
        outcomes,
        registry,
        construction_registry=build_default_proposition_construction_registry(),
        semantics_registry=semantics,
        clock=lambda: FIXED_EVALUATION_TIME,
    )
    return engine, specification


def test_e001_case_a_valid_sequence_constructs_and_synthesizes_neutrally(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    specs = _sequence()
    event_ids, observation_id = _ingest_specs(service, source, specs)
    correlations, generated = _pipeline(repository, event_ids, record_property)

    assert len(correlations) == 2
    assert len(generated.hypotheses) == len(generated.generations) == 1
    generation = generated.generations[0]
    expected_refs = {
        EvidenceReference(event_id=item)
        for item in _all_event_ids(event_ids)
    } | {EvidenceReference(observation_id=observation_id)}
    assert generation.hypothesis_id == generated.hypotheses[0].hypothesis_id
    assert set(generation.evidence) == expected_refs

    evidence_set = _evidence_set(_all_event_ids(event_ids), observation_id)
    constructor, semantics = _constructor(repository)
    started = perf_counter()
    focal = constructor.construct(evidence_set, "account_authentication_sequence", 1)
    construction_seconds = perf_counter() - started
    assert focal.proposition_type == "account_authentication_sequence"
    assert {item.field_name for item in focal.field_provenance} >= {
        "account", "failure_event", "success_event", "host_activity_event"
    }
    assert set(
        ref.observation_id
        for field in focal.field_provenance
        for ref in (EvidenceReference(observation_id=value) for value in field.observation_ids)
    ) == {observation_id}
    repeated = constructor.construct(evidence_set, "account_authentication_sequence", 1)
    assert focal.proposition_id == repeated.proposition_id

    alt_ids, alt_observation_id = _ingest_specs(
        service, source, _sequence(account="account-B", ip="198.51.100.20")
    )
    alternative_set = _evidence_set(_all_event_ids(alt_ids), alt_observation_id)
    alternative = constructor.construct(
        alternative_set, "account_authentication_sequence", 1
    )
    alternative_contract = HypothesisAlternative(
        alternative_id=new_id(),
        proposition=alternative,
        description="A separately constructed valid sequence for a different account.",
    )
    engine, specification = _evaluate_neutrally(
        repository, semantics, focal, evidence_set, alternative_contract,
        generated.hypotheses[0],
    )
    started = perf_counter()
    record = engine.evaluate(specification, evidence_set, production=False)
    evaluation_seconds = perf_counter() - started
    repeated_record = engine.evaluate(specification, evidence_set, production=False)
    assert record.evaluation_id == repeated_record.evaluation_id
    assert record.findings[0].finding_id == repeated_record.findings[0].finding_id
    assert record.findings[0].status == "matched"
    assert record.findings[0].outcome_code == "equal"
    assert record.synthesis.outcome == "inconclusive"
    assert record.synthesis.confidence_ceiling == "low"
    assert all(
        standing.standing == "non_discriminating"
        for standing in record.synthesis.alternative_standings
    )
    record_property("e001.proposition_construction_seconds", construction_seconds)
    record_property("e001.evaluation_seconds", evaluation_seconds)
    record_property("e001.total_event_count", len(_all_event_ids(event_ids)))
    record_property("e001.total_observation_count", 2)


def test_e001_case_b_different_accounts_are_not_combined(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    specs = [
        {"event_type": "authentication_failure", "minute": 0, "account": "account-A", "ip": None, "data": {"result": "failure"}},
        {"event_type": "authentication_success", "minute": 2, "account": "account-B", "ip": None, "data": {"result": "success"}},
        {"event_type": "host_activity", "minute": 4, "account": "account-B", "ip": None, "data": {"activity": "session"}},
    ]
    event_ids, observation_id = _ingest_specs(service, source, specs)
    correlations, generated = _pipeline(repository, event_ids, record_property)
    assert len(correlations) == 1
    assert generated.hypotheses == ()
    constructor, _ = _constructor(repository)
    with pytest.raises(PropositionConstructionError, match="entity role account resolves to different entities"):
        constructor.construct(
            _evidence_set(_all_event_ids(event_ids), observation_id),
            "account_authentication_sequence",
            1,
        )


def test_e001_case_c_reversed_temporal_order_is_rejected(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    event_ids, observation_id = _ingest_specs(
        service, source,
        _sequence(failure_minutes=(2,), success_minute=0, host_minute=4),
    )
    correlations, generated = _pipeline(repository, event_ids, record_property)
    assert len(correlations) == 2
    assert generated.hypotheses == ()
    constructor, _ = _constructor(repository)
    with pytest.raises(PropositionConstructionError):
        constructor.construct(
            _evidence_set(_all_event_ids(event_ids), observation_id),
            "account_authentication_sequence",
            1,
        )


def test_e001_case_d_missing_required_event_is_rejected(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    specs = _sequence(host_minute=4)[:]
    specs = [item for item in specs if item["event_type"] != "host_activity"]
    event_ids, observation_id = _ingest_specs(service, source, specs)
    correlations, generated = _pipeline(repository, event_ids, record_property)
    assert len(correlations) == 1
    assert generated.hypotheses == ()
    constructor, _ = _constructor(repository)
    with pytest.raises(PropositionConstructionError, match="expected exactly one host_activity event"):
        constructor.construct(
            _evidence_set(_all_event_ids(event_ids), observation_id),
            "account_authentication_sequence",
            1,
        )


def test_e001_case_e_multiple_matching_failures_are_ambiguous(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    event_ids, observation_id = _ingest_specs(
        service, source,
        _sequence(failure_minutes=(0, 1), success_minute=2, host_minute=4),
    )
    correlations, generated = _pipeline(repository, event_ids, record_property)
    assert len(correlations) == 3
    assert generated.hypotheses == ()
    constructor, _ = _constructor(repository)
    with pytest.raises(PropositionConstructionError, match="event-role selection is ambiguous"):
        constructor.construct(
            _evidence_set(_all_event_ids(event_ids), observation_id),
            "account_authentication_sequence",
            1,
        )


def test_e001_case_f_shared_ip_does_not_establish_shared_account(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    specs = [
        {"event_type": "authentication_failure", "minute": 0, "account": "account-A", "ip": "203.0.113.7", "data": {"result": "failure"}},
        {"event_type": "authentication_success", "minute": 2, "account": "account-B", "ip": "203.0.113.7", "data": {"result": "success"}},
        {"event_type": "host_activity", "minute": 4, "account": "account-B", "ip": "203.0.113.7", "data": {"activity": "session"}},
    ]
    event_ids, observation_id = _ingest_specs(service, source, specs)
    correlations, generated = _pipeline(repository, event_ids, record_property)
    assert len(correlations) == 2
    assert generated.hypotheses == ()
    constructor, _ = _constructor(repository)
    with pytest.raises(PropositionConstructionError, match="entity role account resolves to different entities"):
        constructor.construct(
            _evidence_set(_all_event_ids(event_ids), observation_id),
            "account_authentication_sequence",
            1,
        )


def test_e001_case_g_missing_normalized_attribute_is_insufficient(tmp_path, record_property):
    repository, source, service = _context(tmp_path)
    event_ids, observation_id = _ingest_specs(
        service, source, _sequence(success_data={"username": "account-A"})
    )
    correlations, generated = _pipeline(repository, event_ids, record_property)
    assert len(correlations) == 2
    assert len(generated.hypotheses) == 1
    evidence_set = _evidence_set(_all_event_ids(event_ids), observation_id)
    constructor, semantics = _constructor(repository)
    focal = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )

    alt_ids, alt_observation_id = _ingest_specs(
        service, source, _sequence(account="account-B", ip="198.51.100.21")
    )
    alternative = constructor.construct(
        _evidence_set(_all_event_ids(alt_ids), alt_observation_id),
        "account_authentication_sequence",
        1,
    )
    alternative_contract = HypothesisAlternative(
        alternative_id=new_id(),
        proposition=alternative,
        description="A separately constructed valid sequence for a different account.",
    )
    engine, specification = _evaluate_neutrally(
        repository, semantics, focal, evidence_set, alternative_contract,
        generated.hypotheses[0],
    )
    record = engine.evaluate(specification, evidence_set, production=False)
    finding = record.findings[0]
    assert finding.status == "insufficient"
    assert finding.outcome_code is None
    assert finding.missing_information
    assert record.synthesis.outcome == "inconclusive"
    assert record.synthesis.confidence_ceiling == "low"
