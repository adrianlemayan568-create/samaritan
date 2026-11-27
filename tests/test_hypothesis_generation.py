"""Integration tests for evidence-backed hypothesis generation."""

from dataclasses import replace
from datetime import timedelta

import pytest

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import (
    Event,
    EventParticipant,
    new_id,
)
from samaritan.correlation import CorrelationEngine
from samaritan.correlation_contracts import CorrelationRule
from samaritan.database import DatabaseIntegrityError
from samaritan.hypothesis_generation import HypothesisGenerationEngine
from samaritan.hypothesis_generation_contracts import (
    HypothesisGenerationRule,
    HypothesisPattern,
    PatternCorrelationConstraint,
    PatternEntityBinding,
    PatternEventRole,
    PatternTemporalConstraint,
)

from test_repository import (
    make_entity,
    make_observation,
    make_repository,
)


def persist_test_event(
    repository,
    source,
    entity,
    event_type,
    occurred_at,
):
    """Persist an event and its source observation."""
    observation = make_observation(source)

    event = Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type=event_type,
        occurred_at=occurred_at,
        original_timestamp=occurred_at.isoformat().replace(
            "+00:00", "Z"
        ),
        normalized_data_json="{}",
        participants=(
            EventParticipant(
                entity_id=entity.entity_id,
                role="subject",
            ),
        ),
    )

    repository.persist_observation(
        observation,
        entities=(entity,),
        events=(event,),
    )

    return event


def make_correlation_rules():
    """Return the two rules connecting the synthetic event sequence."""
    return (
        CorrelationRule(
            rule_id="failure-success",
            event_type_pairs=(
                ("authentication_failure", "authentication_success"),
            ),
            shared_entity_types=("user_account",),
            max_time_delta_seconds=300,
            description="Authentication failure followed by success",
        ),
        CorrelationRule(
            rule_id="success-host",
            event_type_pairs=(
                ("authentication_success", "host_activity"),
            ),
            shared_entity_types=("user_account",),
            max_time_delta_seconds=300,
            description="Successful authentication followed by host activity",
        ),
    )


def make_generation_rule(
    rule_id="suspicious-authentication-sequence",
    participant_role="subject",
):
    pattern = HypothesisPattern(
        pattern_id="test-authentication-sequence",
        version=1,
        event_roles=(
            PatternEventRole("failure", "authentication_failure"),
            PatternEventRole("success", "authentication_success"),
            PatternEventRole("host", "host_activity"),
        ),
        correlation_constraints=(
            PatternCorrelationConstraint("failure", "success", "failure-success"),
            PatternCorrelationConstraint("success", "host", "success-host"),
        ),
        entity_bindings=(
            PatternEntityBinding("account", "failure", "user_account", participant_role),
            PatternEntityBinding("account", "success", "user_account", participant_role),
            PatternEntityBinding("account", "host", "user_account", participant_role),
        ),
        temporal_constraints=(
            PatternTemporalConstraint("failure", "success", True, 300),
            PatternTemporalConstraint("success", "host", True, 300),
        ),
    )
    return HypothesisGenerationRule(
        rule_id=rule_id,
        pattern=pattern,
        statement="The observed sequence may indicate suspicious account activity",
        description=(
            "Authentication failure, subsequent success, and host activity "
            "share an account identity within the configured time windows."
        ),
    )


def make_event_chain(repository, source, entity, start_time):
    """Persist a three-event synthetic sequence."""
    return (
        persist_test_event(
            repository,
            source,
            entity,
            "authentication_failure",
            start_time,
        ),
        persist_test_event(
            repository,
            source,
            entity,
            "authentication_success",
            start_time + timedelta(minutes=1),
        ),
        persist_test_event(
            repository,
            source,
            entity,
            "host_activity",
            start_time + timedelta(minutes=2),
        ),
    )


def correlate_chain(repository, events):
    engine = CorrelationEngine(
        repository,
        make_correlation_rules(),
    )

    return engine.correlate_events(
        event.event_id for event in events
    )


def make_engine(repository, rules=None, correlation_rules=None):
    if rules is None:
        rules = (make_generation_rule(),)
    if correlation_rules is None:
        correlation_rules = make_correlation_rules()

    return HypothesisGenerationEngine(
        repository,
        rules,
        correlation_rules=correlation_rules,
    )


def test_generates_hypothesis_with_complete_evidence_provenance(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    assert len(correlations) == 2

    result = make_engine(repository).generate(correlations)

    assert len(result.hypotheses) == 1
    assert len(result.generations) == 1

    hypothesis = result.hypotheses[0]
    generation = result.generations[0]

    assert hypothesis.statement == (
        "The observed sequence may indicate suspicious account activity"
    )
    assert generation.hypothesis_id == hypothesis.hypothesis_id
    assert generation.rule_id == "suspicious-authentication-sequence"
    assert set(generation.correlation_ids) == {
        correlation.correlation_id
        for correlation in correlations
    }

    expected_evidence = {
        EvidenceReference(event_id=event.event_id)
        for event in events
    } | {
        EvidenceReference(observation_id=event.observation_id)
        for event in events
    }

    assert set(generation.evidence) == expected_evidence
    assert len(generation.evidence) == 6


def test_no_qualifying_pattern_returns_empty_result(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    rule = make_generation_rule(participant_role="missing-role")

    result = make_engine(repository, (rule,)).generate(correlations)

    assert result.hypotheses == ()
    assert result.generations == ()


def test_empty_correlations_return_empty_result(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    result = make_engine(repository).generate(())

    assert result.hypotheses == ()
    assert result.generations == ()


def test_multiple_disconnected_components_generate_separate_hypotheses(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)

    first_user = make_entity("user_account", "synthetic-user-1")
    second_user = make_entity("user_account", "synthetic-user-2")

    first_chain = make_event_chain(
        repository,
        source,
        first_user,
        source.created_at,
    )
    second_chain = make_event_chain(
        repository,
        source,
        second_user,
        source.created_at + timedelta(minutes=20),
    )

    correlations = correlate_chain(
        repository,
        first_chain + second_chain,
    )

    assert len(correlations) == 4

    result = make_engine(repository).generate(correlations)

    assert len(result.hypotheses) == 2
    assert len(result.generations) == 2

    generated_event_sets = []

    for generation in result.generations:
        generated_event_sets.append({
            reference.event_id
            for reference in generation.evidence
            if reference.event_id is not None
        })

    assert {
        frozenset(event.event_id for event in first_chain),
        frozenset(event.event_id for event in second_chain),
    } == {
        frozenset(event_ids)
        for event_ids in generated_event_sets
    }


def test_multiple_generation_rules_can_match_same_component(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    first_rule = make_generation_rule(
        rule_id="sequence-hypothesis",
    )
    second_rule = make_generation_rule(
        rule_id="alternative-sequence-hypothesis",
    )

    result = make_engine(
        repository,
        (first_rule, second_rule),
    ).generate(correlations)

    assert len(result.hypotheses) == 2
    assert len(result.generations) == 2
    assert {
        generation.rule_id
        for generation in result.generations
    } == {
        "sequence-hypothesis",
        "alternative-sequence-hypothesis",
    }

    assert len({
        hypothesis.hypothesis_id
        for hypothesis in result.hypotheses
    }) == 2


def test_missing_event_raises_integrity_error(
    tmp_path,
    monkeypatch,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    original_get_event = repository.get_event
    missing_id = events[0].event_id

    def get_event(event_id):
        if event_id == missing_id:
            return None
        return original_get_event(event_id)

    monkeypatch.setattr(repository, "get_event", get_event)

    with pytest.raises(
        DatabaseIntegrityError,
        match="referenced event does not exist",
    ):
        make_engine(repository).generate(correlations)


def test_missing_observation_raises_integrity_error(
    tmp_path,
    monkeypatch,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    original_get_observation = repository.get_observation
    missing_id = events[0].observation_id

    def get_observation(observation_id):
        if observation_id == missing_id:
            return None
        return original_get_observation(observation_id)

    monkeypatch.setattr(
        repository,
        "get_observation",
        get_observation,
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="referenced observation does not exist",
    ):
        make_engine(repository).generate(correlations)


def test_mismatched_event_observation_provenance_is_rejected(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    corrupted = replace(
        correlations[0],
        observation_ids=(
            new_id(),
            correlations[0].observation_ids[1],
        ),
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="does not match event provenance",
    ):
        make_engine(repository).generate((corrupted,))


def test_repeated_event_and_observation_records_are_retrieved_once(
    tmp_path,
    monkeypatch,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    original_get_event = repository.get_event
    original_get_observation = repository.get_observation

    event_calls = []
    observation_calls = []

    def counted_get_event(event_id):
        event_calls.append(event_id)
        return original_get_event(event_id)

    def counted_get_observation(observation_id):
        observation_calls.append(observation_id)
        return original_get_observation(observation_id)

    monkeypatch.setattr(repository, "get_event", counted_get_event)
    monkeypatch.setattr(
        repository,
        "get_observation",
        counted_get_observation,
    )

    result = make_engine(repository).generate(correlations)

    assert len(result.hypotheses) == 1
    assert len(event_calls) == 3
    assert len(observation_calls) == 3
    assert len(event_calls) == len(set(event_calls))
    assert len(observation_calls) == len(set(observation_calls))


def test_duplicate_correlation_ids_are_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    duplicate = replace(
        correlations[1],
        correlation_id=correlations[0].correlation_id,
    )

    with pytest.raises(
        ValueError,
        match="duplicate correlation IDs",
    ):
        make_engine(repository).generate(
            (correlations[0], duplicate)
        )


def test_duplicate_unordered_event_pairs_are_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    events = make_event_chain(
        repository,
        source,
        user,
        source.created_at,
    )
    correlations = correlate_chain(repository, events)

    duplicate_pair = replace(
        correlations[0],
        correlation_id=new_id(),
    )

    with pytest.raises(
        ValueError,
        match="duplicate unordered event pairs",
    ):
        make_engine(repository).generate(
            (correlations[0], duplicate_pair)
        )


def test_generation_rejects_string_as_correlation_collection(
    tmp_path,
):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(TypeError, match="iterable"):
        make_engine(repository).generate("invalid")


def test_generation_rejects_non_correlation_items(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(
        TypeError,
        match="Correlation objects",
    ):
        make_engine(repository).generate((object(),))


def test_engine_rejects_non_repository_argument():
    with pytest.raises(TypeError, match="Repository"):
        HypothesisGenerationEngine(
            "not a repository",
            (make_generation_rule(),),
            correlation_rules=make_correlation_rules(),
        )


def test_engine_rejects_empty_rules(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        HypothesisGenerationEngine(
            repository,
            (),
            correlation_rules=make_correlation_rules(),
        )


def test_engine_rejects_duplicate_generation_rule_ids(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    first = make_generation_rule(rule_id="duplicate-rule")
    second = make_generation_rule(rule_id="duplicate-rule")

    with pytest.raises(ValueError, match="duplicate"):
        HypothesisGenerationEngine(
            repository,
            (first, second),
            correlation_rules=make_correlation_rules(),
        )


def test_engine_rejects_non_generation_rule_items(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(TypeError, match="HypothesisGenerationRule"):
        HypothesisGenerationEngine(
            repository,
            (object(),),
            correlation_rules=make_correlation_rules(),
        )


def test_forged_shared_entity_claim_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")
    events = make_event_chain(repository, source, user, source.created_at)
    correlations = correlate_chain(repository, events)
    forged = replace(
        correlations[0],
        shared_entity_ids=(new_id(),),
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="shared entity claims do not match",
    ):
        make_engine(repository).generate((forged,))


def test_forged_time_delta_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")
    events = make_event_chain(repository, source, user, source.created_at)
    correlations = correlate_chain(repository, events)
    forged = replace(
        correlations[0],
        time_delta_seconds=correlations[0].time_delta_seconds + 1,
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="time delta does not match",
    ):
        make_engine(repository).generate((forged,))


def test_altered_correlation_rule_definition_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")
    events = make_event_chain(repository, source, user, source.created_at)
    correlations = correlate_chain(repository, events)
    altered_rule = replace(
        correlations[0].matched_rules[0],
        description="Altered rule definition",
    )
    forged = replace(
        correlations[0],
        matched_rules=(altered_rule,),
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="matched rules do not match",
    ):
        make_engine(repository).generate((forged,))


def test_fabricated_matching_rule_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")
    events = make_event_chain(repository, source, user, source.created_at)
    correlations = correlate_chain(repository, events)
    fabricated_rule = CorrelationRule(
        rule_id="fabricated-rule",
        event_type_pairs=(("authentication_failure", "authentication_success"),),
        shared_entity_types=("user_account",),
        max_time_delta_seconds=300,
        description="Fabricated matching rule",
    )
    forged = replace(
        correlations[0],
        matched_rules=(fabricated_rule,),
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="matched rules do not match",
    ):
        make_engine(repository).generate((forged,))


def test_invalid_irrelevant_correlation_is_rejected_before_pattern_matching(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")
    events = make_event_chain(repository, source, user, source.created_at)
    correlations = correlate_chain(repository, events)
    forged = replace(
        correlations[0],
        shared_entity_ids=(new_id(),),
    )
    irrelevant_rule = make_generation_rule(participant_role="missing-role")

    with pytest.raises(
        DatabaseIntegrityError,
        match="shared entity claims do not match",
    ):
        make_engine(repository, (irrelevant_rule,)).generate((forged,))


def test_engine_rejects_pattern_constraint_without_configured_rule(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    unrelated_rule = CorrelationRule(
        rule_id="unrelated-rule",
        event_type_pairs=(("unrelated_a", "unrelated_b"),),
        shared_entity_types=("user_account",),
        max_time_delta_seconds=300,
        description="Does not implement the declared pattern edge",
    )

    with pytest.raises(ValueError, match="unconfigured correlation rule"):
        HypothesisGenerationEngine(
            repository,
            (make_generation_rule(),),
            correlation_rules=(unrelated_rule,),
        )


def test_correlation_without_configured_match_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "unconfigured-correlation-account")
    valid_events = make_event_chain(repository, source, account, source.created_at)
    unrelated_events = (
        persist_test_event(
            repository, source, account, "unrelated_a",
            source.created_at + timedelta(minutes=10)
        ),
        persist_test_event(
            repository, source, account, "unrelated_b",
            source.created_at + timedelta(minutes=10, seconds=30)
        ),
    )
    unrelated_rule = CorrelationRule(
        rule_id="unconfigured-rule",
        event_type_pairs=(("unrelated_a", "unrelated_b"),),
        shared_entity_types=("user_account",),
        max_time_delta_seconds=300,
        description="An unrelated pair used to verify C-01 fail-closed behavior.",
    )
    unrelated_correlation = CorrelationEngine(
        repository, (unrelated_rule,)
    ).correlate_events(event.event_id for event in unrelated_events)

    with pytest.raises(
        DatabaseIntegrityError,
        match="no match under the configured correlation rules",
    ):
        make_engine(repository).generate(
            correlate_chain(repository, valid_events) + unrelated_correlation
        )


def test_engine_rejects_empty_correlation_rules(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(ValueError, match="at least one correlation rule"):
        HypothesisGenerationEngine(
            repository,
            (make_generation_rule(),),
            correlation_rules=(),
        )


def test_engine_rejects_duplicate_correlation_rule_ids(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    rule = make_correlation_rules()[0]

    with pytest.raises(ValueError, match="duplicate correlation rule IDs"):
        HypothesisGenerationEngine(
            repository,
            (make_generation_rule(),),
            correlation_rules=(rule, rule),
        )


def test_engine_rejects_non_correlation_rule_items(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(TypeError, match="CorrelationRule"):
        HypothesisGenerationEngine(
            repository,
            (make_generation_rule(),),
            correlation_rules=(object(),),
        )
