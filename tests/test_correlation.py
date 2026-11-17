"""Integration tests for evidence-preserving event correlation."""

from datetime import timedelta

import pytest

from samaritan.contracts import (
    Event,
    EventParticipant,
    new_id,
)
from samaritan.correlation import CorrelationEngine
from samaritan.correlation_contracts import CorrelationRule
from samaritan.database import DatabaseIntegrityError

from test_repository import (
    make_entity,
    make_observation,
    make_repository,
)


def make_test_event(
    repository,
    source,
    entities,
    event_type,
    occurred_at,
):
    """Persist one event and its observation using the real repository."""
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
        participants=tuple(
            EventParticipant(
                entity_id=entity.entity_id,
                role="source",
            )
            for entity in entities
        ),
    )

    repository.persist_observation(
        observation,
        entities=tuple(entities),
        events=(event,),
    )

    return event


def make_rule(
    rule_id="test-rule",
    event_types=("authentication_failure", "authentication_success"),
    entity_types=("user_account",),
    max_seconds=300,
):
    return CorrelationRule(
        rule_id=rule_id,
        event_type_pairs=(event_types,),
        shared_entity_types=entity_types,
        max_time_delta_seconds=max_seconds,
        description=f"Test rule: {rule_id}",
    )


def make_engine(repository, rules=None):
    if rules is None:
        rules = (make_rule(),)

    return CorrelationEngine(repository, rules)


def test_correlates_events_sharing_an_allowed_entity(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)

    ip = make_entity("ip_address", "192.0.2.10")
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at
    second_time = first_time + timedelta(minutes=5)

    first = make_test_event(
        repository,
        source,
        (ip, user),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (ip, user),
        "authentication_success",
        second_time,
    )

    engine = make_engine(repository)
    results = engine.correlate_pair(first.event_id, second.event_id)

    assert len(results) == 1

    correlation = results[0]

    assert correlation.event_ids == (
        first.event_id,
        second.event_id,
    )
    assert correlation.observation_ids == (
        first.observation_id,
        second.observation_id,
    )
    assert correlation.shared_entity_ids == (user.entity_id,)
    assert correlation.matched_rules[0].rule_id == "test-rule"
    assert correlation.time_delta_seconds == 300
    assert "does not establish causation" in correlation.explanation


def test_event_type_matching_is_order_independent(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    failure = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        first_time,
    )
    success = make_test_event(
        repository,
        source,
        (user,),
        "authentication_success",
        first_time + timedelta(seconds=30),
    )

    engine = make_engine(repository)

    results = engine.correlate_pair(
        success.event_id,
        failure.event_id,
    )

    assert len(results) == 1
    assert results[0].event_ids == (
        success.event_id,
        failure.event_id,
    )


def test_accepts_exact_time_window_boundary(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    first = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (user,),
        "authentication_success",
        first_time + timedelta(seconds=300),
    )

    results = make_engine(repository).correlate_pair(
        first.event_id,
        second.event_id,
    )

    assert len(results) == 1
    assert results[0].time_delta_seconds == 300


def test_rejects_events_outside_time_window(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    first = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (user,),
        "authentication_success",
        first_time + timedelta(seconds=301),
    )

    results = make_engine(repository).correlate_pair(
        first.event_id,
        second.event_id,
    )

    assert results == ()


def test_rejects_shared_entity_of_disallowed_type(tmp_path):
    repository, _, source = make_repository(tmp_path)
    ip = make_entity("ip_address", "192.0.2.10")

    first_time = source.created_at

    first = make_test_event(
        repository,
        source,
        (ip,),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (ip,),
        "authentication_success",
        first_time + timedelta(seconds=30),
    )

    rule = make_rule(entity_types=("user_account",))
    results = make_engine(repository, (rule,)).correlate_pair(
        first.event_id,
        second.event_id,
    )

    assert results == ()


def test_multiple_matching_rules_are_attached_to_one_correlation(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)

    ip = make_entity("ip_address", "192.0.2.10")
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    first = make_test_event(
        repository,
        source,
        (ip, user),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (ip, user),
        "authentication_success",
        first_time + timedelta(seconds=60),
    )

    rules = (
        make_rule(
            rule_id="user-rule",
            entity_types=("user_account",),
        ),
        make_rule(
            rule_id="ip-rule",
            entity_types=("ip_address",),
        ),
    )

    results = make_engine(repository, rules).correlate_pair(
        first.event_id,
        second.event_id,
    )

    assert len(results) == 1
    assert {
        rule.rule_id for rule in results[0].matched_rules
    } == {"user-rule", "ip-rule"}

    assert results[0].shared_entity_ids == (
        ip.entity_id,
        user.entity_id,
    )


def test_nonmatching_event_types_produce_no_correlation(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    first = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (user,),
        "host_activity",
        first_time + timedelta(seconds=30),
    )

    results = make_engine(repository).correlate_pair(
        first.event_id,
        second.event_id,
    )

    assert results == ()


def test_batch_correlation_analyzes_each_unique_pair_once(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    events = tuple(
        make_test_event(
            repository,
            source,
            (user,),
            "authentication_failure",
            first_time + timedelta(seconds=offset),
        )
        for offset in (0, 60, 120)
    )

    rule = make_rule(
        event_types=(
            "authentication_failure",
            "authentication_failure",
        ),
        max_seconds=300,
    )

    engine = make_engine(repository, (rule,))

    results = engine.correlate_events(
        (
            events[0].event_id,
            events[1].event_id,
            events[0].event_id,
            events[2].event_id,
        )
    )

    assert len(results) == 3

    assert tuple(
        result.event_ids for result in results
    ) == (
        (events[0].event_id, events[1].event_id),
        (events[0].event_id, events[2].event_id),
        (events[1].event_id, events[2].event_id),
    )


def test_batch_with_fewer_than_two_events_returns_empty_tuple(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    event = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        source.created_at,
    )

    engine = make_engine(repository)

    assert engine.correlate_events(()) == ()
    assert engine.correlate_events((event.event_id,)) == ()
    assert engine.correlate_events(
        (event.event_id, event.event_id)
    ) == ()


def test_missing_event_raises_integrity_error(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    engine = make_engine(repository)

    with pytest.raises(
        DatabaseIntegrityError,
        match="referenced event does not exist",
    ):
        engine.correlate_pair(new_id(), new_id())


def test_invalid_event_id_is_rejected(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    engine = make_engine(repository)

    with pytest.raises(ValueError):
        engine.correlate_pair("invalid-id", new_id())


def test_correlating_event_with_itself_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    event = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        source.created_at,
    )

    engine = make_engine(repository)

    with pytest.raises(ValueError, match="distinct events"):
        engine.correlate_pair(event.event_id, event.event_id)


def test_duplicate_rule_ids_are_rejected(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    rule = make_rule()

    with pytest.raises(ValueError, match="duplicate"):
        make_engine(repository, (rule, rule))


def test_engine_rejects_non_repository_argument():
    with pytest.raises(TypeError, match="Repository"):
        CorrelationEngine(object(), (make_rule(),))


def test_engine_rejects_empty_rules(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        CorrelationEngine(repository, ())


def test_batch_rejects_string_as_event_collection(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    engine = make_engine(repository)

    with pytest.raises(TypeError, match="iterable"):
        engine.correlate_events("not-an-event-collection")


def test_missing_participant_entity_raises_integrity_error(
    tmp_path,
    monkeypatch,
):
    repository, _, source = make_repository(tmp_path)
    user = make_entity("user_account", "synthetic-user")

    first_time = source.created_at

    first = make_test_event(
        repository,
        source,
        (user,),
        "authentication_failure",
        first_time,
    )
    second = make_test_event(
        repository,
        source,
        (user,),
        "authentication_success",
        first_time + timedelta(seconds=30),
    )

    def return_no_entities(entity_ids):
        return ()

    monkeypatch.setattr(
        repository,
        "get_entities_by_ids",
        return_no_entities,
    )

    engine = make_engine(repository)

    with pytest.raises(
        DatabaseIntegrityError,
        match="participant entities do not exist",
    ):
        engine.correlate_pair(first.event_id, second.event_id)


def test_batch_over_limit_is_rejected_before_event_retrieval(
    tmp_path,
    monkeypatch,
):
    repository, _, _ = make_repository(tmp_path)
    engine = make_engine(repository)

    def unexpected_event_lookup(event_id):
        raise AssertionError(
            "event retrieval must not occur for an oversized batch"
        )

    monkeypatch.setattr(
        repository,
        "get_event",
        unexpected_event_lookup,
    )

    event_ids = tuple(new_id() for _ in range(101))

    with pytest.raises(ValueError, match="100 unique events"):
        engine.correlate_events(event_ids)
