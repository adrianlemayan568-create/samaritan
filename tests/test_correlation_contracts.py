"""Tests for immutable correlation contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from samaritan.correlation_contracts import (
    Correlation,
    CorrelationRule,
)


def make_id():
    return str(uuid4())


def make_rule(**overrides):
    values = {
        "rule_id": "auth-transition-v1",
        "event_type_pairs": (
            ("authentication_failure", "authentication_success"),
        ),
        "shared_entity_types": ("user_account",),
        "max_time_delta_seconds": 300,
        "description": "Correlate authentication failures and successes.",
    }
    values.update(overrides)
    return CorrelationRule(**values)


def make_correlation(**overrides):
    values = {
        "correlation_id": make_id(),
        "event_ids": (make_id(), make_id()),
        "observation_ids": (make_id(), make_id()),
        "shared_entity_ids": (make_id(),),
        "matched_rules": (make_rule(),),
        "time_delta_seconds": 120,
        "explanation": "Both events involve the same user within two minutes.",
        "created_at": datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return Correlation(**values)


def test_valid_correlation_rule():
    rule = make_rule()

    assert rule.rule_id == "auth-transition-v1"
    assert rule.shared_entity_types == ("user_account",)
    assert rule.max_time_delta_seconds == 300


def test_valid_correlation():
    correlation = make_correlation()

    assert len(correlation.event_ids) == 2
    assert len(correlation.shared_entity_ids) == 1
    assert correlation.time_delta_seconds == 120


def test_rule_allows_same_event_type_pair():
    rule = make_rule(
        event_type_pairs=(("authentication_failure", "authentication_failure"),)
    )

    assert len(rule.event_type_pairs) == 1


@pytest.mark.parametrize(
    "pairs",
    [
        (),
        (("failure", "success"), ("success", "failure")),
        (("failure", "success"), ("failure", "success")),
    ],
)
def test_rule_rejects_empty_or_duplicate_event_type_pairs(pairs):
    with pytest.raises(ValueError):
        make_rule(event_type_pairs=pairs)


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        (("failure",),),
        (("failure", "success", "other"),),
        (("failure", 123),),
    ],
)
def test_rule_rejects_malformed_event_type_pairs(pairs):
    with pytest.raises((TypeError, ValueError)):
        make_rule(event_type_pairs=pairs)


def test_rule_rejects_non_tuple_event_type_pairs():
    with pytest.raises(TypeError):
        make_rule(event_type_pairs=[("failure", "success")])


@pytest.mark.parametrize(
    "entity_types",
    [
        (),
        ("user_account", "user_account"),
    ],
)
def test_rule_rejects_empty_or_duplicate_entity_types(entity_types):
    with pytest.raises(ValueError):
        make_rule(shared_entity_types=entity_types)


def test_rule_rejects_non_tuple_entity_types():
    with pytest.raises(TypeError):
        make_rule(shared_entity_types=["user_account"])


@pytest.mark.parametrize(
    "delta",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rule_rejects_invalid_time_windows(delta):
    with pytest.raises(ValueError):
        make_rule(max_time_delta_seconds=delta)


def test_rule_rejects_boolean_time_window():
    with pytest.raises(TypeError):
        make_rule(max_time_delta_seconds=True)


def test_rule_rejects_invalid_identifier():
    with pytest.raises(ValueError):
        make_rule(rule_id="")


def test_rule_rejects_unsupported_schema_version():
    with pytest.raises(ValueError):
        make_rule(schema_version=999)


def test_rule_rejects_boolean_schema_version():
    with pytest.raises(TypeError):
        make_rule(schema_version=True)


def test_rule_is_immutable():
    rule = make_rule()

    with pytest.raises(FrozenInstanceError):
        rule.rule_id = "changed"


def test_correlation_rejects_invalid_correlation_id():
    with pytest.raises(ValueError):
        make_correlation(correlation_id="not-a-uuid")


def test_correlation_rejects_identical_event_ids():
    event_id = make_id()

    with pytest.raises(ValueError):
        make_correlation(event_ids=(event_id, event_id))


def test_correlation_rejects_invalid_event_id():
    with pytest.raises(ValueError):
        make_correlation(event_ids=("invalid", make_id()))


def test_correlation_rejects_malformed_event_ids():
    with pytest.raises(TypeError):
        make_correlation(event_ids=[make_id(), make_id()])


def test_correlation_rejects_malformed_observation_ids():
    with pytest.raises(TypeError):
        make_correlation(observation_ids=(make_id(),))


def test_correlation_allows_same_observation_for_both_events():
    observation_id = make_id()

    correlation = make_correlation(
        observation_ids=(observation_id, observation_id)
    )

    assert correlation.observation_ids == (
        observation_id,
        observation_id,
    )


def test_correlation_rejects_empty_shared_entities():
    with pytest.raises(ValueError):
        make_correlation(shared_entity_ids=())


def test_correlation_rejects_duplicate_shared_entities():
    entity_id = make_id()

    with pytest.raises(ValueError):
        make_correlation(shared_entity_ids=(entity_id, entity_id))


def test_correlation_rejects_invalid_shared_entity_id():
    with pytest.raises(ValueError):
        make_correlation(shared_entity_ids=("invalid",))


def test_correlation_rejects_empty_matched_rules():
    with pytest.raises(ValueError):
        make_correlation(matched_rules=())


def test_correlation_rejects_non_tuple_matched_rules():
    with pytest.raises(TypeError):
        make_correlation(matched_rules=[make_rule()])


def test_correlation_rejects_invalid_matched_rule_type():
    with pytest.raises(TypeError):
        make_correlation(matched_rules=("not-a-rule",))


def test_correlation_rejects_duplicate_matched_rule_ids():
    rule = make_rule()

    with pytest.raises(ValueError):
        make_correlation(matched_rules=(rule, rule))


@pytest.mark.parametrize(
    "delta",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_correlation_rejects_invalid_time_delta(delta):
    with pytest.raises(ValueError):
        make_correlation(time_delta_seconds=delta)


def test_correlation_rejects_boolean_time_delta():
    with pytest.raises(TypeError):
        make_correlation(time_delta_seconds=True)


def test_correlation_accepts_zero_time_delta():
    correlation = make_correlation(time_delta_seconds=0)

    assert correlation.time_delta_seconds == 0


def test_correlation_normalizes_timestamp_to_utc():
    local_time = datetime(
        2026, 11, 16, 15, 0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    correlation = make_correlation(created_at=local_time)

    assert correlation.created_at == datetime(
        2026, 11, 16, 12, 0, tzinfo=timezone.utc
    )
    assert correlation.created_at.utcoffset() == timedelta(0)


def test_correlation_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        make_correlation(
            created_at=datetime(2026, 11, 16, 12, 0)
        )


def test_correlation_rejects_invalid_explanation():
    with pytest.raises(ValueError):
        make_correlation(explanation="   ")


def test_correlation_rejects_unsupported_schema_version():
    with pytest.raises(ValueError):
        make_correlation(schema_version=999)


def test_correlation_rejects_boolean_schema_version():
    with pytest.raises(TypeError):
        make_correlation(schema_version=True)


def test_correlation_is_immutable():
    correlation = make_correlation()

    with pytest.raises(FrozenInstanceError):
        correlation.time_delta_seconds = 500

def test_correlation_accepts_delta_within_all_matched_rule_limits():
    rules = (
        make_rule(rule_id="rule-300", max_time_delta_seconds=300),
        make_rule(rule_id="rule-600", max_time_delta_seconds=600),
    )

    correlation = make_correlation(
        matched_rules=rules,
        time_delta_seconds=250,
    )

    assert correlation.time_delta_seconds == 250


def test_correlation_accepts_delta_exactly_at_rule_limit():
    rule = make_rule(max_time_delta_seconds=300)

    correlation = make_correlation(
        matched_rules=(rule,),
        time_delta_seconds=300,
    )

    assert correlation.time_delta_seconds == 300


def test_correlation_rejects_delta_exceeding_rule_limit():
    rule = make_rule(max_time_delta_seconds=300)

    with pytest.raises(ValueError, match="maximum time delta"):
        make_correlation(
            matched_rules=(rule,),
            time_delta_seconds=301,
        )


def test_correlation_checks_every_matched_rule_limit():
    rules = (
        make_rule(rule_id="rule-300", max_time_delta_seconds=300),
        make_rule(rule_id="rule-600", max_time_delta_seconds=600),
    )

    with pytest.raises(ValueError, match="maximum time delta"):
        make_correlation(
            matched_rules=rules,
            time_delta_seconds=450,
        )


def test_correlation_accepts_delta_with_different_rule_limits():
    rules = (
        make_rule(rule_id="rule-600", max_time_delta_seconds=600),
        make_rule(rule_id="rule-900", max_time_delta_seconds=900),
    )

    correlation = make_correlation(
        matched_rules=rules,
        time_delta_seconds=450,
    )

    assert correlation.time_delta_seconds == 450
