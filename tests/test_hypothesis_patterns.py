"""Tests for pure correlation graph matching."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from samaritan.correlation_contracts import (
    Correlation,
    CorrelationRule,
)
from samaritan.hypothesis_generation_contracts import (
    HypothesisGenerationRule,
)
from samaritan.hypothesis_patterns import (
    match_correlation_components,
)


def make_correlation_rule(rule_id):
    return CorrelationRule(
        rule_id=rule_id,
        event_type_pairs=(("event_a", "event_b"),),
        shared_entity_types=("user_account",),
        max_time_delta_seconds=300,
        description=f"Test correlation rule: {rule_id}",
    )


def make_correlation(
    event_ids=None,
    rule_ids=("rule_a",),
    correlation_id=None,
):
    if event_ids is None:
        event_ids = (str(uuid4()), str(uuid4()))

    matched_rules = tuple(
        make_correlation_rule(rule_id)
        for rule_id in rule_ids
    )

    return Correlation(
        correlation_id=correlation_id or str(uuid4()),
        event_ids=event_ids,
        observation_ids=(str(uuid4()), str(uuid4())),
        shared_entity_ids=(str(uuid4()),),
        matched_rules=matched_rules,
        time_delta_seconds=30,
        explanation="Synthetic correlation for graph testing.",
        created_at=datetime.now(timezone.utc),
    )


def make_generation_rule(
    required=("rule_a", "rule_b"),
    minimum=2,
):
    return HypothesisGenerationRule(
        rule_id="test_generation",
        required_correlation_rule_ids=required,
        minimum_correlations=minimum,
        statement="A test hypothesis.",
        description="A test generation rule.",
    )


class TestCorrelationComponentMatching:

    def test_connected_component_with_all_required_rules_matches(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())
        fourth = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((second, third), ("rule_b",)),
            make_correlation((third, fourth), ("rule_a",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=3),
        )

        assert len(result) == 1
        assert len(result[0]) == 3
        assert {
            rule.rule_id
            for correlation in result[0]
            for rule in correlation.matched_rules
        } == {"rule_a", "rule_b"}

    def test_required_rules_in_separate_components_do_not_match(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())
        fourth = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((third, fourth), ("rule_b",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(),
        )

        assert result == ()

    def test_missing_required_rule_does_not_match(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((second, third), ("rule_a",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(),
        )

        assert result == ()

    def test_minimum_correlation_count_is_enforced(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((second, third), ("rule_b",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=3),
        )

        assert result == ()

    def test_exact_minimum_correlation_count_matches(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((second, third), ("rule_b",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=2),
        )

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_one_correlation_matching_two_rules_counts_once(self):
        first = str(uuid4())
        second = str(uuid4())

        correlations = (
            make_correlation(
                (first, second),
                ("rule_a", "rule_b"),
            ),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=2),
        )

        assert result == ()

    def test_one_correlation_matching_two_rules_can_match_minimum_one(self):
        first = str(uuid4())
        second = str(uuid4())

        correlations = (
            make_correlation(
                (first, second),
                ("rule_a", "rule_b"),
            ),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=1),
        )

        assert len(result) == 1
        assert len(result[0]) == 1

    def test_unrelated_correlations_are_ignored(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())
        fourth = str(uuid4())
        fifth = str(uuid4())
        sixth = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((second, third), ("rule_b",)),
            make_correlation((fourth, fifth), ("unrelated_rule",)),
            make_correlation((fifth, sixth), ("unrelated_rule",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=2),
        )

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_multiple_qualifying_components_are_returned(self):
        events = [
            str(uuid4())
            for _ in range(6)
        ]

        correlations = (
            make_correlation((events[0], events[1]), ("rule_a",)),
            make_correlation((events[1], events[2]), ("rule_b",)),
            make_correlation((events[3], events[4]), ("rule_a",)),
            make_correlation((events[4], events[5]), ("rule_b",)),
        )

        result = match_correlation_components(
            correlations,
            make_generation_rule(minimum=2),
        )

        assert len(result) == 2
        assert all(len(component) == 2 for component in result)

    def test_result_order_is_independent_of_input_order(self):
        events = [
            str(uuid4())
            for _ in range(4)
        ]

        first = make_correlation(
            (events[0], events[1]),
            ("rule_a",),
        )
        second = make_correlation(
            (events[1], events[2]),
            ("rule_b",),
        )
        third = make_correlation(
            (events[2], events[3]),
            ("rule_a",),
        )

        rule = make_generation_rule(minimum=3)

        forward = match_correlation_components(
            (first, second, third),
            rule,
        )
        reversed_result = match_correlation_components(
            (third, first, second),
            rule,
        )

        assert forward == reversed_result

    def test_empty_input_returns_empty_tuple(self):
        assert match_correlation_components(
            (),
            make_generation_rule(),
        ) == ()

    def test_generator_input_is_supported(self):
        first = str(uuid4())
        second = str(uuid4())
        third = str(uuid4())

        correlations = (
            make_correlation((first, second), ("rule_a",)),
            make_correlation((second, third), ("rule_b",)),
        )

        result = match_correlation_components(
            (item for item in correlations),
            make_generation_rule(),
        )

        assert len(result) == 1


class TestCorrelationInputValidation:

    def test_non_rule_argument_is_rejected(self):
        with pytest.raises(TypeError, match="HypothesisGenerationRule"):
            match_correlation_components((), object())

    @pytest.mark.parametrize("invalid", ["text", b"bytes"])
    def test_string_collection_is_rejected(self, invalid):
        with pytest.raises(TypeError, match="iterable"):
            match_correlation_components(
                invalid,
                make_generation_rule(),
            )

    def test_non_iterable_collection_is_rejected(self):
        with pytest.raises(TypeError, match="iterable"):
            match_correlation_components(
                None,
                make_generation_rule(),
            )

    def test_non_correlation_item_is_rejected(self):
        with pytest.raises(TypeError, match="Correlation"):
            match_correlation_components(
                (object(),),
                make_generation_rule(),
            )

    def test_duplicate_correlation_ids_are_rejected(self):
        correlation_id = str(uuid4())

        first = make_correlation(
            correlation_id=correlation_id,
        )
        second = make_correlation(
            correlation_id=correlation_id,
        )

        with pytest.raises(ValueError, match="duplicate correlation IDs"):
            match_correlation_components(
                (first, second),
                make_generation_rule(),
            )

    def test_duplicate_unordered_event_pairs_are_rejected(self):
        first_event = str(uuid4())
        second_event = str(uuid4())

        first = make_correlation(
            (first_event, second_event),
        )
        second = make_correlation(
            (second_event, first_event),
        )

        with pytest.raises(ValueError, match="duplicate unordered event pairs"):
            match_correlation_components(
                (first, second),
                make_generation_rule(),
            )
