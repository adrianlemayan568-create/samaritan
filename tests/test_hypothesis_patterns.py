"""Tests for complete declarative hypothesis-pattern matching."""

from dataclasses import replace
from datetime import timedelta

from samaritan.contracts import new_id
from samaritan.correlation import CorrelationEngine
from samaritan.correlation_contracts import CorrelationRule
from samaritan.hypothesis_generation import HypothesisGenerationEngine
from samaritan.hypothesis_generation_contracts import (
    HypothesisGenerationRule,
    HypothesisPattern,
    PatternCorrelationConstraint,
    PatternEntityBinding,
    PatternEventRole,
)
from samaritan.hypothesis_patterns import match_hypothesis_pattern

from test_hypothesis_generation import (
    correlate_chain,
    make_correlation_rules,
    make_engine,
    make_event_chain,
    make_generation_rule,
    make_repository,
    persist_test_event,
)
from test_repository import make_entity


def test_valid_three_event_sequence_matches_exact_roles(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "pattern-account")
    events = make_event_chain(repository, source, account, source.created_at)

    result = make_engine(repository).generate(correlate_chain(repository, events))

    assert len(result.hypotheses) == 1
    assert len(result.generations) == 1
    assert {
        item.event_id
        for item in result.generations[0].evidence
        if item.event_id is not None
    } == {event.event_id for event in events}


def test_connected_graph_with_different_account_identities_does_not_match(tmp_path):
    repository, _, source = make_repository(tmp_path)
    first = make_entity("user_account", "first-account")
    second = make_entity("user_account", "second-account")
    failure = persist_test_event(
        repository, source, first, "authentication_failure", source.created_at
    )
    success = persist_test_event(
        repository, source, second, "authentication_success",
        source.created_at + timedelta(minutes=1)
    )
    host = persist_test_event(
        repository, source, second, "host_activity",
        source.created_at + timedelta(minutes=2)
    )

    correlations = correlate_chain(repository, (failure, success, host))
    result = make_engine(repository).generate(correlations)

    assert len(correlations) == 1
    assert result.hypotheses == ()


def test_failure_after_success_is_rejected_by_temporal_pattern(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "reversed-account")
    failure = persist_test_event(
        repository, source, account, "authentication_failure",
        source.created_at + timedelta(minutes=2)
    )
    success = persist_test_event(
        repository, source, account, "authentication_success",
        source.created_at
    )
    host = persist_test_event(
        repository, source, account, "host_activity",
        source.created_at + timedelta(minutes=4)
    )

    correlations = correlate_chain(repository, (failure, success, host))
    result = make_engine(repository).generate(correlations)

    assert len(correlations) == 2
    assert result.hypotheses == ()


def test_host_activity_before_success_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "host-before-success")
    failure = persist_test_event(
        repository, source, account, "authentication_failure",
        source.created_at
    )
    success = persist_test_event(
        repository, source, account, "authentication_success",
        source.created_at + timedelta(minutes=2)
    )
    host = persist_test_event(
        repository, source, account, "host_activity",
        source.created_at + timedelta(minutes=1)
    )

    correlations = correlate_chain(repository, (failure, success, host))
    result = make_engine(repository).generate(correlations)

    assert len(correlations) == 2
    assert result.hypotheses == ()


def test_wrong_participant_roles_do_not_match(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "wrong-participant-role")
    events = make_event_chain(repository, source, account, source.created_at)

    result = make_engine(
        repository,
        (make_generation_rule(participant_role="target"),),
    ).generate(correlate_chain(repository, events))

    assert result.hypotheses == ()


def test_multiple_failure_candidates_for_one_sequence_are_ambiguous(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "ambiguous-account")
    events = (
        persist_test_event(
            repository, source, account, "authentication_failure",
            source.created_at
        ),
        persist_test_event(
            repository, source, account, "authentication_failure",
            source.created_at + timedelta(seconds=30)
        ),
        persist_test_event(
            repository, source, account, "authentication_success",
            source.created_at + timedelta(minutes=2)
        ),
        persist_test_event(
            repository, source, account, "host_activity",
            source.created_at + timedelta(minutes=4)
        ),
    )

    correlations = correlate_chain(repository, events)
    result = make_engine(repository).generate(correlations)

    assert len(correlations) == 3
    assert result.hypotheses == ()


def test_missing_required_event_role_produces_no_match(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "missing-role-account")
    events = make_event_chain(repository, source, account, source.created_at)
    supplied = tuple(
        item for item in correlate_chain(repository, events)
        if "host_activity" not in {
            repository.get_event(event_id).event_type
            for event_id in item.event_ids
        }
    )

    result = make_engine(repository).generate(supplied)

    assert result.hypotheses == ()


def test_correlation_input_order_does_not_change_match_set(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "order-independent-account")
    events = make_event_chain(repository, source, account, source.created_at)
    correlations = correlate_chain(repository, events)
    engine = make_engine(repository)

    forward = engine.generate(correlations)
    reverse = engine.generate(tuple(reversed(correlations)))

    def evidence_sets(result):
        return {
            frozenset(
                item.event_id
                for item in generation.evidence
                if item.event_id is not None
            )
            for generation in result.generations
        }

    assert evidence_sets(forward) == evidence_sets(reverse)


def test_unrelated_partial_correlations_cannot_contaminate_valid_pattern(tmp_path):
    repository, _, source = make_repository(tmp_path)
    incomplete_account = make_entity("user_account", "incomplete-account")
    valid_account = make_entity("user_account", "valid-account")
    incomplete = (
        persist_test_event(
            repository, source, incomplete_account, "authentication_failure",
            source.created_at
        ),
        persist_test_event(
            repository, source, incomplete_account, "authentication_success",
            source.created_at + timedelta(minutes=1)
        ),
    )
    valid = make_event_chain(
        repository, source, valid_account, source.created_at + timedelta(minutes=20)
    )
    correlations = correlate_chain(repository, incomplete + valid)

    result = make_engine(repository).generate(correlations)

    assert len(result.hypotheses) == 1
    generated_ids = {
        item.event_id
        for item in result.generations[0].evidence
        if item.event_id is not None
    }
    assert generated_ids == {event.event_id for event in valid}


def test_independent_complete_sequences_produce_two_matches(tmp_path):
    repository, _, source = make_repository(tmp_path)
    first_account = make_entity("user_account", "independent-account-1")
    second_account = make_entity("user_account", "independent-account-2")
    first = make_event_chain(repository, source, first_account, source.created_at)
    second = make_event_chain(
        repository, source, second_account, source.created_at + timedelta(minutes=20)
    )

    result = make_engine(repository).generate(
        correlate_chain(repository, first + second)
    )

    assert len(result.hypotheses) == 2
    generated_sets = {
        frozenset(
            item.event_id
            for item in generation.evidence
            if item.event_id is not None
        )
        for generation in result.generations
    }
    assert generated_sets == {
        frozenset(event.event_id for event in first),
        frozenset(event.event_id for event in second),
    }


def test_equal_timestamps_are_rejected_by_strict_ordering(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "equal-time-account")
    failure = persist_test_event(
        repository, source, account, "authentication_failure", source.created_at
    )
    success = persist_test_event(
        repository, source, account, "authentication_success", source.created_at
    )
    host = persist_test_event(
        repository, source, account, "host_activity",
        source.created_at + timedelta(minutes=1)
    )

    correlations = correlate_chain(repository, (failure, success, host))
    result = make_engine(repository).generate(correlations)

    assert len(correlations) == 2
    assert result.hypotheses == ()


def test_event_reuse_across_generation_rules_is_allowed_but_roles_are_distinct(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "reusable-event-account")
    events = make_event_chain(repository, source, account, source.created_at)
    correlations = correlate_chain(repository, events)
    first = make_generation_rule(rule_id="pattern-rule-a")
    second = make_generation_rule(rule_id="pattern-rule-b")

    result = HypothesisGenerationEngine(
        repository,
        (first, second),
        correlation_rules=make_correlation_rules(),
    ).generate(correlations)

    assert len(result.hypotheses) == 2
    event_sets = [
        {
            item.event_id
            for item in generation.evidence
            if item.event_id is not None
        }
        for generation in result.generations
    ]
    assert event_sets[0] == event_sets[1] == {
        event.event_id for event in events
    }



def test_pattern_maximum_interval_is_inclusive_at_boundary(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "inclusive-boundary-account")
    events = (
        persist_test_event(repository, source, account, "authentication_failure", source.created_at),
        persist_test_event(repository, source, account, "authentication_success", source.created_at + timedelta(seconds=300)),
        persist_test_event(repository, source, account, "host_activity", source.created_at + timedelta(seconds=600)),
    )
    result = make_engine(repository).generate(correlate_chain(repository, events))
    assert len(result.hypotheses) == 1


def test_pattern_interval_is_enforced_even_if_correlation_allows_more(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "pattern-interval-account")
    events = (
        persist_test_event(repository, source, account, "authentication_failure", source.created_at),
        persist_test_event(repository, source, account, "authentication_success", source.created_at + timedelta(seconds=360)),
        persist_test_event(repository, source, account, "host_activity", source.created_at + timedelta(seconds=420)),
    )
    relaxed_rules = tuple(
        replace(rule, max_time_delta_seconds=600)
        for rule in make_correlation_rules()
    )
    correlations = CorrelationEngine(repository, relaxed_rules).correlate_events(
        event.event_id for event in events
    )
    result = HypothesisGenerationEngine(
        repository,
        (make_generation_rule(),),
        correlation_rules=relaxed_rules,
    ).generate(correlations)
    assert len(correlations) == 2
    assert result.hypotheses == ()



def test_one_event_cannot_fill_two_required_roles(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "single-event-account")
    event = persist_test_event(
        repository, source, account, "authentication_failure", source.created_at
    )
    pattern = HypothesisPattern(
        pattern_id="two-failure-roles",
        version=1,
        event_roles=(
            PatternEventRole("first_failure", "authentication_failure"),
            PatternEventRole("second_failure", "authentication_failure"),
        ),
        correlation_constraints=(
            PatternCorrelationConstraint(
                "first_failure", "second_failure", "same-type"
            ),
        ),
        entity_bindings=(
            PatternEntityBinding(
                "account", "first_failure", "user_account", "subject"
            ),
            PatternEntityBinding(
                "account", "second_failure", "user_account", "subject"
            ),
        ),
        temporal_constraints=(),
    )

    result = match_hypothesis_pattern(
        pattern,
        (),
        (event,),
        {account.entity_id: account},
    )

    assert result == ()


def test_role_permutations_of_the_same_event_set_are_ambiguous(tmp_path):
    repository, _, source = make_repository(tmp_path)
    account = make_entity("user_account", "permuted-role-account")
    events = (
        persist_test_event(
            repository, source, account, "authentication_failure",
            source.created_at
        ),
        persist_test_event(
            repository, source, account, "authentication_failure",
            source.created_at + timedelta(seconds=30)
        ),
    )
    same_type_rule = CorrelationRule(
        rule_id="same-type",
        event_type_pairs=(("authentication_failure", "authentication_failure"),),
        shared_entity_types=("user_account",),
        max_time_delta_seconds=300,
        description="Two authentication-failure events share an account.",
    )
    pattern = HypothesisPattern(
        pattern_id="two-failure-roles",
        version=1,
        event_roles=(
            PatternEventRole("first_failure", "authentication_failure"),
            PatternEventRole("second_failure", "authentication_failure"),
        ),
        correlation_constraints=(
            PatternCorrelationConstraint(
                "first_failure", "second_failure", "same-type"
            ),
        ),
        entity_bindings=(
            PatternEntityBinding(
                "account", "first_failure", "user_account", "subject"
            ),
            PatternEntityBinding(
                "account", "second_failure", "user_account", "subject"
            ),
        ),
        temporal_constraints=(),
    )
    generation_rule = HypothesisGenerationRule(
        rule_id="two-failure-rule",
        pattern=pattern,
        statement="Two failures were observed.",
        description="Ambiguity policy test.",
    )
    correlations = CorrelationEngine(
        repository, (same_type_rule,)
    ).correlate_events(event.event_id for event in events)

    result = HypothesisGenerationEngine(
        repository,
        (generation_rule,),
        correlation_rules=(same_type_rule,),
    ).generate(correlations)

    assert len(correlations) == 1
    assert result.hypotheses == ()
