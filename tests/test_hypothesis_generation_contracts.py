from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from samaritan.analysis_contracts import EvidenceReference, Hypothesis
from samaritan.hypothesis_generation_contracts import (
    HypothesisGeneration,
    HypothesisGenerationResult,
    HypothesisGenerationRule,
    HypothesisPattern,
    PatternCorrelationConstraint,
    PatternEntityBinding,
    PatternEventRole,
    PatternTemporalConstraint,
)


def make_pattern(**overrides):
    values = {
        "pattern_id": "authentication_sequence",
        "version": 1,
        "event_roles": (
            PatternEventRole("failure", "authentication_failure"),
            PatternEventRole("success", "authentication_success"),
            PatternEventRole("host", "host_activity"),
        ),
        "correlation_constraints": (
            PatternCorrelationConstraint(
                "failure", "success", "repeated_auth_failures"
            ),
            PatternCorrelationConstraint(
                "success", "host", "failure_then_success"
            ),
        ),
        "entity_bindings": (
            PatternEntityBinding("account", "failure", "user_account", "target"),
            PatternEntityBinding("account", "success", "user_account", "target"),
            PatternEntityBinding("account", "host", "user_account", "subject"),
        ),
        "temporal_constraints": (
            PatternTemporalConstraint("failure", "success", True, 300),
            PatternTemporalConstraint("success", "host", True, 300),
        ),
    }
    values.update(overrides)
    return HypothesisPattern(**values)


def make_rule(**overrides):
    values = {
        "rule_id": "auth_sequence",
        "pattern": make_pattern(),
        "statement": "Observed authentication activity may indicate "
        "possible credential misuse.",
        "description": "Repeated authentication failures were followed "
        "by a successful authentication event.",
    }
    values.update(overrides)
    return HypothesisGenerationRule(**values)


def make_hypothesis(**overrides):
    values = {
        "hypothesis_id": str(uuid4()),
        "statement": "Observed activity may indicate credential misuse.",
        "description": "A defined authentication pattern was observed.",
        "created_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return Hypothesis(**values)


def make_generation(**overrides):
    values = {
        "generation_id": str(uuid4()),
        "hypothesis_id": str(uuid4()),
        "rule_id": "auth_sequence",
        "correlation_ids": (str(uuid4()), str(uuid4())),
        "evidence": (
            EvidenceReference(event_id=str(uuid4())),
            EvidenceReference(observation_id=str(uuid4())),
        ),
        "created_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return HypothesisGeneration(**values)


class TestHypothesisPatternContracts:
    def test_valid_pattern_and_rule(self):
        pattern = make_pattern()
        rule = make_rule(pattern=pattern)

        assert pattern.pattern_id == "authentication_sequence"
        assert pattern.version == 1
        assert len(pattern.event_roles) == 3
        assert rule.pattern is pattern

    def test_pattern_is_immutable(self):
        pattern = make_pattern()
        with pytest.raises(FrozenInstanceError):
            pattern.pattern_id = "changed"

    def test_event_cardinality_policy_is_explicit(self):
        assert make_pattern().event_cardinality == "exactly_one_per_role"

    def test_unsupported_event_cardinality_policy_is_rejected(self):
        with pytest.raises(ValueError, match="event cardinality"):
            make_pattern(event_cardinality="one_or_more")

    def test_unsupported_event_reuse_policy_is_rejected(self):
        with pytest.raises(ValueError, match="event reuse"):
            make_pattern(event_reuse_policy="unrestricted")

    def test_unsupported_ambiguity_policy_is_rejected(self):
        with pytest.raises(ValueError, match="ambiguity policy"):
            make_pattern(ambiguity_policy="choose_first")

    def test_rule_requires_a_pattern(self):
        with pytest.raises(TypeError, match="HypothesisPattern"):
            make_rule(pattern=object())

    def test_event_roles_must_be_unique(self):
        roles = (
            PatternEventRole("failure", "authentication_failure"),
            PatternEventRole("failure", "authentication_success"),
        )
        with pytest.raises(ValueError, match="unique"):
            make_pattern(event_roles=roles)

    def test_unknown_correlation_role_is_rejected(self):
        constraints = (
            PatternCorrelationConstraint("missing", "success", "rule_a"),
            PatternCorrelationConstraint("success", "host", "rule_b"),
        )
        with pytest.raises(ValueError, match="unknown event role"):
            make_pattern(correlation_constraints=constraints)

    def test_pattern_correlation_graph_must_be_connected(self):
        constraints = (
            PatternCorrelationConstraint("failure", "success", "rule_a"),
            PatternCorrelationConstraint("host", "success", "rule_b"),
        )
        # This is connected; remove the second edge to isolate host.
        with pytest.raises(ValueError, match="connected"):
            make_pattern(
                correlation_constraints=constraints[:1],
                temporal_constraints=(),
            )

    def test_duplicate_correlation_constraints_are_rejected(self):
        constraint = PatternCorrelationConstraint(
            "failure", "success", "repeated_auth_failures"
        )
        with pytest.raises(ValueError, match="duplicate"):
            make_pattern(
                correlation_constraints=(
                    constraint,
                    constraint,
                    PatternCorrelationConstraint(
                        "success", "host", "failure_then_success"
                    ),
                )
            )

    def test_each_named_binding_must_connect_two_roles(self):
        bindings = (
            PatternEntityBinding("account", "failure", "user_account", "target"),
            PatternEntityBinding("account", "success", "user_account", "target"),
            PatternEntityBinding("host", "host", "hostname", "target"),
        )
        with pytest.raises(ValueError, match="at least two event roles"):
            make_pattern(entity_bindings=bindings)

    def test_binding_entity_type_must_be_consistent(self):
        bindings = (
            PatternEntityBinding("account", "failure", "user_account", "target"),
            PatternEntityBinding("account", "success", "hostname", "target"),
            PatternEntityBinding("account", "host", "user_account", "subject"),
        )
        with pytest.raises(ValueError, match="consistent entity type"):
            make_pattern(entity_bindings=bindings)

    def test_correlated_roles_must_share_a_named_binding(self):
        bindings = (
            PatternEntityBinding("account", "failure", "user_account", "target"),
            PatternEntityBinding("account", "success", "user_account", "target"),
            PatternEntityBinding("other", "failure", "user_account", "target"),
            PatternEntityBinding("other", "host", "user_account", "subject"),
        )
        with pytest.raises(ValueError, match="share a named entity binding"):
            make_pattern(entity_bindings=bindings)

    def test_temporal_constraints_must_reference_known_roles(self):
        with pytest.raises(ValueError, match="unknown event role"):
            make_pattern(
                temporal_constraints=(
                    PatternTemporalConstraint("failure", "missing", True, 300),
                )
            )

    def test_duplicate_temporal_constraints_are_rejected(self):
        constraint = PatternTemporalConstraint("failure", "success", True, 300)
        with pytest.raises(ValueError, match="duplicate"):
            make_pattern(
                temporal_constraints=(
                    constraint,
                    constraint,
                    PatternTemporalConstraint("success", "host", True, 300),
                )
            )

    def test_impossible_strict_temporal_cycle_is_rejected(self):
        constraints = (
            PatternTemporalConstraint("failure", "success", True, 300),
            PatternTemporalConstraint("success", "host", True, 300),
            PatternTemporalConstraint("host", "failure", True, 300),
        )
        with pytest.raises(ValueError, match="strict cycle"):
            make_pattern(temporal_constraints=constraints)

    def test_strict_zero_interval_is_rejected(self):
        with pytest.raises(ValueError, match="zero maximum interval"):
            PatternTemporalConstraint("failure", "success", True, 0)

    def test_boolean_temporal_interval_is_rejected(self):
        with pytest.raises(TypeError, match="numeric"):
            PatternTemporalConstraint("failure", "success", True, True)

    def test_non_finite_temporal_interval_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            PatternTemporalConstraint(
                "failure", "success", True, float("inf")
            )

    def test_temporal_strict_flag_must_be_boolean(self):
        with pytest.raises(TypeError, match="boolean"):
            PatternTemporalConstraint("failure", "success", 1, 300)

    def test_pattern_version_must_be_positive_integer(self):
        with pytest.raises(ValueError, match="positive integer"):
            make_pattern(version=True)

    def test_pattern_schema_version_must_be_supported(self):
        with pytest.raises(ValueError, match="schema version"):
            make_pattern(schema_version=999)

    def test_rule_schema_version_must_be_supported(self):
        with pytest.raises(ValueError, match="schema version"):
            make_rule(schema_version=True)

    def test_rule_required_text_must_not_be_empty(self):
        with pytest.raises((TypeError, ValueError)):
            make_rule(statement="")


class TestHypothesisGeneration:
    def test_valid_generation(self):
        generation = make_generation()

        assert generation.rule_id == "auth_sequence"
        assert len(generation.correlation_ids) == 2
        assert len(generation.evidence) == 2

    @pytest.mark.parametrize(
        "field",
        ["generation_id", "hypothesis_id"],
    )
    def test_invalid_record_ids_are_rejected(self, field):
        with pytest.raises((TypeError, ValueError)):
            make_generation(**{field: "invalid"})

    def test_rule_id_must_be_nonempty(self):
        with pytest.raises((TypeError, ValueError)):
            make_generation(rule_id="")

    def test_correlation_ids_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_generation(correlation_ids=[str(uuid4())])

    def test_at_least_one_correlation_is_required(self):
        with pytest.raises(ValueError):
            make_generation(correlation_ids=())

    def test_invalid_correlation_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_generation(correlation_ids=("invalid",))

    def test_duplicate_correlation_ids_are_rejected(self):
        correlation_id = str(uuid4())

        with pytest.raises(ValueError):
            make_generation(
                correlation_ids=(correlation_id, correlation_id)
            )

    def test_evidence_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_generation(evidence=[EvidenceReference(
                event_id=str(uuid4())
            )])

    def test_at_least_one_evidence_reference_is_required(self):
        with pytest.raises(ValueError):
            make_generation(evidence=())

    def test_evidence_items_must_be_references(self):
        with pytest.raises(TypeError):
            make_generation(evidence=("invalid",))

    def test_duplicate_evidence_references_are_rejected(self):
        reference = EvidenceReference(event_id=str(uuid4()))

        with pytest.raises(ValueError):
            make_generation(evidence=(reference, reference))

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_generation(created_at=datetime(2026, 1, 1))

    def test_timestamp_is_normalized_to_utc(self):
        generation = make_generation()

        assert generation.created_at.tzinfo == timezone.utc

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_generation(schema_version=999)

    def test_boolean_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_generation(schema_version=True)

    def test_generation_is_immutable(self):
        generation = make_generation()

        with pytest.raises(FrozenInstanceError):
            generation.rule_id = "changed"


class TestHypothesisGenerationResult:
    def test_valid_result(self):
        hypothesis = make_hypothesis()
        generation = make_generation(
            hypothesis_id=hypothesis.hypothesis_id
        )

        result = HypothesisGenerationResult(
            hypotheses=(hypothesis,),
            generations=(generation,),
        )

        assert result.hypotheses == (hypothesis,)
        assert result.generations == (generation,)

    def test_empty_result_is_valid(self):
        result = HypothesisGenerationResult(
            hypotheses=(),
            generations=(),
        )

        assert result.hypotheses == ()
        assert result.generations == ()

    def test_hypotheses_must_be_tuple(self):
        with pytest.raises(TypeError):
            HypothesisGenerationResult(
                hypotheses=[],
                generations=(),
            )

    def test_hypothesis_items_must_be_valid(self):
        with pytest.raises(TypeError):
            HypothesisGenerationResult(
                hypotheses=("invalid",),
                generations=(),
            )

    def test_generations_must_be_tuple(self):
        with pytest.raises(TypeError):
            HypothesisGenerationResult(
                hypotheses=(),
                generations=[],
            )

    def test_generation_items_must_be_valid(self):
        with pytest.raises(TypeError):
            HypothesisGenerationResult(
                hypotheses=(),
                generations=("invalid",),
            )

    def test_duplicate_hypothesis_ids_are_rejected(self):
        hypothesis_id = str(uuid4())
        first = make_hypothesis(hypothesis_id=hypothesis_id)
        second = make_hypothesis(hypothesis_id=hypothesis_id)

        with pytest.raises(ValueError):
            HypothesisGenerationResult(
                hypotheses=(first, second),
                generations=(),
            )

    def test_duplicate_generation_ids_are_rejected(self):
        generation_id = str(uuid4())
        first = make_generation(generation_id=generation_id)
        second = make_generation(generation_id=generation_id)

        with pytest.raises(ValueError):
            HypothesisGenerationResult(
                hypotheses=(),
                generations=(first, second),
            )

    def test_one_hypothesis_cannot_have_multiple_generations(self):
        hypothesis = make_hypothesis()
        first = make_generation(
            hypothesis_id=hypothesis.hypothesis_id
        )
        second = make_generation(
            hypothesis_id=hypothesis.hypothesis_id
        )

        with pytest.raises(ValueError):
            HypothesisGenerationResult(
                hypotheses=(hypothesis,),
                generations=(first, second),
            )

    def test_generation_cannot_reference_missing_hypothesis(self):
        hypothesis = make_hypothesis()
        generation = make_generation()

        with pytest.raises(ValueError):
            HypothesisGenerationResult(
                hypotheses=(hypothesis,),
                generations=(generation,),
            )

    def test_hypothesis_cannot_exist_without_generation(self):
        hypothesis = make_hypothesis()

        with pytest.raises(ValueError):
            HypothesisGenerationResult(
                hypotheses=(hypothesis,),
                generations=(),
            )

    def test_generation_cannot_exist_without_hypothesis(self):
        generation = make_generation()

        with pytest.raises(ValueError):
            HypothesisGenerationResult(
                hypotheses=(),
                generations=(generation,),
            )
