from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from samaritan.analysis_contracts import EvidenceReference, Hypothesis
from samaritan.hypothesis_generation_contracts import (
    HypothesisGeneration,
    HypothesisGenerationResult,
    HypothesisGenerationRule,
)


def make_rule(**overrides):
    values = {
        "rule_id": "auth_sequence",
        "required_correlation_rule_ids": (
            "repeated_auth_failures",
            "failure_then_success",
        ),
        "minimum_correlations": 2,
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


class TestHypothesisGenerationRule:
    def test_valid_rule(self):
        rule = make_rule()

        assert rule.rule_id == "auth_sequence"
        assert len(rule.required_correlation_rule_ids) == 2
        assert rule.minimum_correlations == 2

    def test_rule_id_must_be_nonempty(self):
        with pytest.raises((TypeError, ValueError)):
            make_rule(rule_id="")

    def test_correlation_rule_ids_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_rule(required_correlation_rule_ids=["rule_a"])

    def test_at_least_one_correlation_rule_is_required(self):
        with pytest.raises(ValueError):
            make_rule(required_correlation_rule_ids=())

    def test_correlation_rule_ids_must_be_nonempty_text(self):
        with pytest.raises((TypeError, ValueError)):
            make_rule(required_correlation_rule_ids=("rule_a", ""))

    def test_duplicate_correlation_rule_ids_are_rejected(self):
        with pytest.raises(ValueError):
            make_rule(
                required_correlation_rule_ids=("rule_a", "rule_a")
            )

    def test_minimum_correlations_must_be_integer(self):
        with pytest.raises(TypeError):
            make_rule(minimum_correlations=2.0)

    def test_boolean_minimum_correlations_is_rejected(self):
        with pytest.raises(TypeError):
            make_rule(minimum_correlations=True)

    def test_minimum_correlations_must_be_positive(self):
        with pytest.raises(ValueError):
            make_rule(minimum_correlations=0)

    @pytest.mark.parametrize("field", ["statement", "description"])
    def test_required_text_must_not_be_empty(self, field):
        with pytest.raises((TypeError, ValueError)):
            make_rule(**{field: ""})

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_rule(schema_version=999)

    def test_boolean_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_rule(schema_version=True)

    def test_rule_is_immutable(self):
        rule = make_rule()

        with pytest.raises(FrozenInstanceError):
            rule.rule_id = "changed"


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
