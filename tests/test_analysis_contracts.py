
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from samaritan.analysis_contracts import (
    ASSESSMENT_OUTCOMES,
    CONFIDENCE_LEVELS,
    EVIDENCE_STANCES,
    AlternativeExplanation,
    Assessment,
    EvidenceEvaluation,
    EvidenceReference,
    Hypothesis,
)


def make_reference():
    return EvidenceReference(observation_id=str(uuid4()))


def make_hypothesis(**overrides):
    values = {
        "hypothesis_id": str(uuid4()),
        "statement": "Repeated authentication failures indicate possible credential misuse",
        "description": "Several failed login attempts were followed by a successful login.",
        "created_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return Hypothesis(**values)


def make_evaluation(**overrides):
    values = {
        "evaluation_id": str(uuid4()),
        "evidence": make_reference(),
        "stance": "supports",
        "rationale": "The event sequence is consistent with the hypothesis.",
    }
    values.update(overrides)
    return EvidenceEvaluation(**values)


def make_alternative(**overrides):
    values = {
        "explanation_id": str(uuid4()),
        "statement": "The failures may have resulted from a mistyped password.",
        "rationale": "No independent evidence establishes malicious intent.",
        "evidence": (),
    }
    values.update(overrides)
    return AlternativeExplanation(**values)


def make_assessment(**overrides):
    values = {
        "assessment_id": str(uuid4()),
        "hypothesis_id": str(uuid4()),
        "outcome": "inconclusive",
        "confidence": "low",
        "summary": "Available evidence is insufficient for a firm conclusion.",
        "assessed_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return Assessment(**values)


class TestEvidenceReference:
    def test_observation_reference_is_valid(self):
        reference = make_reference()

        assert reference.observation_id is not None
        assert reference.event_id is None

    def test_event_reference_is_valid(self):
        event_id = str(uuid4())
        reference = EvidenceReference(event_id=event_id)

        assert reference.event_id == event_id
        assert reference.observation_id is None

    def test_missing_reference_is_rejected(self):
        with pytest.raises(ValueError):
            EvidenceReference()

    def test_both_reference_types_are_rejected(self):
        with pytest.raises(ValueError):
            EvidenceReference(
                observation_id=str(uuid4()),
                event_id=str(uuid4()),
            )

    def test_invalid_observation_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            EvidenceReference(observation_id="invalid")

    def test_invalid_event_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            EvidenceReference(event_id="invalid")

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            EvidenceReference(
                observation_id=str(uuid4()),
                schema_version=999,
            )


class TestHypothesis:
    def test_valid_hypothesis(self):
        hypothesis = make_hypothesis()

        assert hypothesis.statement
        assert hypothesis.description

    def test_invalid_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_hypothesis(hypothesis_id="invalid")

    @pytest.mark.parametrize("field", ["statement", "description"])
    def test_empty_text_is_rejected(self, field):
        with pytest.raises((TypeError, ValueError)):
            make_hypothesis(**{field: ""})

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_hypothesis(created_at=datetime(2026, 1, 1))

    def test_timestamp_is_normalized_to_utc(self):
        hypothesis = make_hypothesis()

        assert hypothesis.created_at.tzinfo == timezone.utc

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_hypothesis(schema_version=999)

    def test_hypothesis_is_immutable(self):
        hypothesis = make_hypothesis()

        with pytest.raises(FrozenInstanceError):
            hypothesis.statement = "Changed"


class TestEvidenceEvaluation:
    def test_valid_evaluation(self):
        evaluation = make_evaluation()

        assert evaluation.stance == "supports"
        assert isinstance(evaluation.evidence, EvidenceReference)

    @pytest.mark.parametrize("stance", sorted(EVIDENCE_STANCES))
    def test_supported_stances(self, stance):
        evaluation = make_evaluation(stance=stance)

        assert evaluation.stance == stance

    def test_invalid_stance_is_rejected(self):
        with pytest.raises(ValueError):
            make_evaluation(stance="uncertain")

    def test_invalid_evidence_type_is_rejected(self):
        with pytest.raises(TypeError):
            make_evaluation(evidence="not a reference")

    def test_empty_rationale_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_evaluation(rationale="")

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_evaluation(schema_version=999)


class TestAlternativeExplanation:
    def test_valid_alternative(self):
        alternative = make_alternative()

        assert alternative.statement
        assert alternative.evidence == ()

    def test_alternative_accepts_evidence(self):
        reference = make_reference()
        alternative = make_alternative(evidence=(reference,))

        assert alternative.evidence == (reference,)

    def test_evidence_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_alternative(evidence=[make_reference()])

    def test_evidence_items_must_be_references(self):
        with pytest.raises(TypeError):
            make_alternative(evidence=("invalid",))

    def test_invalid_statement_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_alternative(statement="")

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_alternative(schema_version=999)


class TestAssessment:
    def test_valid_assessment(self):
        assessment = make_assessment()

        assert assessment.outcome == "inconclusive"
        assert assessment.confidence == "low"

    @pytest.mark.parametrize("outcome", sorted(ASSESSMENT_OUTCOMES))
    def test_supported_outcomes(self, outcome):
        assessment = make_assessment(outcome=outcome)

        assert assessment.outcome == outcome

    @pytest.mark.parametrize("confidence", sorted(CONFIDENCE_LEVELS))
    def test_supported_confidence_levels(self, confidence):
        assessment = make_assessment(confidence=confidence)

        assert assessment.confidence == confidence

    def test_invalid_outcome_is_rejected(self):
        with pytest.raises(ValueError):
            make_assessment(outcome="confirmed_attack")

    def test_invalid_confidence_is_rejected(self):
        with pytest.raises(ValueError):
            make_assessment(confidence="certain")

    def test_invalid_hypothesis_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_assessment(hypothesis_id="invalid")

    def test_empty_summary_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_assessment(summary="")

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_assessment(assessed_at=datetime(2026, 1, 1))

    def test_nested_evaluations_are_accepted(self):
        evaluation = make_evaluation()
        assessment = make_assessment(evaluations=(evaluation,))

        assert assessment.evaluations == (evaluation,)

    def test_nested_alternatives_are_accepted(self):
        alternative = make_alternative()
        assessment = make_assessment(alternatives=(alternative,))

        assert assessment.alternatives == (alternative,)

    def test_evaluations_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_assessment(evaluations=[make_evaluation()])

    def test_evaluation_items_must_be_valid_objects(self):
        with pytest.raises(TypeError):
            make_assessment(evaluations=("invalid",))

    def test_duplicate_evaluation_ids_are_rejected(self):
        evaluation_id = str(uuid4())
        first = make_evaluation(evaluation_id=evaluation_id)
        second = make_evaluation(evaluation_id=evaluation_id)

        with pytest.raises(ValueError):
            make_assessment(evaluations=(first, second))

    def test_duplicate_alternative_ids_are_rejected(self):
        explanation_id = str(uuid4())
        first = make_alternative(explanation_id=explanation_id)
        second = make_alternative(explanation_id=explanation_id)

        with pytest.raises(ValueError):
            make_assessment(alternatives=(first, second))

    def test_missing_information_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_assessment(missing_information=["missing logs"])

    def test_missing_information_items_must_be_valid_text(self):
        with pytest.raises((TypeError, ValueError)):
            make_assessment(missing_information=("",))

    def test_assessment_is_immutable(self):
        assessment = make_assessment()

        with pytest.raises(FrozenInstanceError):
            assessment.outcome = "supported"

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_assessment(schema_version=999)
