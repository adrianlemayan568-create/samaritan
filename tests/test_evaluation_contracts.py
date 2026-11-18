from datetime import datetime, timezone

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from samaritan.analysis_contracts import (
    EVIDENCE_STANCES,
    EvidenceReference,
    Hypothesis,
)
from samaritan.evaluation_contracts import (
    CRITERION_STATUSES,
    CriterionEvaluationResult,
    EvaluationCriterion,
    EvidenceSet,
    HypothesisEvaluationRule,
    HypothesisAlternative,
    HypothesisEvaluation,
    HypothesisSpecification,
    Proposition,
)


def make_reference():
    return EvidenceReference(event_id=str(uuid4()))


def make_proposition(**overrides):
    values = {
        "proposition_type": "account_access_by_unauthorized_actor",
        "subject_entity_id": str(uuid4()),
    }
    values.update(overrides)
    return Proposition(**values)

def make_hypothesis(**overrides):
    values = {
        "hypothesis_id": str(uuid4()),
        "statement": (
            "The observed authentication sequence may represent "
            "unauthorized account access."
        ),
        "description": (
            "Evaluate whether the authentication activity indicates "
            "access by an unauthorized actor."
        ),
        "created_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return Hypothesis(**values)

def make_alternative(**overrides):
    values = {
        "alternative_id": str(uuid4()),
        "proposition": make_proposition(
            proposition_type="authorized_user_activity",
        ),
        "description": (
            "The activity may have been performed by an authorized user."
        ),
    }
    values.update(overrides)
    return HypothesisAlternative(**values)


def make_specification(**overrides):
    values = {
        "hypothesis": make_hypothesis(),
        "evaluation_rule": make_rule(),
        "alternatives": (make_alternative(),),
    }
    values.update(overrides)
    return HypothesisSpecification(**values)

def make_criterion(**overrides):
    values = {
        "criterion_id": "authentication-device-inventory",
        "criterion_type": "authentication_device_inventory",
        "match_stance": "supports",
        "non_match_stance": "contradicts",
        "description": (
            "Check whether the authentication device is absent "
            "from an applicable complete inventory."
        ),
    }
    values.update(overrides)
    return EvaluationCriterion(**values)


def make_rule(**overrides):
    values = {
        "rule_id": "unauthorized-account-access",
        "generation_rule_id": "suspicious-authentication-sequence",
        "criteria": (make_criterion(),),
        "description": (
            "Evaluate whether the authentication sequence represents "
            "unauthorized account access."
        ),
    }
    values.update(overrides)
    return HypothesisEvaluationRule(**values)


def make_result(**overrides):
    criterion = make_criterion()
    values = {
        "criterion": criterion,
        "status": "matched",
        "stance": criterion.match_stance,
        "evidence": (make_reference(),),
        "rationale": (
            "The device was absent from the complete applicable inventory."
        ),
    }
    values.update(overrides)
    return CriterionEvaluationResult(**values)


class TestProposition:
    def test_valid_proposition(self):
        proposition = make_proposition()

        assert proposition.proposition_type == (
            "account_access_by_unauthorized_actor"
        )
        assert proposition.subject_entity_id
        assert proposition.object_entity_id is None
        assert proposition.target_event_id is None

    def test_optional_object_entity_is_accepted(self):
        object_id = str(uuid4())

        proposition = make_proposition(object_entity_id=object_id)

        assert proposition.object_entity_id == object_id

    def test_optional_target_event_is_accepted(self):
        event_id = str(uuid4())

        proposition = make_proposition(target_event_id=event_id)

        assert proposition.target_event_id == event_id

    def test_all_entity_and_event_references_are_accepted(self):
        subject_id = str(uuid4())
        object_id = str(uuid4())
        event_id = str(uuid4())

        proposition = make_proposition(
            subject_entity_id=subject_id,
            object_entity_id=object_id,
            target_event_id=event_id,
        )

        assert proposition.subject_entity_id == subject_id
        assert proposition.object_entity_id == object_id
        assert proposition.target_event_id == event_id

    def test_empty_proposition_type_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_proposition(proposition_type="")

    def test_invalid_subject_entity_id_is_rejected(self):
        with pytest.raises(ValueError):
            make_proposition(subject_entity_id="not-a-uuid")

    def test_invalid_object_entity_id_is_rejected(self):
        with pytest.raises(ValueError):
            make_proposition(object_entity_id="not-a-uuid")

    def test_invalid_target_event_id_is_rejected(self):
        with pytest.raises(ValueError):
            make_proposition(target_event_id="not-a-uuid")

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_proposition(schema_version=999)

    def test_boolean_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_proposition(schema_version=True)

    def test_proposition_is_immutable(self):
        proposition = make_proposition()

        with pytest.raises(FrozenInstanceError):
            proposition.proposition_type = "changed"


class TestEvaluationCriterion:
    def test_valid_criterion(self):
        criterion = make_criterion()

        assert criterion.criterion_id == "authentication-device-inventory"
        assert criterion.match_stance == "supports"
        assert criterion.non_match_stance == "contradicts"

    @pytest.mark.parametrize(
        "field",
        [
            "criterion_id",
            "criterion_type",
            "match_stance",
            "non_match_stance",
            "description",
        ],
    )
    def test_empty_text_is_rejected(self, field):
        with pytest.raises((TypeError, ValueError)):
            make_criterion(**{field: ""})

    @pytest.mark.parametrize("stance", sorted(EVIDENCE_STANCES))
    def test_supported_match_stances(self, stance):
        criterion = make_criterion(match_stance=stance)

        assert criterion.match_stance == stance

    @pytest.mark.parametrize("stance", sorted(EVIDENCE_STANCES))
    def test_supported_non_match_stances(self, stance):
        criterion = make_criterion(non_match_stance=stance)

        assert criterion.non_match_stance == stance

    def test_invalid_match_stance_is_rejected(self):
        with pytest.raises(ValueError):
            make_criterion(match_stance="insufficient")

    def test_invalid_non_match_stance_is_rejected(self):
        with pytest.raises(ValueError):
            make_criterion(non_match_stance="insufficient")

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_criterion(schema_version=999)

    def test_criterion_is_immutable(self):
        criterion = make_criterion()

        with pytest.raises(FrozenInstanceError):
            criterion.description = "Changed"


class TestHypothesisEvaluationRule:
    def test_valid_rule(self):
        rule = make_rule()

        assert rule.generation_rule_id == "suspicious-authentication-sequence"
        assert len(rule.criteria) == 1

    def test_multiple_independent_criteria_are_accepted(self):
        first = make_criterion()
        second = make_criterion(
            criterion_id="authentication-mfa-context",
            criterion_type="authentication_mfa_context",
        )

        rule = make_rule(criteria=(first, second))

        assert rule.criteria == (first, second)

    @pytest.mark.parametrize(
        "field",
        ["rule_id", "generation_rule_id", "description"],
    )
    def test_empty_text_is_rejected(self, field):
        with pytest.raises((TypeError, ValueError)):
            make_rule(**{field: ""})

    def test_criteria_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_rule(criteria=[make_criterion()])

    def test_empty_criteria_are_rejected(self):
        with pytest.raises(ValueError):
            make_rule(criteria=())

    def test_criterion_items_must_be_valid_objects(self):
        with pytest.raises(TypeError):
            make_rule(criteria=("invalid",))

    def test_duplicate_criterion_ids_are_rejected(self):
        first = make_criterion()
        second = make_criterion()

        with pytest.raises(ValueError):
            make_rule(criteria=(first, second))

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_rule(schema_version=999)

    def test_rule_is_immutable(self):
        rule = make_rule()

        with pytest.raises(FrozenInstanceError):
            rule.description = "Changed"


class TestCriterionEvaluationResult:
    def test_valid_matched_result(self):
        result = make_result()

        assert result.status == "matched"
        assert result.stance == "supports"
        assert len(result.evidence) == 1

    def test_valid_not_matched_result(self):
        criterion = make_criterion()

        result = make_result(
            criterion=criterion,
            status="not_matched",
            stance=criterion.non_match_stance,
        )

        assert result.status == "not_matched"
        assert result.stance == "contradicts"

    def test_valid_insufficient_result(self):
        result = make_result(
            status="insufficient",
            stance=None,
            evidence=(),
            rationale="No applicable inventory snapshot was available.",
            missing_information=("Complete device inventory",),
        )

        assert result.status == "insufficient"
        assert result.stance is None
        assert result.missing_information == ("Complete device inventory",)

    @pytest.mark.parametrize("status", sorted(CRITERION_STATUSES))
    def test_supported_statuses(self, status):
        if status == "insufficient":
            result = make_result(
                status=status,
                stance=None,
                evidence=(),
                missing_information=("Required evidence",),
            )
        else:
            criterion = make_criterion()
            stance = (
                criterion.match_stance
                if status == "matched"
                else criterion.non_match_stance
            )
            result = make_result(status=status, stance=stance)

        assert result.status == status

    def test_invalid_status_is_rejected(self):
        with pytest.raises(ValueError):
            make_result(status="unknown")

    def test_invalid_criterion_type_is_rejected(self):
        with pytest.raises(TypeError):
            make_result(criterion="invalid")

    def test_matched_result_requires_matching_stance(self):
        with pytest.raises(ValueError):
            make_result(stance="contradicts")

    def test_not_matched_result_requires_matching_stance(self):
        criterion = make_criterion()

        with pytest.raises(ValueError):
            make_result(
                criterion=criterion,
                status="not_matched",
                stance=criterion.match_stance,
            )

    def test_insufficient_result_rejects_stance(self):
        with pytest.raises(ValueError):
            make_result(
                status="insufficient",
                stance="neutral",
                missing_information=("Required evidence",),
            )

    def test_evaluated_result_requires_evidence(self):
        with pytest.raises(ValueError):
            make_result(evidence=())

    def test_insufficient_result_can_have_no_evidence(self):
        result = make_result(
            status="insufficient",
            stance=None,
            evidence=(),
            missing_information=("Required evidence",),
        )

        assert result.evidence == ()

    def test_insufficient_result_can_preserve_partial_evidence(self):
        reference = make_reference()

        result = make_result(
            status="insufficient",
            stance=None,
            evidence=(reference,),
            missing_information=("Complete device inventory",),
        )

        assert result.evidence == (reference,)
        assert result.missing_information == ("Complete device inventory",)

    def test_evidence_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_result(evidence=[make_reference()])

    def test_evidence_items_must_be_references(self):
        with pytest.raises(TypeError):
            make_result(evidence=("invalid",))

    def test_duplicate_evidence_references_are_rejected(self):
        reference = make_reference()

        with pytest.raises(ValueError):
            make_result(evidence=(reference, reference))

    def test_empty_rationale_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            make_result(rationale="")

    def test_missing_information_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_result(
                status="insufficient",
                stance=None,
                missing_information=["Required evidence"],
            )

    def test_missing_information_items_must_be_valid_text(self):
        with pytest.raises((TypeError, ValueError)):
            make_result(
                status="insufficient",
                stance=None,
                missing_information=("",),
            )

    def test_insufficient_result_requires_missing_information(self):
        with pytest.raises(ValueError):
            make_result(
                status="insufficient",
                stance=None,
                missing_information=(),
            )

    def test_evaluated_result_rejects_missing_information(self):
        with pytest.raises(ValueError):
            make_result(
                missing_information=("Required evidence",),
            )

    def test_unsupported_schema_version_is_rejected(self):
        with pytest.raises(ValueError):
            make_result(schema_version=999)

    def test_result_is_immutable(self):
        result = make_result()

        with pytest.raises(FrozenInstanceError):
            result.status = "not_matched"



class TestHypothesisAlternative:
    def test_valid_alternative(self):
        proposition = make_proposition()

        alternative = HypothesisAlternative(
            alternative_id=str(uuid4()),
            proposition=proposition,
            description="The activity may have been performed by an authorized user.",
        )

        assert alternative.proposition is proposition
        assert alternative.description.startswith("The activity")
        assert alternative.schema_version == 1

    def test_invalid_alternative_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            HypothesisAlternative(
                alternative_id="invalid-id",
                proposition=make_proposition(),
                description="A competing explanation.",
            )

    def test_non_proposition_is_rejected(self):
        with pytest.raises(TypeError):
            HypothesisAlternative(
                alternative_id=str(uuid4()),
                proposition="not a proposition",
                description="A competing explanation.",
            )

    def test_empty_description_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            HypothesisAlternative(
                alternative_id=str(uuid4()),
                proposition=make_proposition(),
                description="",
            )

    @pytest.mark.parametrize("version", [999, True, 1.0])
    def test_invalid_schema_version_is_rejected(self, version):
        with pytest.raises(ValueError):
            HypothesisAlternative(
                alternative_id=str(uuid4()),
                proposition=make_proposition(),
                description="A competing explanation.",
                schema_version=version,
            )

    def test_alternative_is_immutable(self):
        alternative = HypothesisAlternative(
            alternative_id=str(uuid4()),
            proposition=make_proposition(),
            description="A competing explanation.",
        )

        with pytest.raises(FrozenInstanceError):
            alternative.description = "Changed explanation."


class TestHypothesisSpecification:
    def test_valid_specification(self):
        specification = make_specification()

        assert isinstance(specification.hypothesis, Hypothesis)
        assert isinstance(
            specification.evaluation_rule,
            HypothesisEvaluationRule,
        )
        assert len(specification.alternatives) == 1
        assert specification.schema_version == 1

    def test_preserves_supplied_objects(self):
        hypothesis = make_hypothesis()
        rule = make_rule()
        alternative = make_alternative()

        specification = make_specification(
            hypothesis=hypothesis,
            evaluation_rule=rule,
            alternatives=(alternative,),
        )

        assert specification.hypothesis is hypothesis
        assert specification.evaluation_rule is rule
        assert specification.alternatives[0] is alternative

    def test_multiple_alternatives_are_accepted(self):
        first = make_alternative()
        second = make_alternative()

        specification = make_specification(
            alternatives=(first, second),
        )

        assert specification.alternatives == (first, second)

    def test_invalid_hypothesis_is_rejected(self):
        with pytest.raises(TypeError):
            make_specification(hypothesis="not a hypothesis")

    def test_invalid_evaluation_rule_is_rejected(self):
        with pytest.raises(TypeError):
            make_specification(evaluation_rule="not a rule")

    def test_alternatives_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_specification(
                alternatives=[make_alternative()],
            )

    def test_empty_alternatives_are_rejected(self):
        with pytest.raises(ValueError):
            make_specification(alternatives=())

    def test_invalid_alternative_item_is_rejected(self):
        with pytest.raises(TypeError):
            make_specification(alternatives=("not an alternative",))

    def test_duplicate_alternative_ids_are_rejected(self):
        alternative = make_alternative()

        duplicate = HypothesisAlternative(
            alternative_id=alternative.alternative_id,
            proposition=make_proposition(
                proposition_type="different_explanation",
            ),
            description="Another explanation.",
        )

        with pytest.raises(ValueError):
            make_specification(
                alternatives=(alternative, duplicate),
            )

    @pytest.mark.parametrize("version", [999, True, 1.0])
    def test_invalid_schema_version_is_rejected(self, version):
        with pytest.raises(ValueError):
            make_specification(schema_version=version)

    def test_specification_is_immutable(self):
        specification = make_specification()

        with pytest.raises(FrozenInstanceError):
            specification.hypothesis = make_hypothesis()


class TestEvidenceSet:
    def test_valid_event_reference(self):
        reference = EvidenceReference(event_id=str(uuid4()))

        evidence_set = EvidenceSet(
            evidence_set_id=str(uuid4()),
            evidence=(reference,),
            created_at=datetime.now(timezone.utc),
        )

        assert evidence_set.evidence == (reference,)

    def test_valid_observation_reference(self):
        reference = EvidenceReference(observation_id=str(uuid4()))

        evidence_set = EvidenceSet(
            evidence_set_id=str(uuid4()),
            evidence=(reference,),
            created_at=datetime.now(timezone.utc),
        )

        assert evidence_set.evidence == (reference,)

    def test_mixed_reference_types_are_accepted(self):
        event_reference = EvidenceReference(event_id=str(uuid4()))
        observation_reference = EvidenceReference(
            observation_id=str(uuid4())
        )

        evidence_set = EvidenceSet(
            evidence_set_id=str(uuid4()),
            evidence=(event_reference, observation_reference),
            created_at=datetime.now(timezone.utc),
        )

        assert len(evidence_set.evidence) == 2

    def test_empty_evidence_is_allowed(self):
        evidence_set = EvidenceSet(
            evidence_set_id=str(uuid4()),
            evidence=(),
            created_at=datetime.now(timezone.utc),
        )

        assert evidence_set.evidence == ()

    def test_duplicate_references_are_rejected(self):
        reference = make_reference()

        with pytest.raises(ValueError):
            EvidenceSet(
                evidence_set_id=str(uuid4()),
                evidence=(reference, reference),
                created_at=datetime.now(timezone.utc),
            )

    def test_evidence_must_be_tuple(self):
        with pytest.raises(TypeError):
            EvidenceSet(
                evidence_set_id=str(uuid4()),
                evidence=[make_reference()],
                created_at=datetime.now(timezone.utc),
            )

    def test_evidence_items_must_be_references(self):
        with pytest.raises(TypeError):
            EvidenceSet(
                evidence_set_id=str(uuid4()),
                evidence=("invalid",),
                created_at=datetime.now(timezone.utc),
            )

    def test_invalid_evidence_set_id_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            EvidenceSet(
                evidence_set_id="invalid-id",
                evidence=(),
                created_at=datetime.now(timezone.utc),
            )

    def test_timestamp_is_normalized_to_utc(self):
        from datetime import timedelta, timezone

        offset = timezone(timedelta(hours=3))
        timestamp = datetime(2026, 11, 18, 12, 0, tzinfo=offset)

        evidence_set = EvidenceSet(
            evidence_set_id=str(uuid4()),
            evidence=(),
            created_at=timestamp,
        )

        assert evidence_set.created_at == datetime(
            2026, 11, 18, 9, 0, tzinfo=timezone.utc
        )
        assert evidence_set.created_at.utcoffset().total_seconds() == 0

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises(ValueError):
            EvidenceSet(
                evidence_set_id=str(uuid4()),
                evidence=(),
                created_at=datetime(2026, 11, 18, 12, 0),
            )

    @pytest.mark.parametrize("version", [999, True, 1.0])
    def test_invalid_schema_version_is_rejected(self, version):
        with pytest.raises(ValueError):
            EvidenceSet(
                evidence_set_id=str(uuid4()),
                evidence=(),
                created_at=datetime.now(timezone.utc),
                schema_version=version,
            )

    def test_evidence_set_is_immutable(self):
        evidence_set = EvidenceSet(
            evidence_set_id=str(uuid4()),
            evidence=(),
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(FrozenInstanceError):
            evidence_set.evidence = (make_reference(),)


def make_evaluation(**overrides):
    values = {
        "evaluation_id": str(uuid4()),
        "hypothesis_id": str(uuid4()),
        "evaluation_rule_id": "unauthorized-account-access-v1",
        "evidence_set_id": str(uuid4()),
        "assessment_id": str(uuid4()),
        "evaluated_at": datetime.now(timezone.utc),
        "criterion_results": (make_result(),),
    }
    values.update(overrides)
    return HypothesisEvaluation(**values)


class TestHypothesisEvaluation:
    def test_valid_evaluation(self):
        evaluation = make_evaluation()

        assert evaluation.evaluation_id
        assert evaluation.hypothesis_id
        assert evaluation.evidence_set_id
        assert evaluation.assessment_id
        assert len(evaluation.criterion_results) == 1

    @pytest.mark.parametrize(
        "field_name",
        [
            "evaluation_id",
            "hypothesis_id",
            "evidence_set_id",
            "assessment_id",
        ],
    )
    def test_invalid_ids_are_rejected(self, field_name):
        with pytest.raises(ValueError):
            make_evaluation(**{field_name: "invalid-id"})

    def test_timestamp_is_normalized_to_utc(self):
        from datetime import timedelta

        offset = timezone(timedelta(hours=3))
        timestamp = datetime(2026, 11, 18, 12, 0, tzinfo=offset)

        evaluation = make_evaluation(evaluated_at=timestamp)

        assert evaluation.evaluated_at == datetime(
            2026, 11, 18, 9, 0, tzinfo=timezone.utc
        )

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises(ValueError):
            make_evaluation(
                evaluated_at=datetime(2026, 11, 18, 12, 0)
            )

    def test_criterion_results_must_be_tuple(self):
        with pytest.raises(TypeError):
            make_evaluation(criterion_results=[make_result()])

    def test_empty_criterion_results_are_rejected(self):
        with pytest.raises(ValueError):
            make_evaluation(criterion_results=())

    def test_invalid_criterion_result_type_is_rejected(self):
        with pytest.raises(TypeError):
            make_evaluation(criterion_results=("invalid",))

    def test_duplicate_criterion_ids_are_rejected(self):
        with pytest.raises(ValueError):
            make_evaluation(
                criterion_results=(make_result(), make_result())
            )

    @pytest.mark.parametrize("version", [999, True, 1.0])
    def test_invalid_schema_version_is_rejected(self, version):
        with pytest.raises(ValueError):
            make_evaluation(schema_version=version)

    def test_evaluation_is_immutable(self):
        evaluation = make_evaluation()

        with pytest.raises(FrozenInstanceError):
            evaluation.assessment_id = str(uuid4())

    @pytest.mark.parametrize("rule_id", ["", "   "])
    def test_empty_evaluation_rule_id_is_rejected(self, rule_id):
        with pytest.raises(ValueError):
            make_evaluation(evaluation_rule_id=rule_id)

    def test_non_string_evaluation_rule_id_is_rejected(self):
        with pytest.raises(TypeError):
            make_evaluation(evaluation_rule_id=123)
