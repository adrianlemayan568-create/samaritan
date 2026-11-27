"""E002 regression: orchestration evaluates each alternative on its own evidence.

Synthetic-only test. The focal proposition and EvidenceSet remain constant
while alternative-specific evidence changes.
"""

from dataclasses import replace
from time import perf_counter

import pytest

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import new_id
from samaritan.reasoning_contracts import (
    CriterionFinding,
    HypothesisAlternative,
)
from samaritan.proposition_validation import PropositionProvenanceError

from test_e001_authentication_sequence import (
    _all_event_ids,
    _context,
    _constructor,
    _evaluate_neutrally,
    _evidence_set,
    _ingest_specs,
    _pipeline,
    _sequence,
)


def test_e002_orchestration_evaluates_alternative_against_own_evidence(
    tmp_path, record_property
):
    repository, source, service = _context(tmp_path)

    # One fixed focal sequence is used in both comparisons.
    focal_ids, focal_observation_id = _ingest_specs(
        service, source, _sequence(account="account-A")
    )
    _, generated = _pipeline(repository, focal_ids)
    assert len(generated.hypotheses) == 1

    focal_set = _evidence_set(
        _all_event_ids(focal_ids), focal_observation_id
    )
    constructor, semantics = _constructor(repository)
    focal = constructor.construct(
        focal_set, "account_authentication_sequence", 1
    )

    # Two alternative sequences differ only in the normalized result value.
    alternative_specs = (
        ("alternative-failure", "failure", "not_equal"),
        ("alternative-success", "success", "equal"),
    )
    comparisons = []
    stable_alternative_id = new_id()

    for label, result_value, expected_alternative_outcome in alternative_specs:
        alternative_ids, alternative_observation_id = _ingest_specs(
            service,
            source,
            _sequence(
                account="account-B",
                ip="198.51.100.20",
                success_data={"result": result_value},
            ),
        )
        alternative_set = _evidence_set(
            _all_event_ids(alternative_ids), alternative_observation_id
        )
        alternative = constructor.construct(
            alternative_set, "account_authentication_sequence", 1
        )

        assert alternative.proposition_id != focal.proposition_id
        assert alternative_set.evidence_set_id != focal_set.evidence_set_id
        assert alternative.evidence_manifest.evidence_set_id == (
            alternative_set.evidence_set_id
        )
        assert alternative.evidence_manifest.evidence == alternative_set.evidence

        alternative_contract = HypothesisAlternative(
            alternative_id=stable_alternative_id,
            proposition=alternative,
            description=f"Synthetic alternative with result={result_value}.",
        )

        engine, specification = _evaluate_neutrally(
            repository,
            semantics,
            focal,
            focal_set,
            alternative_contract,
            generated.hypotheses[0],
        )
        criterion = specification.evaluation_rule.criterion_definitions[0]

        # Independent evaluator control: evaluate each proposition against
        # its own separately persisted EvidenceSet.
        started = perf_counter()
        focal_finding = engine.criterion_engine.evaluate(
            criterion, focal, focal_set
        )
        focal_evaluation_seconds = perf_counter() - started

        started = perf_counter()
        alternative_finding = engine.criterion_engine.evaluate(
            criterion, alternative, alternative_set
        )
        alternative_evaluation_seconds = perf_counter() - started

        focal_success_event_id = focal_ids["authentication_success"][0]
        alternative_success_event_id = alternative_ids[
            "authentication_success"
        ][0]

        assert focal_finding.status == "matched"
        assert focal_finding.outcome_code == "equal"
        assert focal_finding.evidence == (
            EvidenceReference(event_id=focal_success_event_id),
            EvidenceReference(observation_id=focal_observation_id),
        )

        assert alternative_finding.outcome_code == expected_alternative_outcome
        assert alternative_finding.evidence == (
            EvidenceReference(event_id=alternative_success_event_id),
            EvidenceReference(observation_id=alternative_observation_id),
        )

        # System under test: orchestration evaluates the focal proposition
        # and the alternative against their separate authenticated selections.
        started = perf_counter()
        record = engine.evaluate(specification, focal_set, production=False)
        orchestration_seconds = perf_counter() - started
        repeated_record = engine.evaluate(
            specification, focal_set, production=False
        )

        assert len(record.findings) == 1
        assert record.findings[0] == focal_finding
        assert record.findings[0].outcome_code == "equal"
        assert len(record.alternative_evaluations) == 1
        alternative_evaluation = record.alternative_evaluations[0]
        assert alternative_evaluation.alternative_id == stable_alternative_id
        assert alternative_evaluation.proposition_id == alternative.proposition_id
        assert alternative_evaluation.evidence_set_id == alternative_set.evidence_set_id
        assert alternative_evaluation.findings == (alternative_finding,)
        assert alternative_evaluation.findings[0].outcome_code == (
            expected_alternative_outcome
        )
        assert record.evaluation_id == repeated_record.evaluation_id
        assert (
            alternative_evaluation.findings[0].finding_id
            == repeated_record.alternative_evaluations[0].findings[0].finding_id
        )
        # C-05A preserves the existing focal-only synthesis policy. Independent
        # alternative findings are recorded, not yet interpreted comparatively.
        assert record.synthesis.outcome == "inconclusive"
        assert record.synthesis.confidence_ceiling == "low"
        assert all(
            standing.standing == "non_discriminating"
            for standing in record.synthesis.alternative_standings
        )

        comparisons.append(
            {
                "label": label,
                "alternative_outcome": alternative_evaluation.findings[0].outcome_code,
                "orchestration_finding": record.findings[0].outcome_code,
                "synthesis_outcome": record.synthesis.outcome,
                "confidence_ceiling": record.synthesis.confidence_ceiling,
            }
        )

        record_property(
            f"e002.{label}.independent_alternative_evaluation_seconds",
            alternative_evaluation_seconds,
        )
        record_property(
            f"e002.{label}.independent_focal_evaluation_seconds",
            focal_evaluation_seconds,
        )
        record_property(
            f"e002.{label}.orchestration_seconds",
            orchestration_seconds,
        )

    # Alternative-specific orchestration findings now change with their own
    # evidence. Focal findings and C-05A's intentionally unchanged synthesis do not.
    assert [item["alternative_outcome"] for item in comparisons] == [
        "not_equal",
        "equal",
    ]
    assert [item["orchestration_finding"] for item in comparisons] == [
        "equal",
        "equal",
    ]
    assert [item["synthesis_outcome"] for item in comparisons] == [
        "inconclusive",
        "inconclusive",
    ]
    assert [item["confidence_ceiling"] for item in comparisons] == [
        "low",
        "low",
    ]

    record_property("e002.comparison_count", len(comparisons))
    record_property("e002.focal_evidence_set_unchanged", True)



def test_finding_evidence_is_limited_to_each_propositions_authenticated_selection(
    tmp_path,
):
    repository, source, service = _context(tmp_path)

    focal_specs = _sequence()
    focal_specs.append(
        {
            "event_type": "unmapped_sibling",
            "minute": 6,
            "account": "account-A",
            "ip": "192.0.2.10",
            "data": {"kind": "unselected sibling event"},
        }
    )
    focal_ids, focal_observation_id = _ingest_specs(
        service, source, focal_specs
    )
    _, generated = _pipeline(repository, focal_ids)
    assert len(generated.hypotheses) == 1

    focal_event_ids = tuple(
        focal_ids[event_type][0]
        for event_type in (
            "authentication_failure",
            "authentication_success",
            "host_activity",
        )
    )
    focal_set = _evidence_set(focal_event_ids)
    constructor, semantics = _constructor(repository)
    focal = constructor.construct(
        focal_set, "account_authentication_sequence", 1
    )

    alternative_ids, alternative_observation_id = _ingest_specs(
        service, source, _sequence(account="account-B", ip="198.51.100.20")
    )
    alternative_set = _evidence_set(
        _all_event_ids(alternative_ids), alternative_observation_id
    )
    alternative = constructor.construct(
        alternative_set, "account_authentication_sequence", 1
    )
    alternative_contract = HypothesisAlternative(
        alternative_id=new_id(),
        proposition=alternative,
        description="A separately selected synthetic alternative.",
    )

    engine, specification = _evaluate_neutrally(
        repository,
        semantics,
        focal,
        focal_set,
        alternative_contract,
        generated.hypotheses[0],
    )
    record = engine.evaluate(specification, focal_set, production=False)
    focal_finding = record.findings[0]
    alternative_finding = record.alternative_evaluations[0].findings[0]
    validator = engine.provenance_validator

    # Event-only selection authorizes the selected event and its parent
    # observation as provenance, but not another event from that observation.
    selected_success_id = focal_ids["authentication_success"][0]
    validator.validate_finding_evidence(
        focal,
        (
            replace(
                focal_finding,
                evidence=(
                    EvidenceReference(event_id=selected_success_id),
                    EvidenceReference(observation_id=focal_observation_id),
                ),
            ),
        ),
    )
    with pytest.raises(
        PropositionProvenanceError,
        match="event is outside the proposition evidence selection",
    ):
        validator.validate_finding_evidence(
            focal,
            (
                replace(
                    focal_finding,
                    evidence=(
                        EvidenceReference(
                            event_id=focal_ids["unmapped_sibling"][0]
                        ),
                    ),
                ),
            ),
        )

    # A different, real observation is still unauthorized for the focal finding.
    with pytest.raises(
        PropositionProvenanceError,
        match="observation is outside the proposition provenance selection",
    ):
        validator.validate_finding_evidence(
            focal,
            (
                replace(
                    focal_finding,
                    evidence=(
                        EvidenceReference(
                            observation_id=alternative_observation_id
                        ),
                    ),
                ),
            ),
        )

    # Alternative findings must be checked against the alternative's own
    # manifest; focal evidence cannot be borrowed across proposition boundaries.
    with pytest.raises(
        PropositionProvenanceError,
        match="event is outside the proposition evidence selection",
    ):
        validator.validate_finding_evidence(
            alternative,
            (
                replace(
                    alternative_finding,
                    evidence=(EvidenceReference(event_id=selected_success_id),),
                ),
            ),
        )

    # Observation selection authorizes its resolved events, unlike event-only
    # selection. This event belongs to the selected observation.
    observation_set = _evidence_set((), focal_observation_id)
    observation_selected_proposition = constructor.construct(
        observation_set, "account_authentication_sequence", 1
    )
    validator.validate_finding_evidence(
        observation_selected_proposition,
        (
            replace(
                focal_finding,
                evidence=(EvidenceReference(event_id=selected_success_id),),
            ),
        ),
    )

    # An insufficient finding may have no evidence, but any supplied reference
    # would still be subject to the same manifest checks.
    insufficient = CriterionFinding(
        finding_id=new_id(),
        criterion_id=focal_finding.criterion_id,
        status="insufficient",
        outcome_code=None,
        evidence=(),
        rationale="The synthetic source lacks a complete result.",
        missing_information=("A complete result value",),
    )
    validator.validate_finding_evidence(focal, (insufficient,))


def test_finding_evidence_is_isolated_between_simultaneous_alternatives(
    tmp_path,
):
    repository, source, service = _context(tmp_path)

    focal_ids, focal_observation_id = _ingest_specs(
        service, source, _sequence(account="account-A")
    )
    _, generated = _pipeline(repository, focal_ids)
    assert len(generated.hypotheses) == 1

    focal_set = _evidence_set(
        _all_event_ids(focal_ids), focal_observation_id
    )
    constructor, semantics = _constructor(repository)
    focal = constructor.construct(
        focal_set, "account_authentication_sequence", 1
    )

    alternative_inputs = (
        ("account-B", "198.51.100.20", "success"),
        ("account-C", "203.0.113.30", "failure"),
    )
    alternatives = []
    alternative_event_ids = []
    alternative_evidence_set_ids = []

    for account, ip, result_value in alternative_inputs:
        event_ids, observation_id = _ingest_specs(
            service,
            source,
            _sequence(
                account=account,
                ip=ip,
                success_data={"result": result_value},
            ),
        )
        evidence_set = _evidence_set(
            _all_event_ids(event_ids), observation_id
        )
        proposition = constructor.construct(
            evidence_set, "account_authentication_sequence", 1
        )
        alternatives.append(
            HypothesisAlternative(
                alternative_id=new_id(),
                proposition=proposition,
                description=f"Separately constructed synthetic alternative for {account}.",
            )
        )
        alternative_event_ids.append(
            event_ids["authentication_success"][0]
        )
        alternative_evidence_set_ids.append(evidence_set.evidence_set_id)

    alternative_contracts = tuple(alternatives)
    assert len({item.alternative_id for item in alternative_contracts}) == 2
    assert len({item.proposition.proposition_id for item in alternative_contracts}) == 2
    assert len(set(alternative_evidence_set_ids)) == 2
    assert set(alternative_event_ids).isdisjoint(_all_event_ids(focal_ids))

    engine, specification = _evaluate_neutrally(
        repository,
        semantics,
        focal,
        focal_set,
        alternative_contracts,
        generated.hypotheses[0],
    )
    assert specification.alternatives == alternative_contracts

    record = engine.evaluate(specification, focal_set, production=False)
    assert len(record.alternative_evaluations) == 2

    validator = engine.provenance_validator
    findings_by_alternative = {}
    for index, alternative in enumerate(alternative_contracts):
        evaluation = record.alternative_evaluations[index]
        assert evaluation.alternative_id == alternative.alternative_id
        assert evaluation.proposition_id == alternative.proposition.proposition_id
        assert evaluation.evidence_set_id == alternative_evidence_set_ids[index]
        assert len(evaluation.findings) == 1

        finding = evaluation.findings[0]
        assert finding.evidence[0] == EvidenceReference(
            event_id=alternative_event_ids[index]
        )
        validator.validate_finding_evidence(
            alternative.proposition, (finding,)
        )
        findings_by_alternative[alternative.alternative_id] = finding

    # Each alternative's own selected event is accepted, but the other
    # alternative's event must not cross the proposition boundary.
    finding_a = findings_by_alternative[alternative_contracts[0].alternative_id]
    finding_b = findings_by_alternative[alternative_contracts[1].alternative_id]
    with pytest.raises(
        PropositionProvenanceError,
        match="event is outside the proposition evidence selection",
    ):
        validator.validate_finding_evidence(
            alternative_contracts[0].proposition,
            (
                replace(
                    finding_a,
                    evidence=(
                        EvidenceReference(event_id=alternative_event_ids[1]),
                    ),
                ),
            ),
        )

    with pytest.raises(
        PropositionProvenanceError,
        match="event is outside the proposition evidence selection",
    ):
        validator.validate_finding_evidence(
            alternative_contracts[1].proposition,
            (
                replace(
                    finding_b,
                    evidence=(
                        EvidenceReference(event_id=alternative_event_ids[0]),
                    ),
                ),
            ),
        )
