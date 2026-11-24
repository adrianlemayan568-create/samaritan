"""E003: reconstructed evidence-integrity experiment.

Synthetic-only tests reconstructed from the surviving E003 protocol and findings.
This is not the lost original test file. No production source is modified.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import new_id, parse_timestamp
from samaritan.correlation import CorrelationEngine
from samaritan.database import Database
from samaritan.hypothesis_generation import HypothesisGenerationEngine
from samaritan.hypothesis_generation_contracts import HypothesisGenerationRule
from samaritan.ingestion import IngestionService

from test_e001_authentication_sequence import (
    _all_event_ids,
    _context,
    _constructor,
    _correlation_rules,
    _evidence_set,
    _ingest_specs,
    _sequence,
)


NOW = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)


def _raw_event_payload(event_id: str, timestamp: str, data: dict) -> str:
    return json.dumps(
        {
            "format_version": 1,
            "entities": [],
            "events": [
                {
                    "event_id": event_id,
                    "event_type": "integrity_probe",
                    "original_timestamp": timestamp,
                    "normalized_data": data,
                    "participants": [],
                }
            ],
            "relationships": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _ingest_single(
    service,
    source,
    *,
    event_type: str,
    minute: int,
    received_at: datetime,
    account: str = "account-A",
    data: dict | None = None,
):
    """Ingest one event with independently controlled occurrence and receipt times."""
    if data is None:
        data = {"result": event_type}

    entity_id = new_id()
    event_id = new_id()
    participants = [
        {
            "entity_id": entity_id,
            "role": "subject" if event_type == "host_activity" else "target",
        }
    ]
    entities = [
        {
            "entity_id": entity_id,
            "entity_type": "user_account",
            "canonical_identifier": account,
        }
    ]
    if event_type == "host_activity":
        hostname_id = new_id()
        entities.append(
            {
                "entity_id": hostname_id,
                "entity_type": "hostname",
                "canonical_identifier": "synthetic-host.example.test",
            }
        )
        participants.append({"entity_id": hostname_id, "role": "target"})

    occurred_at = NOW + timedelta(minutes=minute)
    timestamp_text = occurred_at.isoformat().replace("+00:00", "Z")
    payload = {
        "format_version": 1,
        "entities": entities,
        "events": [
            {
                "event_id": event_id,
                "event_type": event_type,
                "original_timestamp": timestamp_text,
                "normalized_data": data,
                "participants": participants,
            }
        ],
        "relationships": [],
    }
    raw_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    result = service.ingest(source.source_id, raw_payload, received_at)
    assert result.accepted is True, result.rejection_reason
    return event_id, result.observation_id, timestamp_text


def _generate(repository, event_ids, minimum_correlations):
    correlations = CorrelationEngine(
        repository, _correlation_rules()
    ).correlate_events(event_ids)
    rule = HypothesisGenerationRule(
        rule_id=f"e003-minimum-{minimum_correlations}",
        required_correlation_rule_ids=(
            "e001-failure-success",
            "e001-success-host",
        ),
        minimum_correlations=minimum_correlations,
        statement="The selected synthetic events form a candidate authentication sequence.",
        description="E003 controlled correlation-count comparison.",
    )
    generated = HypothesisGenerationEngine(repository, (rule,)).generate(
        correlations
    )
    return correlations, generated


def test_e003_case_a_event_id_collision_rolls_back_observation_transaction(tmp_path):
    repository, source, service = _context(tmp_path)
    event_id = new_id()
    timestamp = NOW.isoformat().replace("+00:00", "Z")

    first = service.ingest(
        source.source_id,
        _raw_event_payload(event_id, timestamp, {"value": "original"}),
        NOW,
    )
    assert first.accepted is True
    original_observation = repository.get_observation(first.observation_id)
    original_event = repository.get_event(event_id)
    assert original_observation is not None
    assert original_event is not None

    with pytest.raises(sqlite3.IntegrityError):
        service.ingest(
            source.source_id,
            _raw_event_payload(event_id, timestamp, {"value": "conflicting"}),
            NOW + timedelta(minutes=1),
        )

    observations = repository.list_observations()
    assert len(observations) == 1
    assert observations[0] == original_observation
    assert repository.get_observation(first.observation_id) == original_observation
    assert repository.get_event(event_id) == original_event
    assert original_event.observation_id == first.observation_id


def test_e003_case_b_distinct_record_identity_and_controlled_content_comparison(
    tmp_path, record_property
):
    scenarios = {}

    # Baseline: one failure, success, host activity.
    for label, failure_specs in (
        ("baseline", ({"result": "failure", "detail": "same"},)),
        (
            "duplicate_content",
            (
                {"result": "failure", "detail": "same"},
                {"result": "failure", "detail": "same"},
            ),
        ),
        (
            "different_content_control",
            (
                {"result": "failure", "detail": "same"},
                {"result": "failure", "detail": "changed"},
            ),
        ),
    ):
        scenario_path = tmp_path / label
        scenario_path.mkdir()
        repository, source, service = _context(scenario_path)
        specs = [
            {
                "event_type": "authentication_failure",
                "minute": 0,
                "account": "account-A",
                "ip": "192.0.2.10",
                "data": data,
            }
            for data in failure_specs
        ]
        specs.extend(
            _sequence(
                account="account-A",
                ip="192.0.2.10",
                failure_minutes=(),
            )
        )
        # _sequence with no failures contributes only success and host events.
        event_ids_by_type, observation_id = _ingest_specs(
            service, source, specs
        )
        event_ids = _all_event_ids(event_ids_by_type)
        correlations, generated = _generate(
            repository, event_ids, minimum_correlations=3
        )
        failures = repository.list_events(observation_id=observation_id)
        failure_events = tuple(
            event for event in failures
            if event.event_type == "authentication_failure"
        )
        scenarios[label] = {
            "correlation_count": len(correlations),
            "hypothesis_count": len(generated.hypotheses),
            "failure_events": failure_events,
            "event_ids": tuple(event.event_id for event in failure_events),
        }

    baseline = scenarios["baseline"]
    duplicate = scenarios["duplicate_content"]
    distinct = scenarios["different_content_control"]

    assert baseline["correlation_count"] == 2
    assert baseline["hypothesis_count"] == 0
    assert duplicate["correlation_count"] == 3
    assert duplicate["hypothesis_count"] == 1
    assert distinct["correlation_count"] == 3
    assert distinct["hypothesis_count"] == 1

    duplicate_events = duplicate["failure_events"]
    distinct_events = distinct["failure_events"]
    assert len(duplicate_events) == len(distinct_events) == 2
    assert duplicate_events[0].event_id != duplicate_events[1].event_id
    assert duplicate_events[0].normalized_data_json == duplicate_events[1].normalized_data_json
    assert duplicate_events[0].occurred_at == duplicate_events[1].occurred_at
    assert distinct_events[0].event_id != distinct_events[1].event_id
    assert distinct_events[0].occurred_at == distinct_events[1].occurred_at
    assert distinct_events[0].normalized_data_json != distinct_events[1].normalized_data_json

    # The controlled content change does not alter correlation matching:
    # the engine uses event types, shared participant entities, and time.
    assert duplicate["correlation_count"] == distinct["correlation_count"]
    assert duplicate["hypothesis_count"] == distinct["hypothesis_count"]

    record_property("e003.case_b.baseline_correlations", baseline["correlation_count"])
    record_property("e003.case_b.duplicate_content_correlations", duplicate["correlation_count"])
    record_property("e003.case_b.distinct_content_correlations", distinct["correlation_count"])
    record_property("e003.case_b.control_changes_only_normalized_content", True)


def test_e003_case_c_conflicting_attributes_preserve_records_and_surface_ambiguity(
    tmp_path,
):
    repository, source, service = _context(tmp_path)
    first_specs = _sequence(
        account="account-A",
        ip="192.0.2.10",
        success_data={"result": "success"},
    )
    first_ids, first_observation_id = _ingest_specs(
        service, source, first_specs
    )

    conflicting_success, conflicting_observation_id = _ingest_specs(
        service,
        source,
        [
            {
                "event_type": "authentication_success",
                "minute": 2,
                "account": "account-A",
                "ip": "192.0.2.10",
                "data": {"result": "failure"},
            }
        ],
    )

    first_success = repository.get_event(
        first_ids["authentication_success"][0]
    )
    second_success = repository.get_event(
        conflicting_success["authentication_success"][0]
    )
    assert first_success is not None
    assert second_success is not None
    assert first_success.event_id != second_success.event_id
    assert first_success.occurred_at == second_success.occurred_at
    assert first_success.normalized_data_json != second_success.normalized_data_json
    assert first_success.observation_id == first_observation_id
    assert second_success.observation_id == conflicting_observation_id

    event_ids = (
        *_all_event_ids(first_ids),
        *_all_event_ids(conflicting_success),
    )
    evidence_set = _evidence_set(event_ids)
    constructor, _ = _constructor(repository)

    with pytest.raises(
        Exception, match="event-role selection is ambiguous"
    ):
        constructor.construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_e003_case_d_future_event_timestamp_is_preserved_exactly(tmp_path):
    repository, source, service = _context(tmp_path)
    received_at = NOW
    event_id, observation_id, original_timestamp = _ingest_single(
        service,
        source,
        event_type="authentication_failure",
        minute=60,
        received_at=received_at,
    )

    event = repository.get_event(event_id)
    observation = repository.get_observation(observation_id)
    assert event is not None
    assert observation is not None

    expected_occurrence = parse_timestamp(
        original_timestamp, "original_timestamp"
    )
    assert event.original_timestamp == original_timestamp
    assert event.occurred_at == expected_occurrence
    assert event.occurred_at > observation.received_at
    assert observation.received_at == received_at
    assert observation.received_at == NOW


def test_e003_case_e_delayed_historical_event_retains_receipt_and_event_order(
    tmp_path,
):
    repository, source, service = _context(tmp_path)

    success_id, success_observation_id, _ = _ingest_single(
        service,
        source,
        event_type="authentication_success",
        minute=12,
        received_at=NOW + timedelta(minutes=13),
    )
    host_id, host_observation_id, _ = _ingest_single(
        service,
        source,
        event_type="host_activity",
        minute=14,
        received_at=NOW + timedelta(minutes=15),
    )
    failure_id, failure_observation_id, _ = _ingest_single(
        service,
        source,
        event_type="authentication_failure",
        minute=10,
        received_at=NOW + timedelta(minutes=30),
    )

    observations = repository.list_observations()
    events = repository.list_events()
    assert tuple(item.observation_id for item in observations) == (
        success_observation_id,
        host_observation_id,
        failure_observation_id,
    )
    assert tuple(item.event_id for item in events) == (
        failure_id,
        success_id,
        host_id,
    )

    delayed_failure = repository.get_event(failure_id)
    delayed_observation = repository.get_observation(failure_observation_id)
    assert delayed_failure is not None
    assert delayed_observation is not None
    assert delayed_failure.occurred_at < delayed_observation.received_at

    event_ids = (failure_id, success_id, host_id)
    correlations, generated = _generate(
        repository, event_ids, minimum_correlations=2
    )
    assert len(correlations) == 2
    assert len(generated.hypotheses) == 1

    evidence_set = _evidence_set(
        event_ids,
        observation_id=None,
    )
    constructor, _ = _constructor(repository)
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    assert proposition.proposition_type == "account_authentication_sequence"
