"""Tests for Samaritan's telemetry ingestion service."""

import json
import sqlite3

from datetime import datetime, timezone

import pytest

from samaritan.contracts import Source, new_id
from samaritan.database import Database
from samaritan.ingestion import IngestionService
from samaritan.repository import Repository


NOW = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def setup_ingestion(tmp_path):
    database = Database(tmp_path / "samaritan.db")
    database.initialize()

    repository = Repository(database)

    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="ingestion test source",
        created_at=NOW,
    )
    repository.register_source(source)

    service = IngestionService(repository)

    return service, repository, database, source


def make_envelope(
    entities=None,
    events=None,
    relationships=None,
):
    return {
        "format_version": 1,
        "entities": entities or [],
        "events": events or [],
        "relationships": relationships or [],
    }


def make_entity(entity_id=None, identifier="192.0.2.10"):
    return {
        "entity_id": entity_id or new_id(),
        "entity_type": "ip_address",
        "canonical_identifier": identifier,
    }


def make_event(event_id=None, entity_id=None):
    participants = []
    if entity_id:
        participants.append({
            "entity_id": entity_id,
            "role": "source",
        })

    return {
        "event_id": event_id or new_id(),
        "event_type": "authentication_failure",
        "original_timestamp": "2026-11-16T10:00:00Z",
        "normalized_data": {
            "username": "test-user",
            "result": "failure",
        },
        "participants": participants,
    }


def counts(database):
    with database.connection() as connection:
        tables = (
            "observations",
            "quarantined_observations",
            "entities",
            "events",
            "event_participants",
            "relationships",
        )

        return {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in tables
        }


def test_service_requires_repository():
    with pytest.raises(TypeError, match="Repository"):
        IngestionService("not a repository")


def test_accepts_empty_valid_envelope(setup_ingestion):
    service, _, database, source = setup_ingestion

    raw = json.dumps(make_envelope())

    result = service.ingest(source.source_id, raw, NOW)

    assert result.accepted is True
    assert result.observation_id is not None
    assert result.quarantine_id is None

    actual = counts(database)
    assert actual["observations"] == 1
    assert actual["quarantined_observations"] == 0


def test_preserves_exact_raw_payload(setup_ingestion):
    service, _, database, source = setup_ingestion

    raw = (
        '{\n'
        '  "format_version": 1,\n'
        '  "entities": [],\n'
        '  "events": [],\n'
        '  "relationships": []\n'
        '}\n'
    )

    result = service.ingest(source.source_id, raw, NOW)

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT raw_payload
            FROM observations
            WHERE observation_id = ?
            """,
            (result.observation_id,),
        ).fetchone()

    assert row["raw_payload"] == raw


def test_persists_entities_events_participants_and_relationships(
    setup_ingestion,
):
    service, _, database, source = setup_ingestion

    first_id = new_id()
    second_id = new_id()

    envelope = make_envelope(
        entities=[
            make_entity(first_id, "192.0.2.10"),
            make_entity(second_id, "192.0.2.20"),
        ],
        events=[
            make_event(entity_id=first_id),
        ],
        relationships=[
            {
                "relationship_id": new_id(),
                "source_entity_id": first_id,
                "target_entity_id": second_id,
                "relationship_type": "communicates_with",
                "attributes": {"protocol": "tcp"},
            }
        ],
    )

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is True
    assert result.resolved_entity_ids[first_id] == first_id
    assert result.resolved_entity_ids[second_id] == second_id

    actual = counts(database)

    assert actual["observations"] == 1
    assert actual["entities"] == 2
    assert actual["events"] == 1
    assert actual["event_participants"] == 1
    assert actual["relationships"] == 1

    with database.connection() as connection:
        event = connection.execute(
            "SELECT observation_id FROM events"
        ).fetchone()

        relationship = connection.execute(
            """
            SELECT observation_id, observed_at, attributes_json
            FROM relationships
            """
        ).fetchone()

        participant = connection.execute(
            "SELECT entity_id, role FROM event_participants"
        ).fetchone()

    assert event["observation_id"] == result.observation_id
    assert relationship["observation_id"] == result.observation_id
    assert relationship["attributes_json"] == '{"protocol":"tcp"}'
    assert participant["entity_id"] == first_id
    assert participant["role"] == "source"


def test_malformed_json_is_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    raw = '{"format_version": '

    result = service.ingest(source.source_id, raw, NOW)

    assert result.accepted is False
    assert result.rejection_code == "invalid_payload"
    assert result.quarantine_id is not None

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT raw_payload, rejection_code
            FROM quarantined_observations
            WHERE quarantine_id = ?
            """,
            (result.quarantine_id,),
        ).fetchone()

    assert row["raw_payload"] == raw
    assert row["rejection_code"] == "invalid_payload"

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["quarantined_observations"] == 1


def test_duplicate_json_keys_are_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    raw = (
        '{"format_version":1,'
        '"format_version":1,'
        '"entities":[],"events":[],"relationships":[]}'
    )

    result = service.ingest(source.source_id, raw, NOW)

    assert result.accepted is False
    assert result.rejection_code == "invalid_payload"

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["quarantined_observations"] == 1


def test_nonstandard_json_constants_are_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    raw = (
        '{"format_version":1,"entities":[],"events":[],'
        '"relationships":[],"unexpected":NaN}'
    )

    result = service.ingest(source.source_id, raw, NOW)

    assert result.accepted is False
    assert result.rejection_code == "invalid_payload"

    actual = counts(database)
    assert actual["quarantined_observations"] == 1


def test_unsupported_format_version_is_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    raw = json.dumps({
        "format_version": 2,
        "entities": [],
        "events": [],
        "relationships": [],
    })

    result = service.ingest(source.source_id, raw, NOW)

    assert result.accepted is False
    assert result.rejection_code == "invalid_payload"

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["quarantined_observations"] == 1


def test_invalid_entity_is_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    envelope = make_envelope(
        entities=[{
            "entity_id": new_id(),
            "entity_type": "ip_address",
            "canonical_identifier": " ",
        }]
    )

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is False
    assert result.rejection_code == "invalid_record"

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["entities"] == 0
    assert actual["quarantined_observations"] == 1


def test_existing_entity_identity_is_reused(setup_ingestion):
    service, _, database, source = setup_ingestion

    first_entity_id = new_id()

    first = make_envelope(
        entities=[make_entity(first_entity_id)]
    )

    first_result = service.ingest(
        source.source_id,
        json.dumps(first),
        NOW,
    )

    assert first_result.accepted is True

    second_entity_id = new_id()

    second = make_envelope(
        entities=[make_entity(second_entity_id)],
        events=[make_event(entity_id=second_entity_id)],
    )

    second_result = service.ingest(
        source.source_id,
        json.dumps(second),
        NOW,
    )

    assert second_result.accepted is True
    assert (
        second_result.resolved_entity_ids[second_entity_id]
        == first_entity_id
    )

    actual = counts(database)
    assert actual["entities"] == 1
    assert actual["observations"] == 2
    assert actual["events"] == 1

    with database.connection() as connection:
        participant = connection.execute(
            "SELECT entity_id FROM event_participants"
        ).fetchone()

    assert participant["entity_id"] == first_entity_id


def test_duplicate_observation_id_propagates_database_error(
    setup_ingestion,
):
    service, _, database, source = setup_ingestion

    observation_id = new_id()
    raw = json.dumps(make_envelope())

    first = service.ingest(
        source.source_id,
        raw,
        NOW,
        observation_id=observation_id,
    )

    assert first.accepted is True

    with pytest.raises(sqlite3.IntegrityError):
        service.ingest(
            source.source_id,
            raw,
            NOW,
            observation_id=observation_id,
        )

    actual = counts(database)

    assert actual["observations"] == 1
    assert actual["quarantined_observations"] == 0


def test_missing_participant_entity_rolls_back_entire_observation(
    setup_ingestion,
):
    service, _, database, source = setup_ingestion

    missing_entity_id = new_id()

    envelope = make_envelope(
        entities=[make_entity()],
        events=[make_event(entity_id=missing_entity_id)],
    )

    with pytest.raises(sqlite3.IntegrityError):
        service.ingest(
            source.source_id,
            json.dumps(envelope),
            NOW,
        )

    actual = counts(database)

    assert actual["observations"] == 0
    assert actual["entities"] == 0
    assert actual["events"] == 0
    assert actual["event_participants"] == 0
    assert actual["quarantined_observations"] == 0


def test_invalid_relationship_endpoint_rolls_back_observation(
    setup_ingestion,
):
    service, _, database, source = setup_ingestion

    first_id = new_id()
    missing_id = new_id()

    envelope = make_envelope(
        entities=[make_entity(first_id)],
        relationships=[{
            "relationship_id": new_id(),
            "source_entity_id": first_id,
            "target_entity_id": missing_id,
            "relationship_type": "communicates_with",
        }],
    )

    with pytest.raises(sqlite3.IntegrityError):
        service.ingest(
            source.source_id,
            json.dumps(envelope),
            NOW,
        )

    actual = counts(database)

    assert actual["observations"] == 0
    assert actual["entities"] == 0
    assert actual["relationships"] == 0
    assert actual["quarantined_observations"] == 0


def test_unknown_source_database_failure_is_not_quarantined(
    setup_ingestion,
):
    service, _, database, source = setup_ingestion

    unknown_source_id = new_id()
    raw = '{"invalid":'

    with pytest.raises(sqlite3.IntegrityError):
        service.ingest(unknown_source_id, raw, NOW)

    actual = counts(database)

    assert actual["observations"] == 0
    assert actual["quarantined_observations"] == 0


def test_caller_supplied_observation_id_is_preserved(setup_ingestion):
    service, _, database, source = setup_ingestion

    observation_id = new_id()

    result = service.ingest(
        source.source_id,
        json.dumps(make_envelope()),
        NOW,
        observation_id=observation_id,
    )

    assert result.accepted is True
    assert result.observation_id == observation_id

    with database.connection() as connection:
        row = connection.execute(
            "SELECT observation_id FROM observations"
        ).fetchone()

    assert row["observation_id"] == observation_id


def test_duplicate_event_ids_in_envelope_are_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    event = make_event()

    envelope = make_envelope(events=[event, dict(event)])

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is False
    assert result.rejection_code == "invalid_payload"
    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["events"] == 0
    assert actual["quarantined_observations"] == 1


def test_duplicate_relationship_ids_in_envelope_are_quarantined(
    setup_ingestion,
):
    service, _, database, source = setup_ingestion

    first_id = new_id()
    second_id = new_id()
    relationship = {
        "relationship_id": new_id(),
        "source_entity_id": first_id,
        "target_entity_id": second_id,
        "relationship_type": "communicates_with",
    }

    envelope = make_envelope(
        entities=[
            make_entity(first_id, "192.0.2.10"),
            make_entity(second_id, "192.0.2.20"),
        ],
        relationships=[relationship, dict(relationship)],
    )

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is False
    assert result.rejection_code == "invalid_payload"
    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["relationships"] == 0
    assert actual["quarantined_observations"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_events",
        "unexpected_field",
    ],
)
def test_invalid_envelope_fields_are_quarantined(
    setup_ingestion,
    mutation,
):
    service, _, database, source = setup_ingestion
    envelope = make_envelope()

    if mutation == "missing_events":
        del envelope["events"]
    else:
        envelope["unexpected"] = "not allowed"

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is False
    assert result.quarantine_id is not None

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["quarantined_observations"] == 1


def test_invalid_event_timestamp_is_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    event = make_event()
    event["original_timestamp"] = "not-a-timestamp"

    result = service.ingest(
        source.source_id,
        json.dumps(make_envelope(events=[event])),
        NOW,
    )

    assert result.accepted is False
    assert result.quarantine_id is not None

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["events"] == 0
    assert actual["quarantined_observations"] == 1


def test_self_referencing_relationship_is_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    entity_id = new_id()

    envelope = make_envelope(
        entities=[make_entity(entity_id)],
        relationships=[
            {
                "relationship_id": new_id(),
                "source_entity_id": entity_id,
                "target_entity_id": entity_id,
                "relationship_type": "communicates_with",
            }
        ],
    )

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is False
    assert result.quarantine_id is not None

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["relationships"] == 0
    assert actual["quarantined_observations"] == 1


def test_non_object_normalized_data_is_quarantined(setup_ingestion):
    service, _, database, source = setup_ingestion

    event = make_event()
    event["normalized_data"] = ["not", "an", "object"]

    result = service.ingest(
        source.source_id,
        json.dumps(make_envelope(events=[event])),
        NOW,
    )

    assert result.accepted is False
    assert result.quarantine_id is not None

    actual = counts(database)
    assert actual["observations"] == 0
    assert actual["events"] == 0
    assert actual["quarantined_observations"] == 1


def test_multiple_events_are_persisted(setup_ingestion):
    service, _, database, source = setup_ingestion

    envelope = make_envelope(
        events=[
            make_event(),
            make_event(),
            make_event(),
        ],
    )

    result = service.ingest(
        source.source_id,
        json.dumps(envelope),
        NOW,
    )

    assert result.accepted is True

    actual = counts(database)
    assert actual["observations"] == 1
    assert actual["events"] == 3
    assert actual["quarantined_observations"] == 0
