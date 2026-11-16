"""Tests for Samaritan's persistence repository."""

from datetime import datetime, timezone
import sqlite3

import pytest

from samaritan.contracts import (
    Entity,
    Event,
    EventParticipant,
    Observation,
    QuarantinedObservation,
    Relationship,
    Source,
    new_id,
    payload_checksum,
)
from samaritan.database import Database, DatabaseIntegrityError
from samaritan.repository import Repository


NOW = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)


def make_repository(tmp_path):
    database = Database(tmp_path / "samaritan.db")
    database.initialize()
    repository = Repository(database)

    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="repository test source",
        created_at=NOW,
    )
    repository.register_source(source)

    return repository, database, source


def make_observation(source, raw_payload="{}"):
    return Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=NOW,
        raw_payload=raw_payload,
        payload_sha256=payload_checksum(raw_payload),
    )


def make_entity(
    entity_type="ip_address",
    canonical_identifier="192.0.2.10",
):
    return Entity(
        entity_id=new_id(),
        entity_type=entity_type,
        canonical_identifier=canonical_identifier,
        created_at=NOW,
    )


def make_event(observation, participant_ids=()):
    return Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type="authentication_failure",
        occurred_at=NOW,
        original_timestamp="2026-11-16T10:00:00Z",
        normalized_data_json='{"result":"failure"}',
        participants=tuple(
            EventParticipant(entity_id=entity_id, role="source")
            for entity_id in participant_ids
        ),
    )


def test_repository_requires_database_instance(tmp_path):
    with pytest.raises(TypeError, match="Database"):
        Repository("not a database")


def test_register_source_persists_source(tmp_path):
    repository, database, source = make_repository(tmp_path)

    with database.connection() as connection:
        row = connection.execute(
            "SELECT * FROM sources WHERE source_id = ?",
            (source.source_id,),
        ).fetchone()

    assert row is not None
    assert row["source_type"] == "synthetic"
    assert row["name"] == "repository test source"


def test_duplicate_source_registration_is_rejected(tmp_path):
    repository, _, source = make_repository(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        repository.register_source(source)


def test_persist_observation_preserves_raw_payload(tmp_path):
    repository, database, source = make_repository(tmp_path)
    raw_payload = '{ "test": true, "number": 1 }\n'
    observation = make_observation(source, raw_payload)

    result = repository.persist_observation(observation)

    assert result == {}

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT raw_payload, payload_sha256
            FROM observations
            WHERE observation_id = ?
            """,
            (observation.observation_id,),
        ).fetchone()

    assert row["raw_payload"] == raw_payload
    assert row["payload_sha256"] == payload_checksum(raw_payload)


def test_persist_observation_and_entity(tmp_path):
    repository, database, source = make_repository(tmp_path)
    observation = make_observation(source)
    entity = make_entity()

    result = repository.persist_observation(
        observation,
        entities=(entity,),
    )

    assert result == {entity.entity_id: entity.entity_id}

    with database.connection() as connection:
        row = connection.execute(
            "SELECT entity_id FROM entities WHERE entity_id = ?",
            (entity.entity_id,),
        ).fetchone()

    assert row is not None


def test_existing_entity_is_reused_by_typed_identity(tmp_path):
    repository, database, source = make_repository(tmp_path)

    first_observation = make_observation(source)
    first_entity = make_entity()

    repository.persist_observation(
        first_observation,
        entities=(first_entity,),
    )

    second_observation = make_observation(source)
    second_entity = make_entity()

    result = repository.persist_observation(
        second_observation,
        entities=(second_entity,),
    )

    assert result == {
        second_entity.entity_id: first_entity.entity_id
    }

    with database.connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM entities
            WHERE entity_type = ?
              AND canonical_identifier = ?
            """,
            first_entity.identity_key,
        ).fetchone()[0]

    assert count == 1


def test_same_identifier_with_different_types_creates_distinct_entities(
    tmp_path,
):
    repository, database, source = make_repository(tmp_path)

    observation = make_observation(source)
    ip_entity = make_entity("ip_address", "shared-value")
    hostname_entity = make_entity("hostname", "shared-value")

    result = repository.persist_observation(
        observation,
        entities=(ip_entity, hostname_entity),
    )

    assert result == {
        ip_entity.entity_id: ip_entity.entity_id,
        hostname_entity.entity_id: hostname_entity.entity_id,
    }

    with database.connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM entities
            WHERE canonical_identifier = ?
            """,
            ("shared-value",),
        ).fetchone()[0]

    assert count == 2


def test_event_participant_uses_resolved_entity_id(tmp_path):
    repository, database, source = make_repository(tmp_path)

    first_observation = make_observation(source)
    existing_entity = make_entity()

    repository.persist_observation(
        first_observation,
        entities=(existing_entity,),
    )

    second_observation = make_observation(source)
    submitted_entity = make_entity()
    event = make_event(second_observation, (submitted_entity.entity_id,))

    result = repository.persist_observation(
        second_observation,
        entities=(submitted_entity,),
        events=(event,),
    )

    assert result[submitted_entity.entity_id] == existing_entity.entity_id

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT entity_id, role, participant_index
            FROM event_participants
            WHERE event_id = ?
            """,
            (event.event_id,),
        ).fetchone()

    assert row["entity_id"] == existing_entity.entity_id
    assert row["role"] == "source"
    assert row["participant_index"] == 0


def test_relationship_uses_resolved_entity_ids(tmp_path):
    repository, database, source = make_repository(tmp_path)

    first_observation = make_observation(source)
    original_source = make_entity("ip_address", "192.0.2.10")
    original_target = make_entity("hostname", "example.test")

    repository.persist_observation(
        first_observation,
        entities=(original_source, original_target),
    )

    second_observation = make_observation(source)
    submitted_source = make_entity("ip_address", "192.0.2.10")
    submitted_target = make_entity("hostname", "example.test")

    relationship = Relationship(
        relationship_id=new_id(),
        source_entity_id=submitted_source.entity_id,
        target_entity_id=submitted_target.entity_id,
        relationship_type="communicated_with",
        observation_id=second_observation.observation_id,
        observed_at=NOW,
    )

    repository.persist_observation(
        second_observation,
        entities=(submitted_source, submitted_target),
        relationships=(relationship,),
    )

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT source_entity_id, target_entity_id
            FROM relationships
            WHERE relationship_id = ?
            """,
            (relationship.relationship_id,),
        ).fetchone()

    assert row["source_entity_id"] == original_source.entity_id
    assert row["target_entity_id"] == original_target.entity_id


def test_missing_participant_entity_rolls_back_entire_observation(
    tmp_path,
):
    repository, database, source = make_repository(tmp_path)

    observation = make_observation(source)
    entity = make_entity()
    missing_entity_id = new_id()
    event = make_event(observation, (missing_entity_id,))

    with pytest.raises(sqlite3.IntegrityError, match="does not exist"):
        repository.persist_observation(
            observation,
            entities=(entity,),
            events=(event,),
        )

    with database.connection() as connection:
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        entity_count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()[0]

    assert observation_count == 0
    assert entity_count == 0
    assert event_count == 0


def test_duplicate_observation_id_is_rejected_without_overwrite(
    tmp_path,
):
    repository, database, source = make_repository(tmp_path)

    first = make_observation(source, '{"value":"first"}')
    repository.persist_observation(first)

    duplicate = Observation(
        observation_id=first.observation_id,
        source_id=source.source_id,
        received_at=NOW,
        raw_payload='{"value":"second"}',
        payload_sha256=payload_checksum('{"value":"second"}'),
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.persist_observation(duplicate)

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT raw_payload
            FROM observations
            WHERE observation_id = ?
            """,
            (first.observation_id,),
        ).fetchone()

    assert row["raw_payload"] == '{"value":"first"}'


def test_entity_id_collision_with_different_identity_rolls_back(
    tmp_path,
):
    repository, database, source = make_repository(tmp_path)

    first_observation = make_observation(source)
    original_entity = make_entity("ip_address", "192.0.2.10")

    repository.persist_observation(
        first_observation,
        entities=(original_entity,),
    )

    second_observation = make_observation(source)
    conflicting_entity = Entity(
        entity_id=original_entity.entity_id,
        entity_type="hostname",
        canonical_identifier="example.test",
        created_at=NOW,
    )

    with pytest.raises(DatabaseIntegrityError, match="different entity identity"):
        repository.persist_observation(
            second_observation,
            entities=(conflicting_entity,),
        )

    with database.connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM observations
            WHERE observation_id = ?
            """,
            (second_observation.observation_id,),
        ).fetchone()[0]

    assert count == 0


def test_quarantine_preserves_malformed_raw_input(tmp_path):
    repository, database, source = make_repository(tmp_path)
    raw_payload = '{"broken": '

    quarantined = QuarantinedObservation(
        quarantine_id=new_id(),
        source_id=source.source_id,
        received_at=NOW,
        raw_payload=raw_payload,
        payload_sha256=payload_checksum(raw_payload),
        rejection_code="INVALID_JSON",
        rejection_reason="Input is not valid JSON",
    )

    repository.persist_quarantine(quarantined)

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT raw_payload, payload_sha256, rejection_code,
                   rejection_reason
            FROM quarantined_observations
            WHERE quarantine_id = ?
            """,
            (quarantined.quarantine_id,),
        ).fetchone()

    assert row["raw_payload"] == raw_payload
    assert row["payload_sha256"] == payload_checksum(raw_payload)
    assert row["rejection_code"] == "INVALID_JSON"
    assert row["rejection_reason"] == "Input is not valid JSON"


def test_get_entity_by_identity_returns_contract(tmp_path):
    repository, _, source = make_repository(tmp_path)
    observation = make_observation(source)
    entity = make_entity()

    repository.persist_observation(
        observation,
        entities=(entity,),
    )

    result = repository.get_entity_by_identity(
        entity.entity_type,
        entity.canonical_identifier,
    )

    assert result == entity


def test_get_entity_by_identity_returns_none_when_missing(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    result = repository.get_entity_by_identity(
        "ip_address",
        "192.0.2.99",
    )

    assert result is None


def test_event_must_reference_current_observation(tmp_path):
    repository, database, source = make_repository(tmp_path)

    observation = make_observation(source)
    unrelated_observation = make_observation(source)
    event = make_event(unrelated_observation)

    with pytest.raises(ValueError, match="persisted observation"):
        repository.persist_observation(
            observation,
            events=(event,),
        )

    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

    assert count == 0


def test_relationship_must_reference_current_observation(tmp_path):
    repository, database, source = make_repository(tmp_path)

    observation = make_observation(source)
    unrelated_observation = make_observation(source)
    source_entity = make_entity()
    target_entity = make_entity("hostname", "example.test")

    relationship = Relationship(
        relationship_id=new_id(),
        source_entity_id=source_entity.entity_id,
        target_entity_id=target_entity.entity_id,
        relationship_type="communicated_with",
        observation_id=unrelated_observation.observation_id,
        observed_at=NOW,
    )

    with pytest.raises(ValueError, match="persisted observation"):
        repository.persist_observation(
            observation,
            entities=(source_entity, target_entity),
            relationships=(relationship,),
        )

    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

    assert count == 0


def test_duplicate_entity_id_in_submission_is_rejected(tmp_path):
    repository, database, source = make_repository(tmp_path)

    observation = make_observation(source)
    entity = make_entity()

    with pytest.raises(ValueError, match="duplicate entity IDs"):
        repository.persist_observation(
            observation,
            entities=(entity, entity),
        )

    with database.connection() as connection:
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

        entity_count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

    assert observation_count == 0
    assert entity_count == 0


def test_missing_relationship_endpoint_rolls_back_entire_observation(
    tmp_path,
):
    repository, database, source = make_repository(tmp_path)

    observation = make_observation(source)
    source_entity = make_entity()
    missing_entity_id = new_id()

    relationship = Relationship(
        relationship_id=new_id(),
        source_entity_id=source_entity.entity_id,
        target_entity_id=missing_entity_id,
        relationship_type="communicated_with",
        observation_id=observation.observation_id,
        observed_at=NOW,
    )

    with pytest.raises(sqlite3.IntegrityError, match="does not exist"):
        repository.persist_observation(
            observation,
            entities=(source_entity,),
            relationships=(relationship,),
        )

    with database.connection() as connection:
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

        entity_count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        relationship_count = connection.execute(
            "SELECT COUNT(*) FROM relationships"
        ).fetchone()[0]

    assert observation_count == 0
    assert entity_count == 0
    assert relationship_count == 0


def test_quarantine_with_unknown_source_is_rejected(tmp_path):
    repository, database, source = make_repository(tmp_path)

    raw_payload = '{"invalid": '
    quarantined = QuarantinedObservation(
        quarantine_id=new_id(),
        source_id=new_id(),
        received_at=NOW,
        raw_payload=raw_payload,
        payload_sha256=payload_checksum(raw_payload),
        rejection_code="INVALID_JSON",
        rejection_reason="Input is not valid JSON",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.persist_quarantine(quarantined)

    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM quarantined_observations"
        ).fetchone()[0]

    assert count == 0
