"""Tests for Samaritan's persistence repository."""

from datetime import datetime, timezone
import sqlite3

import pytest

from samaritan.analysis_contracts import EvidenceReference
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
from samaritan.evaluation_contracts import EvidenceSet
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


def test_get_observation_preserves_original_evidence(tmp_path):
    repository, _, source = make_repository(tmp_path)
    raw_payload = '{ "evidence": true, "sequence": 7 }\n'
    observation = make_observation(source, raw_payload)

    repository.persist_observation(observation)

    retrieved = repository.get_observation(observation.observation_id)

    assert retrieved == observation
    assert retrieved.raw_payload == raw_payload
    assert retrieved.payload_sha256 == payload_checksum(raw_payload)


def test_get_observation_returns_none_when_missing(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    assert repository.get_observation(new_id()) is None


def test_list_observations_orders_filters_and_paginates(tmp_path):
    repository, _, source = make_repository(tmp_path)

    first = make_observation(source, '{"sequence":1}')
    second = make_observation(source, '{"sequence":2}')
    third = make_observation(source, '{"sequence":3}')

    for observation in (first, second, third):
        repository.persist_observation(observation)

    expected = tuple(
        sorted(
            (first, second, third),
            key=lambda item: (item.received_at, item.observation_id),
        )
    )

    observations = repository.list_observations(
        source_id=source.source_id,
        limit=2,
    )

    assert observations == expected[:2]

    next_page = repository.list_observations(
        source_id=source.source_id,
        limit=2,
        offset=2,
    )

    assert next_page == expected[2:]



def test_list_observations_uses_stable_id_order_for_equal_timestamps(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)

    first = make_observation(source, '{"sequence":1}')
    second = make_observation(source, '{"sequence":2}')

    ordered = sorted(
        (first, second),
        key=lambda item: item.observation_id,
    )

    for observation in (first, second):
        repository.persist_observation(observation)

    retrieved = repository.list_observations(
        source_id=source.source_id,
    )

    assert retrieved == tuple(ordered)


def test_get_event_reconstructs_ordered_participants(tmp_path):
    repository, _, source = make_repository(tmp_path)

    observation = make_observation(source)
    first_entity = make_entity("ip_address", "192.0.2.10")
    second_entity = make_entity("username", "synthetic-user")

    event = Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type="authentication_failure",
        occurred_at=NOW,
        original_timestamp="2026-11-16T10:00:00Z",
        normalized_data_json='{"result":"failure"}',
        participants=(
            EventParticipant(first_entity.entity_id, "source"),
            EventParticipant(second_entity.entity_id, "target"),
        ),
    )

    repository.persist_observation(
        observation,
        entities=(first_entity, second_entity),
        events=(event,),
    )

    retrieved = repository.get_event(event.event_id)

    assert retrieved == event
    assert retrieved.participants == (
        EventParticipant(first_entity.entity_id, "source"),
        EventParticipant(second_entity.entity_id, "target"),
    )


def test_get_event_returns_none_when_missing(tmp_path):
    repository, _, _ = make_repository(tmp_path)

    assert repository.get_event(new_id()) is None


def test_list_events_orders_filters_and_paginates(tmp_path):
    repository, _, source = make_repository(tmp_path)

    first_observation = make_observation(source, '{"sequence":1}')
    second_observation = make_observation(source, '{"sequence":2}')

    earlier = Event(
        event_id=new_id(),
        observation_id=first_observation.observation_id,
        event_type="authentication_failure",
        occurred_at=NOW.replace(hour=9),
        original_timestamp="2026-11-16T09:00:00Z",
        normalized_data_json='{"result":"failure"}',
        participants=(),
    )

    later = Event(
        event_id=new_id(),
        observation_id=second_observation.observation_id,
        event_type="authentication_success",
        occurred_at=NOW.replace(hour=11),
        original_timestamp="2026-11-16T11:00:00Z",
        normalized_data_json='{"result":"success"}',
        participants=(),
    )

    repository.persist_observation(
        first_observation,
        events=(earlier,),
    )
    repository.persist_observation(
        second_observation,
        events=(later,),
    )

    assert repository.list_events() == (earlier, later)
    assert repository.list_events(limit=1) == (earlier,)
    assert repository.list_events(limit=1, offset=1) == (later,)
    assert repository.list_events(
        observation_id=second_observation.observation_id,
    ) == (later,)


@pytest.mark.parametrize(
    ("limit", "offset", "exception"),
    [
        (0, 0, ValueError),
        (501, 0, ValueError),
        (-1, 0, ValueError),
        (True, 0, TypeError),
        (1.5, 0, TypeError),
        (1, -1, ValueError),
        (1, True, TypeError),
        (1, 1.5, TypeError),
    ],
)
def test_list_methods_reject_invalid_pagination(
    tmp_path,
    limit,
    offset,
    exception,
):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(exception):
        repository.list_observations(limit=limit, offset=offset)

    with pytest.raises(exception):
        repository.list_events(limit=limit, offset=offset)

def test_list_events_hydrates_participants_in_batch(tmp_path):
    repository, _, source = make_repository(tmp_path)
    observation = make_observation(source)

    first_entity = make_entity()
    second_entity = make_entity("username", "synthetic-user")

    first = make_event(
        observation,
        (first_entity.entity_id, second_entity.entity_id),
    )
    second = make_event(
        observation,
        (second_entity.entity_id, first_entity.entity_id),
    )

    repository.persist_observation(
        observation,
        entities=(first_entity, second_entity),
        events=(first, second),
    )

    retrieved = repository.list_events()

    expected = {
        first.event_id: first.participants,
        second.event_id: second.participants,
    }

    assert len(retrieved) == 2

    for event in retrieved:
        assert event.participants == expected[event.event_id]


def test_evidence_set_round_trip_preserves_order_and_reference_granularity(tmp_path):
    repository, _, source = make_repository(tmp_path)
    observation = make_observation(source)
    entity = make_entity()
    event = make_event(observation, (entity.entity_id,))
    repository.persist_observation(
        observation,
        entities=(entity,),
        events=(event,),
    )
    references = (
        EvidenceReference(observation_id=observation.observation_id),
        EvidenceReference(event_id=event.event_id),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=references,
        created_at=NOW,
    )

    repository.persist_evidence_set(evidence_set)
    retrieved = repository.get_evidence_set(evidence_set.evidence_set_id)

    assert retrieved == evidence_set
    assert retrieved.evidence == references
    assert retrieved.evidence[0].observation_id == observation.observation_id
    assert retrieved.evidence[1].event_id == event.event_id


def test_evidence_set_allows_overlapping_observation_and_event_references(tmp_path):
    repository, _, source = make_repository(tmp_path)
    observation = make_observation(source)
    event = make_event(observation)
    repository.persist_observation(observation, events=(event,))
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(
            EvidenceReference(observation_id=observation.observation_id),
            EvidenceReference(event_id=event.event_id),
        ),
        created_at=NOW,
    )

    repository.persist_evidence_set(evidence_set)

    assert repository.get_evidence_set(evidence_set.evidence_set_id) == evidence_set


def test_empty_evidence_set_can_be_persisted_and_retrieved(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(),
        created_at=NOW,
    )

    repository.persist_evidence_set(evidence_set)

    assert repository.get_evidence_set(evidence_set.evidence_set_id) == evidence_set


def test_persisting_identical_evidence_set_again_is_idempotent(tmp_path):
    repository, _, source = make_repository(tmp_path)
    observation = make_observation(source)
    repository.persist_observation(observation)
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(EvidenceReference(observation_id=observation.observation_id),),
        created_at=NOW,
    )

    repository.persist_evidence_set(evidence_set)
    repository.persist_evidence_set(evidence_set)

    assert repository.get_evidence_set(evidence_set.evidence_set_id) == evidence_set


def test_evidence_set_id_cannot_be_reused_for_a_different_selection(tmp_path):
    repository, _, source = make_repository(tmp_path)
    first = make_observation(source, '{"selection":1}')
    second = make_observation(source, '{"selection":2}')
    repository.persist_observation(first)
    repository.persist_observation(second)
    evidence_set_id = new_id()
    original = EvidenceSet(
        evidence_set_id=evidence_set_id,
        evidence=(EvidenceReference(observation_id=first.observation_id),),
        created_at=NOW,
    )
    conflicting = EvidenceSet(
        evidence_set_id=evidence_set_id,
        evidence=(EvidenceReference(observation_id=second.observation_id),),
        created_at=NOW,
    )
    repository.persist_evidence_set(original)

    with pytest.raises(
        DatabaseIntegrityError,
        match="different immutable selection",
    ):
        repository.persist_evidence_set(conflicting)

    assert repository.get_evidence_set(evidence_set_id) == original


@pytest.mark.parametrize("reference_kind", ["observation", "event"])
def test_evidence_set_rejects_missing_source_reference_without_partial_record(
    tmp_path, reference_kind
):
    repository, _, _ = make_repository(tmp_path)
    reference = (
        EvidenceReference(observation_id=new_id())
        if reference_kind == "observation"
        else EvidenceReference(event_id=new_id())
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=(reference,),
        created_at=NOW,
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.persist_evidence_set(evidence_set)

    assert repository.get_evidence_set(evidence_set.evidence_set_id) is None
