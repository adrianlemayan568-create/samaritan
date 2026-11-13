import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from samaritan.contracts import (
    SCHEMA_VERSION,
    Entity,
    Event,
    EventParticipant,
    Observation,
    QuarantinedObservation,
    Relationship,
    Source,
    new_id,
    normalize_timestamp,
    parse_timestamp,
    payload_checksum,
    validate_json_mapping,
    validate_json_object,
)


UTC = timezone.utc
NOW = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)


def make_source(**overrides):
    values = {
        "source_id": new_id(),
        "source_type": "synthetic_lab",
        "name": "Synthetic authentication logs",
        "created_at": NOW,
    }
    values.update(overrides)
    return Source(**values)


def make_observation(**overrides):
    raw_payload = '{"event": "login", "result": "success"}'
    values = {
        "observation_id": new_id(),
        "source_id": new_id(),
        "received_at": NOW,
        "raw_payload": raw_payload,
        "payload_sha256": payload_checksum(raw_payload),
    }
    values.update(overrides)
    return Observation(**values)


def make_quarantine(**overrides):
    raw_payload = '{"event": "login",'
    values = {
        "quarantine_id": new_id(),
        "source_id": new_id(),
        "received_at": NOW,
        "raw_payload": raw_payload,
        "payload_sha256": payload_checksum(raw_payload),
        "rejection_code": "INVALID_JSON",
        "rejection_reason": "Input is incomplete JSON",
    }
    values.update(overrides)
    return QuarantinedObservation(**values)


def make_entity(**overrides):
    values = {
        "entity_id": new_id(),
        "entity_type": "user",
        "canonical_identifier": "alice@example.invalid",
        "created_at": NOW,
    }
    values.update(overrides)
    return Entity(**values)


def make_event(**overrides):
    timestamp = "2026-11-13T10:00:00Z"
    values = {
        "event_id": new_id(),
        "observation_id": new_id(),
        "event_type": "authentication",
        "occurred_at": NOW,
        "original_timestamp": timestamp,
        "normalized_data_json": '{"result":"success"}',
        "participants": (EventParticipant(new_id(), "subject"),),
    }
    values.update(overrides)
    return Event(**values)


def make_relationship(**overrides):
    values = {
        "relationship_id": new_id(),
        "source_entity_id": new_id(),
        "target_entity_id": new_id(),
        "relationship_type": "authenticated_from",
        "observation_id": new_id(),
        "observed_at": NOW,
    }
    values.update(overrides)
    return Relationship(**values)


# Identity and timestamp helpers

def test_new_id_generates_valid_unique_uuids():
    first = new_id()
    second = new_id()

    assert first != second
    assert len(first) == 36


def test_normalize_timestamp_converts_to_utc():
    local_time = datetime(
        2026, 11, 13, 13, 0,
        tzinfo=timezone(timedelta(hours=3)),
    )

    result = normalize_timestamp(local_time, "timestamp")

    assert result == NOW
    assert result.utcoffset() == timedelta(0)


def test_normalize_timestamp_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone"):
        normalize_timestamp(datetime(2026, 11, 13, 10, 0), "timestamp")


def test_parse_timestamp_converts_to_utc():
    result = parse_timestamp("2026-11-13T13:00:00+03:00", "timestamp")

    assert result == NOW


# Source contract

def test_source_accepts_valid_data():
    source = make_source()

    assert source.source_type == "synthetic_lab"
    assert source.created_at == NOW
    assert source.schema_version == SCHEMA_VERSION


def test_source_rejects_invalid_uuid():
    with pytest.raises(ValueError, match="valid UUID"):
        make_source(source_id="not-a-uuid")


def test_source_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        make_source(name="")


def test_source_rejects_unsupported_schema_version():
    with pytest.raises(ValueError, match="schema version"):
        make_source(schema_version=99)


# Observation contract

def test_observation_accepts_valid_json_and_checksum():
    observation = make_observation()

    assert observation.raw_payload == (
        '{"event": "login", "result": "success"}'
    )
    assert observation.payload_sha256 == payload_checksum(
        observation.raw_payload
    )


def test_observation_preserves_exact_original_payload():
    raw = '{ "event" : "login", "result" : "success" }'
    observation = make_observation(
        raw_payload=raw,
        payload_sha256=payload_checksum(raw),
    )

    assert observation.raw_payload == raw


def test_observation_rejects_malformed_json():
    raw = '{"event": "login",'

    with pytest.raises(ValueError, match="strict JSON"):
        make_observation(
            raw_payload=raw,
            payload_sha256=payload_checksum(raw),
        )


def test_observation_rejects_non_object_json():
    raw = '["login", "success"]'

    with pytest.raises(ValueError, match="JSON object"):
        make_observation(
            raw_payload=raw,
            payload_sha256=payload_checksum(raw),
        )


def test_observation_rejects_checksum_mismatch():
    with pytest.raises(ValueError, match="checksum"):
        make_observation(payload_sha256="0" * 64)


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": NaN}',
        '{"score": Infinity}',
        '{"action": "first", "action": "second"}',
    ],
)
def test_observation_rejects_non_strict_json(raw):
    with pytest.raises(ValueError, match="strict JSON"):
        make_observation(
            raw_payload=raw,
            payload_sha256=payload_checksum(raw),
        )


# Quarantine contract

def test_quarantine_preserves_invalid_raw_input():
    raw = '{"event": "login",'

    quarantine = make_quarantine(
        raw_payload=raw,
        payload_sha256=payload_checksum(raw),
    )

    assert quarantine.raw_payload == raw
    assert quarantine.rejection_code == "INVALID_JSON"
    assert quarantine.payload_sha256 == hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def test_quarantine_rejects_checksum_mismatch():
    with pytest.raises(ValueError, match="checksum"):
        make_quarantine(payload_sha256="0" * 64)


def test_quarantine_rejects_empty_rejection_reason():
    with pytest.raises(ValueError, match="non-empty"):
        make_quarantine(rejection_reason="")


# Entity and participant contracts

def test_entity_identity_is_scoped_by_entity_type():
    user = make_entity(
        entity_type="user",
        canonical_identifier="shared-identifier",
    )
    device = make_entity(
        entity_type="device",
        canonical_identifier="shared-identifier",
    )

    assert user.identity_key == ("user", "shared-identifier")
    assert device.identity_key == ("device", "shared-identifier")
    assert user.identity_key != device.identity_key


def test_entity_rejects_invalid_uuid():
    with pytest.raises(ValueError, match="valid UUID"):
        make_entity(entity_id="invalid")


def test_event_participant_requires_valid_entity_id():
    with pytest.raises(ValueError, match="valid UUID"):
        EventParticipant(entity_id="invalid", role="subject")


def test_event_participant_rejects_empty_role():
    with pytest.raises(ValueError, match="non-empty"):
        EventParticipant(entity_id=new_id(), role="")


# Event contract

def test_event_accepts_valid_data():
    participant = EventParticipant(new_id(), "subject")
    event = make_event(participants=(participant,))

    assert event.event_type == "authentication"
    assert event.participants == (participant,)
    assert event.occurred_at == NOW


def test_event_rejects_timestamp_mismatch():
    different_time = NOW + timedelta(minutes=5)

    with pytest.raises(ValueError, match="different instants"):
        make_event(occurred_at=different_time)


def test_event_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        make_event(occurred_at=datetime(2026, 11, 13, 10, 0))


def test_event_rejects_non_tuple_participants():
    with pytest.raises(TypeError, match="tuple"):
        make_event(participants=[])


def test_event_rejects_invalid_participant_objects():
    with pytest.raises(TypeError, match="EventParticipant"):
        make_event(participants=("not a participant",))


# Relationship contract

def test_relationship_accepts_valid_directed_edge():
    relationship = make_relationship()

    assert relationship.source_entity_id != relationship.target_entity_id
    assert relationship.relationship_type == "authenticated_from"


def test_relationship_rejects_self_relationship():
    entity_id = new_id()

    with pytest.raises(ValueError, match="different entities"):
        make_relationship(
            source_entity_id=entity_id,
            target_entity_id=entity_id,
        )


def test_relationship_rejects_invalid_observation_reference():
    with pytest.raises(ValueError, match="valid UUID"):
        make_relationship(observation_id="invalid")


def test_relationship_rejects_non_object_attributes():
    with pytest.raises(ValueError, match="JSON object"):
        make_relationship(attributes_json='["not", "an", "object"]')


# Deterministic JSON mapping

def test_json_mapping_serialization_is_deterministic():
    first = validate_json_mapping(
        {"b": 2, "a": 1},
        "attributes",
    )
    second = validate_json_mapping(
        {"a": 1, "b": 2},
        "attributes",
    )

    assert first == second
    assert first == '{"a":1,"b":2}'


def test_json_mapping_rejects_non_json_values():
    with pytest.raises(ValueError, match="non-JSON"):
        validate_json_mapping({"score": float("nan")}, "attributes")
