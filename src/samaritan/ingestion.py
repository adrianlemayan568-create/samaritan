"""Validated telemetry ingestion for Samaritan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from samaritan.contracts import (
    SCHEMA_VERSION,
    Entity,
    Event,
    EventParticipant,
    Observation,
    QuarantinedObservation,
    Relationship,
    new_id,
    normalize_timestamp,
    parse_timestamp,
    payload_checksum,
    validate_id,
    validate_json_mapping,
)
from samaritan.repository import Repository


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of one ingestion attempt."""

    accepted: bool
    observation_id: str | None = None
    quarantine_id: str | None = None
    rejection_code: str | None = None
    rejection_reason: str | None = None
    resolved_entity_ids: dict[str, str] | None = None


class PayloadValidationError(ValueError):
    """An input payload violates the ingestion envelope."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value

    return result


def _parse_payload(raw_payload: str) -> dict[str, Any]:
    if not isinstance(raw_payload, str):
        raise TypeError("raw_payload must be a string")

    try:
        value = json.loads(
            raw_payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PayloadValidationError(
            f"payload is not strict JSON: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise PayloadValidationError("payload root must be a JSON object")

    return value


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PayloadValidationError(f"{field} must be an object")
    return value


def _require_array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PayloadValidationError(f"{field} must be an array")
    return value


def _require_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str] | None = None,
    field: str = "record",
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional

    if missing:
        raise PayloadValidationError(
            f"{field} is missing fields: {', '.join(sorted(missing))}"
        )

    if unknown:
        raise PayloadValidationError(
            f"{field} contains unsupported fields: {', '.join(sorted(unknown))}"
        )


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise PayloadValidationError(f"{field} must be a string")
    return value


class IngestionService:
    """Validate, normalize, persist, or quarantine incoming telemetry."""

    def __init__(self, repository: Repository) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("repository must be a Repository instance")
        self.repository = repository

    def ingest(
        self,
        source_id: str,
        raw_payload: str,
        received_at: datetime,
        observation_id: str | None = None,
    ) -> IngestionResult:
        """Ingest one raw JSON envelope.

        Invalid telemetry is quarantined. Persistence and infrastructure
        failures propagate to the caller and are never mislabeled as
        payload validation failures.
        """
        source_id = validate_id(source_id, "source_id")
        received_at = normalize_timestamp(received_at, "received_at")

        if not isinstance(raw_payload, str):
            raise TypeError("raw_payload must be a string")

        if observation_id is None:
            observation_id = new_id()
        else:
            observation_id = validate_id(observation_id, "observation_id")

        try:
            envelope = _parse_payload(raw_payload)
            entities, events, relationships = self._build_records(
                envelope,
                observation_id,
                received_at,
            )

            observation = Observation(
                observation_id=observation_id,
                source_id=source_id,
                received_at=received_at,
                raw_payload=raw_payload,
                payload_sha256=payload_checksum(raw_payload),
            )

        except (PayloadValidationError, TypeError, ValueError) as exc:
            return self._quarantine(
                source_id=source_id,
                raw_payload=raw_payload,
                received_at=received_at,
                rejection_code=self._rejection_code(exc),
                rejection_reason=str(exc),
            )

        # Deliberately outside the validation try-block:
        # database failures must propagate, not become quarantine records.
        resolved_entity_ids = self.repository.persist_observation(
            observation,
            entities=entities,
            events=events,
            relationships=relationships,
        )

        return IngestionResult(
            accepted=True,
            observation_id=observation_id,
            resolved_entity_ids=resolved_entity_ids,
        )

    @staticmethod
    def _rejection_code(exc: Exception) -> str:
        if isinstance(exc, PayloadValidationError):
            return "invalid_payload"
        return "invalid_record"

    def _quarantine(
        self,
        source_id: str,
        raw_payload: str,
        received_at: datetime,
        rejection_code: str,
        rejection_reason: str,
    ) -> IngestionResult:
        quarantined = QuarantinedObservation(
            quarantine_id=new_id(),
            source_id=source_id,
            received_at=received_at,
            raw_payload=raw_payload,
            payload_sha256=payload_checksum(raw_payload),
            rejection_code=rejection_code,
            rejection_reason=rejection_reason or rejection_code,
        )

        self.repository.persist_quarantine(quarantined)

        return IngestionResult(
            accepted=False,
            quarantine_id=quarantined.quarantine_id,
            rejection_code=rejection_code,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def _build_records(
        envelope: dict[str, Any],
        observation_id: str,
        received_at: datetime,
    ) -> tuple[list[Entity], list[Event], list[Relationship]]:
        _require_keys(
            envelope,
            {
                "format_version",
                "entities",
                "events",
                "relationships",
            },
            field="envelope",
        )

        version = envelope["format_version"]
        if type(version) is not int or version != 1:
            raise PayloadValidationError(
                "format_version must be the integer 1"
            )

        entity_records = _require_array(envelope["entities"], "entities")
        event_records = _require_array(envelope["events"], "events")
        relationship_records = _require_array(
            envelope["relationships"], "relationships"
        )

        entities: list[Entity] = []
        events: list[Event] = []
        relationships: list[Relationship] = []
        submitted_event_ids: set[str] = set()
        submitted_relationship_ids: set[str] = set()

        submitted_entity_ids: set[str] = set()

        for index, raw_entity in enumerate(entity_records):
            field = f"entities[{index}]"
            record = _require_object(raw_entity, field)

            _require_keys(
                record,
                {"entity_id", "entity_type", "canonical_identifier"},
                field=field,
            )

            entity_id = _require_string(
                record["entity_id"], f"{field}.entity_id"
            )

            entity = Entity(
                entity_id=entity_id,
                entity_type=_require_string(
                    record["entity_type"], f"{field}.entity_type"
                ),
                canonical_identifier=_require_string(
                    record["canonical_identifier"],
                    f"{field}.canonical_identifier",
                ),
                created_at=received_at,
            )

            if entity.entity_id in submitted_entity_ids:
                raise PayloadValidationError(
                    f"duplicate entity ID in envelope: {entity.entity_id}"
                )

            submitted_entity_ids.add(entity.entity_id)
            entities.append(entity)

        for index, raw_event in enumerate(event_records):
            field = f"events[{index}]"
            record = _require_object(raw_event, field)

            _require_keys(
                record,
                {
                    "event_id",
                    "event_type",
                    "original_timestamp",
                    "normalized_data",
                    "participants",
                },
                field=field,
            )

            original_timestamp = _require_string(
                record["original_timestamp"],
                f"{field}.original_timestamp",
            )

            normalized_data = _require_object(
                record["normalized_data"],
                f"{field}.normalized_data",
            )

            raw_participants = _require_array(
                record["participants"],
                f"{field}.participants",
            )

            participants: list[EventParticipant] = []

            for participant_index, raw_participant in enumerate(
                raw_participants
            ):
                participant_field = (
                    f"{field}.participants[{participant_index}]"
                )
                participant = _require_object(
                    raw_participant,
                    participant_field,
                )

                _require_keys(
                    participant,
                    {"entity_id", "role"},
                    field=participant_field,
                )

                participants.append(
                    EventParticipant(
                        entity_id=_require_string(
                            participant["entity_id"],
                            f"{participant_field}.entity_id",
                        ),
                        role=_require_string(
                            participant["role"],
                            f"{participant_field}.role",
                        ),
                    )
                )

            event = Event(
                event_id=_require_string(
                    record["event_id"], f"{field}.event_id"
                ),
                observation_id=observation_id,
                event_type=_require_string(
                    record["event_type"], f"{field}.event_type"
                ),
                occurred_at=parse_timestamp(
                    original_timestamp,
                    f"{field}.original_timestamp",
                ),
                original_timestamp=original_timestamp,
                normalized_data_json=validate_json_mapping(
                    normalized_data,
                    f"{field}.normalized_data",
                ),
                participants=tuple(participants),
            )

            if event.event_id in submitted_event_ids:
                raise PayloadValidationError(
                    f"duplicate event ID in envelope: {event.event_id}"
                )

            submitted_event_ids.add(event.event_id)
            events.append(event)

        for index, raw_relationship in enumerate(relationship_records):
            field = f"relationships[{index}]"
            record = _require_object(raw_relationship, field)

            _require_keys(
                record,
                {
                    "relationship_id",
                    "source_entity_id",
                    "target_entity_id",
                    "relationship_type",
                },
                optional={"attributes"},
                field=field,
            )

            attributes = _require_object(
                record.get("attributes", {}),
                f"{field}.attributes",
            )

            relationship = Relationship(
                relationship_id=_require_string(
                    record["relationship_id"],
                    f"{field}.relationship_id",
                ),
                source_entity_id=_require_string(
                    record["source_entity_id"],
                    f"{field}.source_entity_id",
                ),
                target_entity_id=_require_string(
                    record["target_entity_id"],
                    f"{field}.target_entity_id",
                ),
                relationship_type=_require_string(
                    record["relationship_type"],
                    f"{field}.relationship_type",
                ),
                observation_id=observation_id,
                observed_at=received_at,
                attributes_json=validate_json_mapping(
                    attributes,
                    f"{field}.attributes",
                ),
            )

            if relationship.relationship_id in submitted_relationship_ids:
                raise PayloadValidationError(
                    "duplicate relationship ID in envelope: "
                    f"{relationship.relationship_id}"
                )

            submitted_relationship_ids.add(relationship.relationship_id)
            relationships.append(relationship)

        return entities, events, relationships
