"""Core information contracts for Samaritan.

These contracts define the shape and integrity rules of information
before persistence, ingestion, correlation, or analytical processing.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
import unicodedata
from typing import Any


SCHEMA_VERSION = 1


def new_id() -> str:
    """Generate a stable-format internal UUID."""
    return str(uuid.uuid4())


def validate_id(value: str, field_name: str) -> str:
    """Require a valid UUID and return its canonical representation."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a UUID string")

    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID") from exc


def validate_text(value: str, field_name: str) -> str:
    """Require non-empty text without surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    if not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )

    return value


def validate_identifier(value: str, field_name: str) -> str:
    """Validate a bounded registry identifier without control or spoofing characters."""
    value = validate_text(value, field_name)
    if len(value) > 128 or not value[0].isalpha():
        raise ValueError(f"{field_name} must be a 1-128 character identifier")
    for char in value:
        if char.isascii() and (char.isalnum() or char in "_.-"):
            continue
        # Permit precomposed Latin letters (e.g. Café), but reject other
        # scripts that create visually confusable registry keys.
        if char.isalpha() and "LATIN" in unicodedata.name(char, ""):
            continue
        raise ValueError(
            f"{field_name} contains a disallowed or visually ambiguous character"
        )
    return value


def normalize_timestamp(value: datetime, field_name: str) -> datetime:
    """Require an aware datetime and normalize it to UTC."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")

    return value.astimezone(timezone.utc)


def parse_timestamp(value: str, field_name: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO-8601 string")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO-8601 timestamp") from exc

    return normalize_timestamp(parsed, field_name)


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys instead of silently keeping the last value."""
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value

    return result


def validate_json_object(value: str, field_name: str) -> str:
    """Require strict JSON object syntax and preserve the exact input text."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a JSON string")

    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain strict JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must contain a JSON object")

    return value


def validate_json_mapping(value: dict[str, Any], field_name: str) -> str:
    """Validate a mapping and serialize it as deterministic JSON."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary")

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains non-JSON data") from exc


def payload_checksum(raw_payload: str) -> str:
    """Calculate SHA-256 over the exact UTF-8 payload text."""
    if not isinstance(raw_payload, str):
        raise TypeError("raw_payload must be a string")

    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


def validate_checksum(raw_payload: str, checksum: str) -> str:
    """Ensure a supplied checksum matches the original payload."""
    expected = payload_checksum(raw_payload)

    if checksum != expected:
        raise ValueError("payload checksum does not match raw payload")

    return expected


@dataclass(frozen=True, slots=True)
class Source:
    """A registered origin of information."""

    source_id: str
    source_type: str
    name: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", validate_id(self.source_id, "source_id")
        )
        object.__setattr__(
            self, "source_type", validate_text(self.source_type, "source_type")
        )
        object.__setattr__(self, "name", validate_text(self.name, "name"))
        object.__setattr__(
            self, "created_at", normalize_timestamp(self.created_at, "created_at")
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported source schema version")


@dataclass(frozen=True, slots=True)
class Observation:
    """An accepted input record with preserved original evidence."""

    observation_id: str
    source_id: str
    received_at: datetime
    raw_payload: str
    payload_sha256: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            validate_id(self.observation_id, "observation_id"),
        )
        object.__setattr__(
            self, "source_id", validate_id(self.source_id, "source_id")
        )
        object.__setattr__(
            self, "received_at", normalize_timestamp(self.received_at, "received_at")
        )
        object.__setattr__(
            self,
            "raw_payload",
            validate_json_object(self.raw_payload, "raw_payload"),
        )
        object.__setattr__(
            self,
            "payload_sha256",
            validate_checksum(self.raw_payload, self.payload_sha256),
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported observation schema version")


@dataclass(frozen=True, slots=True)
class QuarantinedObservation:
    """An invalid input preserved with its rejection details."""

    quarantine_id: str
    source_id: str
    received_at: datetime
    raw_payload: str
    payload_sha256: str
    rejection_code: str
    rejection_reason: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "quarantine_id", validate_id(self.quarantine_id, "quarantine_id")
        )
        object.__setattr__(
            self, "source_id", validate_id(self.source_id, "source_id")
        )
        object.__setattr__(
            self, "received_at", normalize_timestamp(self.received_at, "received_at")
        )

        if not isinstance(self.raw_payload, str):
            raise TypeError("raw_payload must be a string")

        object.__setattr__(
            self,
            "payload_sha256",
            validate_checksum(self.raw_payload, self.payload_sha256),
        )
        object.__setattr__(
            self,
            "rejection_code",
            validate_text(self.rejection_code, "rejection_code"),
        )
        object.__setattr__(
            self,
            "rejection_reason",
            validate_text(self.rejection_reason, "rejection_reason"),
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported quarantine schema version")


@dataclass(frozen=True, slots=True)
class Entity:
    """A typed identity within a separate entity namespace."""

    entity_id: str
    entity_type: str
    canonical_identifier: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entity_id", validate_id(self.entity_id, "entity_id")
        )
        object.__setattr__(
            self, "entity_type", validate_text(self.entity_type, "entity_type")
        )
        object.__setattr__(
            self,
            "canonical_identifier",
            validate_text(self.canonical_identifier, "canonical_identifier"),
        )
        object.__setattr__(
            self, "created_at", normalize_timestamp(self.created_at, "created_at")
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported entity schema version")

    @property
    def identity_key(self) -> tuple[str, str]:
        """Return the identity key scoped to its entity type."""
        return (self.entity_type, self.canonical_identifier)


@dataclass(frozen=True, slots=True)
class EventParticipant:
    """An entity's explicitly defined role in an event."""

    entity_id: str
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entity_id", validate_id(self.entity_id, "entity_id")
        )
        object.__setattr__(self, "role", validate_text(self.role, "role"))


@dataclass(frozen=True, slots=True)
class Event:
    """A normalized occurrence derived from an observation."""

    event_id: str
    observation_id: str
    event_type: str
    occurred_at: datetime
    original_timestamp: str
    normalized_data_json: str
    participants: tuple[EventParticipant, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_id", validate_id(self.event_id, "event_id")
        )
        object.__setattr__(
            self,
            "observation_id",
            validate_id(self.observation_id, "observation_id"),
        )
        object.__setattr__(
            self, "event_type", validate_text(self.event_type, "event_type")
        )
        object.__setattr__(
            self, "occurred_at", normalize_timestamp(self.occurred_at, "occurred_at")
        )
        object.__setattr__(
            self,
            "original_timestamp",
            validate_text(self.original_timestamp, "original_timestamp"),
        )

        original_utc = parse_timestamp(
            self.original_timestamp, "original_timestamp"
        )

        if original_utc != self.occurred_at:
            raise ValueError(
                "original_timestamp and occurred_at represent different instants"
            )

        object.__setattr__(
            self,
            "normalized_data_json",
            validate_json_object(
                self.normalized_data_json, "normalized_data_json"
            ),
        )

        if not isinstance(self.participants, tuple):
            raise TypeError("participants must be a tuple")

        if not all(isinstance(item, EventParticipant) for item in self.participants):
            raise TypeError("participants must contain EventParticipant objects")

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported event schema version")


@dataclass(frozen=True, slots=True)
class Relationship:
    """An evidence-backed, directed relationship between two entities."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    observation_id: str
    observed_at: datetime
    attributes_json: str = "{}"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "relationship_id",
            "source_entity_id",
            "target_entity_id",
            "observation_id",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_id(getattr(self, field_name), field_name),
            )

        if self.source_entity_id == self.target_entity_id:
            raise ValueError("relationship endpoints must be different entities")

        object.__setattr__(
            self,
            "relationship_type",
            validate_text(self.relationship_type, "relationship_type"),
        )
        object.__setattr__(
            self, "observed_at", normalize_timestamp(self.observed_at, "observed_at")
        )
        object.__setattr__(
            self,
            "attributes_json",
            validate_json_object(self.attributes_json, "attributes_json"),
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported relationship schema version")
