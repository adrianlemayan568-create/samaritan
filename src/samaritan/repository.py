"""Persistence operations for Samaritan's intelligence records."""

from collections.abc import Iterable, Sequence
import sqlite3

from samaritan.contracts import (
    Entity,
    Event,
    Observation,
    QuarantinedObservation,
    Relationship,
    Source,
    parse_timestamp,
    validate_text,
)
from samaritan.database import Database, DatabaseIntegrityError


def _require_records(
    records: Iterable[object],
    expected_type: type,
    field_name: str,
) -> tuple:
    """Validate and freeze a collection of contract records."""
    if isinstance(records, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of records")

    try:
        result = tuple(records)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable") from exc

    if not all(isinstance(record, expected_type) for record in result):
        raise TypeError(
            f"{field_name} must contain only {expected_type.__name__} objects"
        )

    return result


class Repository:
    """Persist and retrieve validated Samaritan contracts."""

    def __init__(self, database: Database) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database instance")

        self.database = database

    def register_source(self, source: Source) -> None:
        """Register a source. Existing source IDs are never overwritten."""
        if not isinstance(source, Source):
            raise TypeError("source must be a Source object")

        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    source_id,
                    source_type,
                    name,
                    created_at,
                    schema_version
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    source.source_type,
                    source.name,
                    source.created_at.isoformat(),
                    source.schema_version,
                ),
            )

    def persist_quarantine(
        self,
        quarantined: QuarantinedObservation,
    ) -> None:
        """Persist rejected raw input without altering its contents."""
        if not isinstance(quarantined, QuarantinedObservation):
            raise TypeError(
                "quarantined must be a QuarantinedObservation object"
            )

        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO quarantined_observations (
                    quarantine_id,
                    source_id,
                    received_at,
                    raw_payload,
                    payload_sha256,
                    rejection_code,
                    rejection_reason,
                    schema_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quarantined.quarantine_id,
                    quarantined.source_id,
                    quarantined.received_at.isoformat(),
                    quarantined.raw_payload,
                    quarantined.payload_sha256,
                    quarantined.rejection_code,
                    quarantined.rejection_reason,
                    quarantined.schema_version,
                ),
            )

    def persist_observation(
        self,
        observation: Observation,
        entities: Sequence[Entity] = (),
        events: Sequence[Event] = (),
        relationships: Sequence[Relationship] = (),
    ) -> dict[str, str]:
        """Persist an observation and all derived records atomically.

        Returns a mapping from each submitted entity ID to the entity ID
        actually stored. Existing entities are reused by typed identity.
        """
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation object")

        entities = _require_records(entities, Entity, "entities")
        events = _require_records(events, Event, "events")
        relationships = _require_records(
            relationships, Relationship, "relationships"
        )

        for event in events:
            if event.observation_id != observation.observation_id:
                raise ValueError(
                    "every event must reference the persisted observation"
                )

        for relationship in relationships:
            if relationship.observation_id != observation.observation_id:
                raise ValueError(
                    "every relationship must reference the persisted observation"
                )

        entity_ids = {}
        for entity in entities:
            if entity.entity_id in entity_ids:
                raise ValueError(
                    "duplicate entity IDs are not allowed in one submission"
                )
            entity_ids[entity.entity_id] = entity.identity_key

        resolved_ids: dict[str, str] = {}

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            connection.execute(
                """
                INSERT INTO observations (
                    observation_id,
                    source_id,
                    received_at,
                    raw_payload,
                    payload_sha256,
                    schema_version
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.source_id,
                    observation.received_at.isoformat(),
                    observation.raw_payload,
                    observation.payload_sha256,
                    observation.schema_version,
                ),
            )

            # Resolve or create entities before inserting dependent records.
            for entity in entities:
                existing_by_id = connection.execute(
                    """
                    SELECT entity_type, canonical_identifier
                    FROM entities
                    WHERE entity_id = ?
                    """,
                    (entity.entity_id,),
                ).fetchone()

                if existing_by_id is not None:
                    existing_identity = (
                        existing_by_id["entity_type"],
                        existing_by_id["canonical_identifier"],
                    )
                    if existing_identity != entity.identity_key:
                        raise DatabaseIntegrityError(
                            "submitted entity ID already belongs to "
                            "a different entity identity"
                        )

                existing_by_identity = connection.execute(
                    """
                    SELECT entity_id
                    FROM entities
                    WHERE entity_type = ?
                      AND canonical_identifier = ?
                    """,
                    entity.identity_key,
                ).fetchone()

                if existing_by_identity is not None:
                    resolved_ids[entity.entity_id] = (
                        existing_by_identity["entity_id"]
                    )
                    continue

                connection.execute(
                    """
                    INSERT INTO entities (
                        entity_id,
                        entity_type,
                        canonical_identifier,
                        created_at,
                        schema_version
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entity.entity_id,
                        entity.entity_type,
                        entity.canonical_identifier,
                        entity.created_at.isoformat(),
                        entity.schema_version,
                    ),
                )
                resolved_ids[entity.entity_id] = entity.entity_id

            # Events and participants share the observation transaction.
            for event in events:
                connection.execute(
                    """
                    INSERT INTO events (
                        event_id,
                        observation_id,
                        event_type,
                        occurred_at,
                        original_timestamp,
                        normalized_data_json,
                        schema_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.observation_id,
                        event.event_type,
                        event.occurred_at.isoformat(),
                        event.original_timestamp,
                        event.normalized_data_json,
                        event.schema_version,
                    ),
                )

                for index, participant in enumerate(event.participants):
                    entity_id = self._resolve_reference(
                        connection,
                        participant.entity_id,
                        resolved_ids,
                    )

                    connection.execute(
                        """
                        INSERT INTO event_participants (
                            event_id,
                            participant_index,
                            entity_id,
                            role
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            index,
                            entity_id,
                            participant.role,
                        ),
                    )

            # Relationships also retain their observation provenance.
            for relationship in relationships:
                source_entity_id = self._resolve_reference(
                    connection,
                    relationship.source_entity_id,
                    resolved_ids,
                )
                target_entity_id = self._resolve_reference(
                    connection,
                    relationship.target_entity_id,
                    resolved_ids,
                )

                connection.execute(
                    """
                    INSERT INTO relationships (
                        relationship_id,
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        observation_id,
                        observed_at,
                        attributes_json,
                        schema_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relationship.relationship_id,
                        source_entity_id,
                        target_entity_id,
                        relationship.relationship_type,
                        relationship.observation_id,
                        relationship.observed_at.isoformat(),
                        relationship.attributes_json,
                        relationship.schema_version,
                    ),
                )

        return resolved_ids

    @staticmethod
    def _resolve_reference(
        connection: sqlite3.Connection,
        submitted_entity_id: str,
        resolved_ids: dict[str, str],
    ) -> str:
        """Resolve a submitted entity ID or verify an existing database ID."""
        resolved = resolved_ids.get(submitted_entity_id)
        if resolved is not None:
            return resolved

        existing = connection.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE entity_id = ?
            """,
            (submitted_entity_id,),
        ).fetchone()

        if existing is None:
            raise sqlite3.IntegrityError(
                f"referenced entity does not exist: {submitted_entity_id}"
            )

        return existing["entity_id"]

    def get_entity_by_identity(
        self,
        entity_type: str,
        canonical_identifier: str,
    ) -> Entity | None:
        """Retrieve an entity by its type-scoped canonical identity."""
        entity_type = validate_text(entity_type, "entity_type")
        canonical_identifier = validate_text(
            canonical_identifier,
            "canonical_identifier",
        )

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    entity_id,
                    entity_type,
                    canonical_identifier,
                    created_at,
                    schema_version
                FROM entities
                WHERE entity_type = ?
                  AND canonical_identifier = ?
                """,
                (entity_type, canonical_identifier),
            ).fetchone()

        if row is None:
            return None

        return Entity(
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            canonical_identifier=row["canonical_identifier"],
            created_at=parse_timestamp(row["created_at"], "created_at"),
            schema_version=row["schema_version"],
        )
