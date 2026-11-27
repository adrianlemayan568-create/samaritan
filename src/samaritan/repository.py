"""Persistence operations for Samaritan's intelligence records."""

from collections.abc import Iterable, Sequence
import sqlite3

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import (
    Entity,
    Event,
    EventParticipant,
    Observation,
    QuarantinedObservation,
    Relationship,
    Source,
    parse_timestamp,
    validate_id,
    validate_text,
)
from samaritan.database import Database, DatabaseIntegrityError
from samaritan.evaluation_contracts import EvidenceSet


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

    @staticmethod
    def _validate_pagination(limit: int, offset: int) -> None:
        """Validate bounded pagination parameters."""
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")

        if type(offset) is not int:
            raise TypeError("offset must be an integer")
        if offset < 0:
            raise ValueError("offset must not be negative")

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> Observation:
        """Reconstruct and validate an observation contract."""
        return Observation(
            observation_id=row["observation_id"],
            source_id=row["source_id"],
            received_at=parse_timestamp(
                row["received_at"], "received_at"
            ),
            raw_payload=row["raw_payload"],
            payload_sha256=row["payload_sha256"],
            schema_version=row["schema_version"],
        )

    @staticmethod
    def _load_event_participants(
        connection: sqlite3.Connection,
        event_ids: Sequence[str],
    ) -> dict[str, tuple[EventParticipant, ...]]:
        """Load participants for multiple events in one database query."""
        if not event_ids:
            return {}

        placeholders = ", ".join("?" for _ in event_ids)

        rows = connection.execute(
            f"""
            SELECT event_id, entity_id, role
            FROM event_participants
            WHERE event_id IN ({placeholders})
            ORDER BY event_id, participant_index
            """,
            tuple(event_ids),
        ).fetchall()

        grouped: dict[str, list[EventParticipant]] = {
            event_id: [] for event_id in event_ids
        }

        for row in rows:
            grouped[row["event_id"]].append(
                EventParticipant(
                    entity_id=row["entity_id"],
                    role=row["role"],
                )
            )

        return {
            event_id: tuple(participants)
            for event_id, participants in grouped.items()
        }

    @staticmethod
    def _event_from_row(
        row: sqlite3.Row,
        participants: tuple[EventParticipant, ...],
    ) -> Event:
        """Reconstruct and validate an event contract."""
        return Event(
            event_id=row["event_id"],
            observation_id=row["observation_id"],
            event_type=row["event_type"],
            occurred_at=parse_timestamp(
                row["occurred_at"], "occurred_at"
            ),
            original_timestamp=row["original_timestamp"],
            normalized_data_json=row["normalized_data_json"],
            participants=participants,
            schema_version=row["schema_version"],
        )

    @staticmethod
    def _evidence_set_from_rows(
        parent: sqlite3.Row,
        items: Sequence[sqlite3.Row],
    ) -> EvidenceSet:
        """Reconstruct an EvidenceSet while verifying its sealed ordered items."""
        if parent["is_sealed"] != 1:
            raise DatabaseIntegrityError(
                "persisted evidence set is not sealed"
            )
        indexes = tuple(row["item_index"] for row in items)
        if indexes != tuple(range(len(items))):
            raise DatabaseIntegrityError(
                "persisted evidence set item ordering is incomplete or invalid"
            )

        references = []
        for row in items:
            observation_id = row["observation_id"]
            event_id = row["event_id"]
            if (observation_id is None) == (event_id is None):
                raise DatabaseIntegrityError(
                    "persisted evidence set item must reference exactly one record"
                )
            references.append(
                EvidenceReference(
                    observation_id=observation_id,
                    event_id=event_id,
                )
            )

        return EvidenceSet(
            evidence_set_id=parent["evidence_set_id"],
            evidence=tuple(references),
            created_at=parse_timestamp(parent["created_at"], "created_at"),
            schema_version=parent["schema_version"],
        )

    def persist_evidence_set(self, evidence_set: EvidenceSet) -> None:
        """Persist one immutable evidence selection; identical repeats are idempotent."""
        if type(evidence_set) is not EvidenceSet:
            raise TypeError("evidence_set must be exactly an EvidenceSet")

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT evidence_set_id, created_at, schema_version, is_sealed
                FROM evidence_sets
                WHERE evidence_set_id = ?
                """,
                (evidence_set.evidence_set_id,),
            ).fetchone()

            if existing is not None:
                rows = connection.execute(
                    """
                    SELECT item_index, observation_id, event_id
                    FROM evidence_set_items
                    WHERE evidence_set_id = ?
                    ORDER BY item_index
                    """,
                    (evidence_set.evidence_set_id,),
                ).fetchall()
                stored = self._evidence_set_from_rows(existing, rows)
                if stored != evidence_set:
                    raise DatabaseIntegrityError(
                        "evidence_set_id already identifies a different immutable selection"
                    )
                return

            connection.execute(
                """
                INSERT INTO evidence_sets (
                    evidence_set_id, created_at, schema_version, is_sealed
                )
                VALUES (?, ?, ?, 0)
                """,
                (
                    evidence_set.evidence_set_id,
                    evidence_set.created_at.isoformat(),
                    evidence_set.schema_version,
                ),
            )

            for index, reference in enumerate(evidence_set.evidence):
                if type(reference) is not EvidenceReference:
                    raise TypeError(
                        "evidence set references must be exactly EvidenceReference objects"
                    )
                connection.execute(
                    """
                    INSERT INTO evidence_set_items (
                        evidence_set_id, item_index, observation_id, event_id
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        evidence_set.evidence_set_id,
                        index,
                        reference.observation_id,
                        reference.event_id,
                    ),
                )

            connection.execute(
                """
                UPDATE evidence_sets
                SET is_sealed = 1
                WHERE evidence_set_id = ? AND is_sealed = 0
                """,
                (evidence_set.evidence_set_id,),
            )

    def get_evidence_set(self, evidence_set_id: str) -> EvidenceSet | None:
        """Retrieve one immutable evidence selection with exact reference ordering."""
        evidence_set_id = validate_id(evidence_set_id, "evidence_set_id")
        with self.database.connection() as connection:
            parent = connection.execute(
                """
                SELECT evidence_set_id, created_at, schema_version, is_sealed
                FROM evidence_sets
                WHERE evidence_set_id = ?
                """,
                (evidence_set_id,),
            ).fetchone()
            if parent is None:
                return None

            items = connection.execute(
                """
                SELECT item_index, observation_id, event_id
                FROM evidence_set_items
                WHERE evidence_set_id = ?
                ORDER BY item_index
                """,
                (evidence_set_id,),
            ).fetchall()

        return self._evidence_set_from_rows(parent, items)

    def get_observation(
        self,
        observation_id: str,
    ) -> Observation | None:
        """Retrieve one accepted observation by ID."""
        observation_id = validate_id(observation_id, "observation_id")

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    observation_id,
                    source_id,
                    received_at,
                    raw_payload,
                    payload_sha256,
                    schema_version
                FROM observations
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()

        if row is None:
            return None

        return self._observation_from_row(row)

    def list_observations(
        self,
        source_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Observation, ...]:
        """List observations in stable receipt-time order."""
        self._validate_pagination(limit, offset)

        if source_id is not None:
            source_id = validate_id(source_id, "source_id")

        query = """
            SELECT
                observation_id,
                source_id,
                received_at,
                raw_payload,
                payload_sha256,
                schema_version
            FROM observations
        """
        parameters: list[object] = []

        if source_id is not None:
            query += " WHERE source_id = ?"
            parameters.append(source_id)

        query += """
            ORDER BY received_at, observation_id
            LIMIT ? OFFSET ?
        """
        parameters.extend((limit, offset))

        with self.database.connection() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(self._observation_from_row(row) for row in rows)

    def get_event(self, event_id: str) -> Event | None:
        """Retrieve one event together with its ordered participants."""
        event_id = validate_id(event_id, "event_id")

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    event_id,
                    observation_id,
                    event_type,
                    occurred_at,
                    original_timestamp,
                    normalized_data_json,
                    schema_version
                FROM events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()

            if row is None:
                return None

            participants = self._load_event_participants(
                connection,
                (event_id,),
            )

        return self._event_from_row(
            row,
            participants[event_id],
        )

    def list_events(
        self,
        observation_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Event, ...]:
        """List events chronologically with their ordered participants."""
        self._validate_pagination(limit, offset)

        if observation_id is not None:
            observation_id = validate_id(
                observation_id,
                "observation_id",
            )

        query = """
            SELECT
                event_id,
                observation_id,
                event_type,
                occurred_at,
                original_timestamp,
                normalized_data_json,
                schema_version
            FROM events
        """
        parameters: list[object] = []

        if observation_id is not None:
            query += " WHERE observation_id = ?"
            parameters.append(observation_id)

        query += """
            ORDER BY occurred_at, event_id
            LIMIT ? OFFSET ?
        """
        parameters.extend((limit, offset))

        with self.database.connection() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

            event_ids = tuple(row["event_id"] for row in rows)
            participants = self._load_event_participants(
                connection,
                event_ids,
            )

        return tuple(
            self._event_from_row(row, participants[row["event_id"]])
            for row in rows
        )


    def list_relationships(
        self,
        observation_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Relationship, ...]:
        """List explicit relationships in stable observation-time order."""
        self._validate_pagination(limit, offset)
        if observation_id is not None:
            observation_id = validate_id(observation_id, "observation_id")

        query = """
            SELECT
                relationship_id,
                source_entity_id,
                target_entity_id,
                relationship_type,
                observation_id,
                observed_at,
                attributes_json,
                schema_version
            FROM relationships
        """
        parameters: list[object] = []
        if observation_id is not None:
            query += " WHERE observation_id = ?"
            parameters.append(observation_id)
        query += " ORDER BY observed_at, relationship_id LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))

        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return tuple(
            Relationship(
                relationship_id=row["relationship_id"],
                source_entity_id=row["source_entity_id"],
                target_entity_id=row["target_entity_id"],
                relationship_type=row["relationship_type"],
                observation_id=row["observation_id"],
                observed_at=parse_timestamp(row["observed_at"], "observed_at"),
                attributes_json=row["attributes_json"],
                schema_version=row["schema_version"],
            )
            for row in rows
        )

    def get_entities_by_ids(
        self,
        entity_ids: Iterable[str],
    ) -> tuple[Entity, ...]:
        """Retrieve entities by ID, preserving requested order.

        Duplicate requested IDs are queried only once. Missing IDs
        are omitted from the returned tuple.
        """
        if isinstance(entity_ids, (str, bytes)):
            raise TypeError("entity_ids must be an iterable of IDs")

        try:
            supplied_ids = tuple(entity_ids)
        except TypeError as exc:
            raise TypeError("entity_ids must be iterable") from exc

        requested_ids = tuple(
            validate_id(item, "entity_id")
            for item in supplied_ids
        )

        unique_ids = tuple(dict.fromkeys(requested_ids))

        if not unique_ids:
            return ()

        entities_by_id: dict[str, Entity] = {}

        with self.database.connection() as connection:
            # Stay below SQLite's commonly configured variable limit.
            for start in range(0, len(unique_ids), 900):
                batch = unique_ids[start:start + 900]
                placeholders = ", ".join("?" for _ in batch)

                rows = connection.execute(
                    f"""
                    SELECT
                        entity_id,
                        entity_type,
                        canonical_identifier,
                        created_at,
                        schema_version
                    FROM entities
                    WHERE entity_id IN ({placeholders})
                    """,
                    batch,
                ).fetchall()

                for row in rows:
                    entity = Entity(
                        entity_id=row["entity_id"],
                        entity_type=row["entity_type"],
                        canonical_identifier=row["canonical_identifier"],
                        created_at=parse_timestamp(
                            row["created_at"],
                            "created_at",
                        ),
                        schema_version=row["schema_version"],
                    )
                    entities_by_id[entity.entity_id] = entity

        return tuple(
            entities_by_id[entity_id]
            for entity_id in unique_ids
            if entity_id in entities_by_id
        )

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
