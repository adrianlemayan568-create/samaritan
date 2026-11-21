"""SQLite database foundation for Samaritan."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


SCHEMA_VERSION = 2

REQUIRED_TABLES = frozenset(
    {
        "sources",
        "observations",
        "quarantined_observations",
        "entities",
        "events",
        "event_participants",
        "relationships",
        "evidence_sets",
        "evidence_set_items",
    }
)


class DatabaseError(RuntimeError):
    """Base exception for Samaritan database failures."""


class UnsupportedSchemaVersionError(DatabaseError):
    """Raised when a database uses an unsupported schema version."""


class DatabaseInitializationError(DatabaseError):
    """Raised when an existing database cannot be safely initialized."""


class DatabaseIntegrityError(DatabaseError):
    """Raised when the database structure or integrity is invalid."""


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE sources (
        source_id TEXT PRIMARY KEY NOT NULL,
        source_type TEXT NOT NULL
            CHECK (length(trim(source_type)) > 0),
        name TEXT NOT NULL
            CHECK (length(trim(name)) > 0),
        created_at TEXT NOT NULL,
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1)
    )
    """,
    """
    CREATE TABLE observations (
        observation_id TEXT PRIMARY KEY NOT NULL,
        source_id TEXT NOT NULL,
        received_at TEXT NOT NULL,
        raw_payload TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL
            CHECK (
                length(payload_sha256) = 64
                AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1),

        FOREIGN KEY (source_id)
            REFERENCES sources(source_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE quarantined_observations (
        quarantine_id TEXT PRIMARY KEY NOT NULL,
        source_id TEXT NOT NULL,
        received_at TEXT NOT NULL,
        raw_payload TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL
            CHECK (
                length(payload_sha256) = 64
                AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
        rejection_code TEXT NOT NULL
            CHECK (length(trim(rejection_code)) > 0),
        rejection_reason TEXT NOT NULL
            CHECK (length(trim(rejection_reason)) > 0),
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1),

        FOREIGN KEY (source_id)
            REFERENCES sources(source_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE entities (
        entity_id TEXT PRIMARY KEY NOT NULL,
        entity_type TEXT NOT NULL
            CHECK (length(trim(entity_type)) > 0),
        canonical_identifier TEXT NOT NULL
            CHECK (length(trim(canonical_identifier)) > 0),
        created_at TEXT NOT NULL,
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1),

        UNIQUE (entity_type, canonical_identifier)
    )
    """,
    """
    CREATE TABLE events (
        event_id TEXT PRIMARY KEY NOT NULL,
        observation_id TEXT NOT NULL,
        event_type TEXT NOT NULL
            CHECK (length(trim(event_type)) > 0),
        occurred_at TEXT NOT NULL,
        original_timestamp TEXT NOT NULL,
        normalized_data_json TEXT NOT NULL,
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1),

        FOREIGN KEY (observation_id)
            REFERENCES observations(observation_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE event_participants (
        event_id TEXT NOT NULL,
        participant_index INTEGER NOT NULL
            CHECK (participant_index >= 0),
        entity_id TEXT NOT NULL,
        role TEXT NOT NULL
            CHECK (length(trim(role)) > 0),

        PRIMARY KEY (event_id, participant_index),

        FOREIGN KEY (event_id)
            REFERENCES events(event_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,

        FOREIGN KEY (entity_id)
            REFERENCES entities(entity_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE relationships (
        relationship_id TEXT PRIMARY KEY NOT NULL,
        source_entity_id TEXT NOT NULL,
        target_entity_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL
            CHECK (length(trim(relationship_type)) > 0),
        observation_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        attributes_json TEXT NOT NULL,
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1),

        CHECK (source_entity_id != target_entity_id),

        FOREIGN KEY (source_entity_id)
            REFERENCES entities(entity_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,

        FOREIGN KEY (target_entity_id)
            REFERENCES entities(entity_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,

        FOREIGN KEY (observation_id)
            REFERENCES observations(observation_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX idx_observations_source_received
    ON observations(source_id, received_at)
    """,
    """
    CREATE INDEX idx_quarantine_source_received
    ON quarantined_observations(source_id, received_at)
    """,
    """
    CREATE INDEX idx_events_observation
    ON events(observation_id)
    """,
    """
    CREATE INDEX idx_events_occurred_at
    ON events(occurred_at)
    """,
    """
    CREATE INDEX idx_participants_entity
    ON event_participants(entity_id)
    """,
    """
    CREATE INDEX idx_relationships_source_time
    ON relationships(source_entity_id, observed_at)
    """,
    """
    CREATE INDEX idx_relationships_target_time
    ON relationships(target_entity_id, observed_at)
    """,
    """
    CREATE INDEX idx_relationships_observation
    ON relationships(observation_id)
    """,
)

_EVIDENCE_SET_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE evidence_sets (
        evidence_set_id TEXT PRIMARY KEY NOT NULL,
        created_at TEXT NOT NULL,
        schema_version INTEGER NOT NULL
            CHECK (schema_version = 1)
    )
    """,
    """
    CREATE TABLE evidence_set_items (
        evidence_set_id TEXT NOT NULL,
        item_index INTEGER NOT NULL
            CHECK (item_index >= 0),
        observation_id TEXT,
        event_id TEXT,

        PRIMARY KEY (evidence_set_id, item_index),

        CHECK (
            (observation_id IS NOT NULL AND event_id IS NULL)
            OR (observation_id IS NULL AND event_id IS NOT NULL)
        ),

        FOREIGN KEY (evidence_set_id)
            REFERENCES evidence_sets(evidence_set_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,

        FOREIGN KEY (observation_id)
            REFERENCES observations(observation_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,

        FOREIGN KEY (event_id)
            REFERENCES events(event_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE UNIQUE INDEX idx_evidence_set_items_observation
    ON evidence_set_items(evidence_set_id, observation_id)
    WHERE observation_id IS NOT NULL
    """,
    """
    CREATE UNIQUE INDEX idx_evidence_set_items_event
    ON evidence_set_items(evidence_set_id, event_id)
    WHERE event_id IS NOT NULL
    """,
)


_EXPECTED_COLUMNS = {
    "sources": (
        ("source_id", "TEXT", 1, 1),
        ("source_type", "TEXT", 1, 0),
        ("name", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "observations": (
        ("observation_id", "TEXT", 1, 1),
        ("source_id", "TEXT", 1, 0),
        ("received_at", "TEXT", 1, 0),
        ("raw_payload", "TEXT", 1, 0),
        ("payload_sha256", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "quarantined_observations": (
        ("quarantine_id", "TEXT", 1, 1),
        ("source_id", "TEXT", 1, 0),
        ("received_at", "TEXT", 1, 0),
        ("raw_payload", "TEXT", 1, 0),
        ("payload_sha256", "TEXT", 1, 0),
        ("rejection_code", "TEXT", 1, 0),
        ("rejection_reason", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "entities": (
        ("entity_id", "TEXT", 1, 1),
        ("entity_type", "TEXT", 1, 0),
        ("canonical_identifier", "TEXT", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "events": (
        ("event_id", "TEXT", 1, 1),
        ("observation_id", "TEXT", 1, 0),
        ("event_type", "TEXT", 1, 0),
        ("occurred_at", "TEXT", 1, 0),
        ("original_timestamp", "TEXT", 1, 0),
        ("normalized_data_json", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "event_participants": (
        ("event_id", "TEXT", 1, 1),
        ("participant_index", "INTEGER", 1, 2),
        ("entity_id", "TEXT", 1, 0),
        ("role", "TEXT", 1, 0),
    ),
    "relationships": (
        ("relationship_id", "TEXT", 1, 1),
        ("source_entity_id", "TEXT", 1, 0),
        ("target_entity_id", "TEXT", 1, 0),
        ("relationship_type", "TEXT", 1, 0),
        ("observation_id", "TEXT", 1, 0),
        ("observed_at", "TEXT", 1, 0),
        ("attributes_json", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "evidence_sets": (
        ("evidence_set_id", "TEXT", 1, 1),
        ("created_at", "TEXT", 1, 0),
        ("schema_version", "INTEGER", 1, 0),
    ),
    "evidence_set_items": (
        ("evidence_set_id", "TEXT", 1, 1),
        ("item_index", "INTEGER", 1, 2),
        ("observation_id", "TEXT", 0, 0),
        ("event_id", "TEXT", 0, 0),
    ),
}


_EXPECTED_INDEXES = {
    "idx_observations_source_received": (
        "observations",
        ("source_id", "received_at"),
    ),
    "idx_quarantine_source_received": (
        "quarantined_observations",
        ("source_id", "received_at"),
    ),
    "idx_events_observation": (
        "events",
        ("observation_id",),
    ),
    "idx_events_occurred_at": (
        "events",
        ("occurred_at",),
    ),
    "idx_participants_entity": (
        "event_participants",
        ("entity_id",),
    ),
    "idx_relationships_source_time": (
        "relationships",
        ("source_entity_id", "observed_at"),
    ),
    "idx_relationships_target_time": (
        "relationships",
        ("target_entity_id", "observed_at"),
    ),
    "idx_relationships_observation": (
        "relationships",
        ("observation_id",),
    ),
    "idx_evidence_set_items_observation": (
        "evidence_set_items",
        ("evidence_set_id", "observation_id"),
    ),
    "idx_evidence_set_items_event": (
        "evidence_set_items",
        ("evidence_set_id", "event_id"),
    ),
}


_EXPECTED_FOREIGN_KEYS = {
    "observations": {
        ("sources", "source_id", "source_id", "RESTRICT", "RESTRICT"),
    },
    "quarantined_observations": {
        ("sources", "source_id", "source_id", "RESTRICT", "RESTRICT"),
    },
    "events": {
        ("observations", "observation_id", "observation_id", "RESTRICT", "RESTRICT"),
    },
    "event_participants": {
        ("events", "event_id", "event_id", "RESTRICT", "RESTRICT"),
        ("entities", "entity_id", "entity_id", "RESTRICT", "RESTRICT"),
    },
    "relationships": {
        ("entities", "source_entity_id", "entity_id", "RESTRICT", "RESTRICT"),
        ("entities", "target_entity_id", "entity_id", "RESTRICT", "RESTRICT"),
        ("observations", "observation_id", "observation_id", "RESTRICT", "RESTRICT"),
    },
    "evidence_set_items": {
        ("evidence_sets", "evidence_set_id", "evidence_set_id", "RESTRICT", "RESTRICT"),
        ("observations", "observation_id", "observation_id", "RESTRICT", "RESTRICT"),
        ("events", "event_id", "event_id", "RESTRICT", "RESTRICT"),
    },
}


def _schema_definitions(
    connection: sqlite3.Connection,
) -> dict[tuple[str, str], str]:
    """Return normalized SQL definitions for explicit schema objects."""

    rows = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'index', 'view', 'trigger')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        """
    ).fetchall()

    return {
        (row["type"], row["name"]): " ".join(row["sql"].split())
        for row in rows
    }


class Database:
    """Manage safe connections and initialization of a local SQLite database."""

    def __init__(self, path: str | Path, timeout: float = 5.0) -> None:
        database_path = Path(path).expanduser()

        if not database_path.is_absolute():
            raise ValueError("database path must be absolute")

        if timeout <= 0:
            raise ValueError("database timeout must be greater than zero")

        self.path = database_path
        self.timeout = timeout

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection with foreign keys enabled and safe cleanup."""

        connection = sqlite3.connect(
            str(self.path),
            timeout=self.timeout,
        )
        connection.row_factory = sqlite3.Row

        try:
            connection.execute("PRAGMA foreign_keys = ON")

            result = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()

            if result is None or result[0] != 1:
                raise DatabaseError(
                    "SQLite foreign-key enforcement is unavailable"
                )

            yield connection
            connection.commit()

        except BaseException:
            connection.rollback()
            raise

        finally:
            connection.close()

    def initialize(self) -> None:
        """Create a new database without overwriting existing structures."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]

            existing_tables = {
                row["name"]
                for row in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name NOT LIKE 'sqlite_%'
                    """
                )
            }

            if version == 0:
                if existing_tables:
                    raise DatabaseInitializationError(
                        "database contains existing tables but has no "
                        "recognized schema version; refusing initialization"
                    )

                for statement in _SCHEMA_STATEMENTS + _EVIDENCE_SET_SCHEMA_STATEMENTS:
                    connection.execute(statement)

                connection.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}"
                )

            elif version == 1:
                # M7 adds immutable, ordered EvidenceSet persistence without
                # rewriting any existing source records.
                for statement in _EVIDENCE_SET_SCHEMA_STATEMENTS:
                    connection.execute(statement)

                connection.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}"
                )

            elif version != SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"database schema version {version} is unsupported; "
                    f"expected version {SCHEMA_VERSION}"
                )

            self._verify_schema(connection)

    def _verify_schema(self, connection: sqlite3.Connection) -> None:
        """Verify structure, constraints, and database integrity."""

        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        if version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"expected schema version {SCHEMA_VERSION}, got {version}"
            )

        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                """
            )
        }

        if tables != REQUIRED_TABLES:
            raise DatabaseIntegrityError(
                "database table set does not match the expected schema"
            )

        for table, expected in _EXPECTED_COLUMNS.items():
            actual = tuple(
                (
                    row["name"],
                    row["type"].upper(),
                    row["notnull"],
                    row["pk"],
                )
                for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                )
            )

            if actual != expected:
                raise DatabaseIntegrityError(
                    f"unexpected column structure in table: {table}"
                )

        for index_name, (expected_table, expected_columns) in (
            _EXPECTED_INDEXES.items()
        ):
            index = connection.execute(
                """
                SELECT tbl_name
                FROM sqlite_master
                WHERE type = 'index' AND name = ?
                """,
                (index_name,),
            ).fetchone()

            if index is None:
                raise DatabaseIntegrityError(
                    f"missing required index: {index_name}"
                )

            if index["tbl_name"] != expected_table:
                raise DatabaseIntegrityError(
                    f"index {index_name} belongs to the wrong table"
                )

            actual_columns = tuple(
                row["name"]
                for row in connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                )
            )

            if actual_columns != expected_columns:
                raise DatabaseIntegrityError(
                    f"incorrect columns in index: {index_name}"
                )

        for table, expected in _EXPECTED_FOREIGN_KEYS.items():
            actual = {
                (
                    row["table"],
                    row["from"],
                    row["to"],
                    row["on_update"],
                    row["on_delete"],
                )
                for row in connection.execute(
                    f'PRAGMA foreign_key_list("{table}")'
                )
            }

            if actual != expected:
                raise DatabaseIntegrityError(
                    f"incorrect foreign keys in table: {table}"
                )

        unique_entity_indexes = connection.execute(
            'PRAGMA index_list("entities")'
        ).fetchall()

        identity_constraint_found = False

        for index in unique_entity_indexes:
            if index["unique"] != 1:
                continue

            index_name = index["name"]

            actual_columns = tuple(
                row["name"]
                for row in connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                )
            )

            if actual_columns == (
                "entity_type",
                "canonical_identifier",
            ):
                identity_constraint_found = True
                break

        if not identity_constraint_found:
            raise DatabaseIntegrityError(
                "entity identity uniqueness constraint is missing"
            )

        integrity_results = connection.execute(
            "PRAGMA integrity_check"
        ).fetchall()

        if (
            len(integrity_results) != 1
            or integrity_results[0][0] != "ok"
        ):
            raise DatabaseIntegrityError(
                "SQLite integrity check failed"
            )

        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if foreign_key_errors:
            raise DatabaseIntegrityError(
                "database contains foreign-key violations"
            )

        # Build a reference schema using the canonical SQL statements.
        # This catches missing or modified CHECK, UNIQUE, and other
        # constraints that PRAGMA introspection alone cannot fully verify.
        with sqlite3.connect(":memory:") as reference:
            reference.row_factory = sqlite3.Row

            for statement in _SCHEMA_STATEMENTS + _EVIDENCE_SET_SCHEMA_STATEMENTS:
                reference.execute(statement)

            expected_definitions = _schema_definitions(reference)

        actual_definitions = _schema_definitions(connection)

        if actual_definitions != expected_definitions:
            missing = set(expected_definitions) - set(actual_definitions)
            unexpected = set(actual_definitions) - set(expected_definitions)

            changed = {
                key
                for key in set(expected_definitions) & set(actual_definitions)
                if expected_definitions[key] != actual_definitions[key]
            }

            details = []

            if missing:
                details.append(
                    f"missing objects: {sorted(missing)}"
                )

            if unexpected:
                details.append(
                    f"unexpected objects: {sorted(unexpected)}"
                )

            if changed:
                details.append(
                    f"modified definitions: {sorted(changed)}"
                )

            raise DatabaseIntegrityError(
                "database schema definitions do not match the canonical "
                "schema; " + "; ".join(details)
            )

    def verify_schema(self) -> None:
        """Verify the database schema using a fresh connection."""

        with self.connection() as connection:
            self._verify_schema(connection)
