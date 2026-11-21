"""Tests for Samaritan's SQLite database foundation."""

import sqlite3

import pytest

from samaritan.database import (
    Database,
    DatabaseInitializationError,
    DatabaseIntegrityError,
    UnsupportedSchemaVersionError,
)


def make_database(tmp_path):
    database = Database(tmp_path / "samaritan.db")
    database.initialize()
    return database


def insert_source(connection, source_id="source-1"):
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
        (
            source_id,
            "synthetic",
            "test source",
            "2026-11-13T10:00:00+00:00",
            1,
        ),
    )


def insert_observation(connection, observation_id="observation-1"):
    insert_source(connection)

    connection.execute(
        """
        INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            "source-1",
            "2026-11-13T10:00:00+00:00",
            "{}",
            "0" * 64,
            1,
        ),
    )


def test_database_requires_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        Database("relative.db")


def test_database_requires_positive_timeout(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        Database(tmp_path / "test.db", timeout=0)


def test_initialization_creates_nine_tables(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()

    assert len(rows) == 9


def test_database_schema_version_is_two(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

    assert version == 2



def test_schema_version_one_migrates_to_two_without_losing_records(tmp_path):
    from samaritan.database import _SCHEMA_STATEMENTS

    path = tmp_path / "legacy-v1.db"
    with sqlite3.connect(path) as connection:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
        insert_source(connection)
        insert_observation(connection)

    database = Database(path)
    database.initialize()

    with database.connection() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        source_count = connection.execute(
            "SELECT COUNT(*) FROM sources"
        ).fetchone()[0]
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        evidence_set_tables = connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('evidence_sets', 'evidence_set_items')
            """
        ).fetchone()[0]

    assert version == 2
    assert source_count == 1
    assert observation_count == 1
    assert evidence_set_tables == 2
    database.verify_schema()

def test_foreign_keys_are_enabled_on_every_connection(tmp_path):
    database = make_database(tmp_path)

    for _ in range(3):
        with database.connection() as connection:
            result = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()[0]

            assert result == 1


def test_missing_source_reference_is_rejected(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
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
                    "observation-1",
                    "missing-source",
                    "2026-11-13T10:00:00+00:00",
                    "{}",
                    "0" * 64,
                    1,
                ),
            )


def test_duplicate_entity_identity_is_rejected(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            connection.execute(
                """
                INSERT INTO entities VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "entity-1",
                    "user",
                    "alice",
                    "2026-11-13T10:00:00+00:00",
                    1,
                ),
            )

            connection.execute(
                """
                INSERT INTO entities VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "entity-2",
                    "user",
                    "alice",
                    "2026-11-13T10:01:00+00:00",
                    1,
                ),
            )


def test_same_identifier_with_different_entity_types_is_allowed(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
            (
                "entity-1",
                "user",
                "shared-id",
                "2026-11-13T10:00:00+00:00",
                1,
            ),
        )

        connection.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
            (
                "entity-2",
                "device",
                "shared-id",
                "2026-11-13T10:00:00+00:00",
                1,
            ),
        )

        count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

    assert count == 2


def test_self_referencing_relationship_is_rejected(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            connection.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                (
                    "entity-1",
                    "user",
                    "alice",
                    "2026-11-13T10:00:00+00:00",
                    1,
                ),
            )

            insert_observation(connection)

            connection.execute(
                """
                INSERT INTO relationships VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    "relationship-1",
                    "entity-1",
                    "entity-1",
                    "communicates_with",
                    "observation-1",
                    "2026-11-13T10:00:00+00:00",
                    "{}",
                    1,
                ),
            )


def test_existing_unrecognized_database_is_not_overwritten(tmp_path):
    path = tmp_path / "existing.db"

    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE unrelated_data (value TEXT)"
        )

        connection.execute(
            "INSERT INTO unrelated_data VALUES ('preserve me')"
        )

    database = Database(path)

    with pytest.raises(DatabaseInitializationError):
        database.initialize()

    with sqlite3.connect(path) as connection:
        value = connection.execute(
            "SELECT value FROM unrelated_data"
        ).fetchone()[0]

    assert value == "preserve me"


def test_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "future.db"

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    database = Database(path)

    with pytest.raises(UnsupportedSchemaVersionError):
        database.initialize()


def test_repeated_initialization_preserves_data(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
            (
                "entity-1",
                "user",
                "alice",
                "2026-11-13T10:00:00+00:00",
                1,
            ),
        )

    database.initialize()

    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

    assert count == 1


def test_failed_transaction_rolls_back_all_changes(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            connection.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                (
                    "entity-1",
                    "user",
                    "alice",
                    "2026-11-13T10:00:00+00:00",
                    1,
                ),
            )

            connection.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                (
                    "entity-2",
                    "user",
                    "alice",
                    "2026-11-13T10:00:00+00:00",
                    1,
                ),
            )

    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

    assert count == 0


# ---------------------------------------------------------
# Schema verification tests
# ---------------------------------------------------------


def test_verify_schema_accepts_healthy_database(tmp_path):
    database = make_database(tmp_path)

    database.verify_schema()


def test_initialization_verifies_existing_schema(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "DROP INDEX idx_events_occurred_at"
        )

    with pytest.raises(DatabaseIntegrityError, match="index"):
        database.initialize()


def test_verify_schema_rejects_missing_index(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "DROP INDEX idx_observations_source_received"
        )

    with pytest.raises(DatabaseIntegrityError, match="index"):
        database.verify_schema()


def test_verify_schema_rejects_unexpected_column(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "ALTER TABLE sources ADD COLUMN unexpected_column TEXT"
        )

    with pytest.raises(DatabaseIntegrityError, match="column structure"):
        database.verify_schema()


def test_verify_schema_rejects_unexpected_table(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "CREATE TABLE unexpected_table (value TEXT)"
        )

    with pytest.raises(DatabaseIntegrityError, match="table set"):
        database.verify_schema()


def test_verify_schema_rejects_foreign_key_violations(tmp_path):
    database = make_database(tmp_path)

    # Deliberately bypass foreign-key enforcement to simulate
    # a database containing an orphaned observation.
    with sqlite3.connect(database.path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")

        connection.execute(
            """
            INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "orphan-observation",
                "missing-source",
                "2026-11-13T10:00:00+00:00",
                "{}",
                "0" * 64,
                1,
            ),
        )

    with pytest.raises(DatabaseIntegrityError, match="foreign-key"):
        database.verify_schema()


# ---------------------------------------------------------
# Database-level CHECK constraint tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("source_type", "schema_version"),
    [
        ("", 1),
        ("synthetic", 2),
    ],
)
def test_source_check_constraints_are_enforced(
    tmp_path,
    source_type,
    schema_version,
):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (
                    "source-1",
                    source_type,
                    "test source",
                    "2026-11-13T10:00:00+00:00",
                    schema_version,
                ),
            )


def test_entity_check_constraints_are_enforced(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            connection.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                (
                    "entity-1",
                    "user",
                    "   ",
                    "2026-11-13T10:00:00+00:00",
                    1,
                ),
            )


def test_observation_checksum_constraint_is_enforced(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            insert_source(connection)

            connection.execute(
                """
                INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "observation-1",
                    "source-1",
                    "2026-11-13T10:00:00+00:00",
                    "{}",
                    "G" * 64,
                    1,
                ),
            )


def test_quarantine_rejection_reason_constraint_is_enforced(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            insert_source(connection)

            connection.execute(
                """
                INSERT INTO quarantined_observations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "quarantine-1",
                    "source-1",
                    "2026-11-13T10:00:00+00:00",
                    "invalid-json",
                    "0" * 64,
                    "INVALID_JSON",
                    "   ",
                    1,
                ),
            )


def test_event_check_constraints_are_enforced(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            insert_observation(connection)

            connection.execute(
                """
                INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "event-1",
                    "observation-1",
                    "",
                    "2026-11-13T10:00:00+00:00",
                    "2026-11-13T10:00:00+00:00",
                    "{}",
                    1,
                ),
            )


def test_event_participant_index_constraint_is_enforced(tmp_path):
    database = make_database(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        with database.connection() as connection:
            insert_observation(connection)

            connection.execute(
                """
                INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "event-1",
                    "observation-1",
                    "authentication",
                    "2026-11-13T10:00:00+00:00",
                    "2026-11-13T10:00:00+00:00",
                    "{}",
                    1,
                ),
            )

            connection.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                (
                    "entity-1",
                    "user",
                    "alice",
                    "2026-11-13T10:00:00+00:00",
                    1,
                ),
            )

            connection.execute(
                """
                INSERT INTO event_participants VALUES (?, ?, ?, ?)
                """,
                ("event-1", -1, "entity-1", "actor"),
            )


# ---------------------------------------------------------
# Canonical schema comparison tests
# ---------------------------------------------------------


def create_modified_schema_database(tmp_path, modification):
    """Create a version-two database with a deliberately modified schema."""

    from samaritan.database import (
        _EVIDENCE_SET_SCHEMA_STATEMENTS,
        _SCHEMA_STATEMENTS,
    )

    path = tmp_path / "modified-schema.db"

    with sqlite3.connect(path) as connection:
        for statement in _SCHEMA_STATEMENTS + _EVIDENCE_SET_SCHEMA_STATEMENTS:
            connection.execute(modification(statement))

        connection.execute("PRAGMA user_version = 2")

    return Database(path)


def test_verify_schema_rejects_removed_check_constraint(tmp_path):
    import re

    def remove_check(statement):
        return re.sub(
            r"(source_type TEXT NOT NULL)\s+CHECK "
            r"\(length\(trim\(source_type\)\) > 0\),",
            r"\1,",
            statement,
            count=1,
        )

    database = create_modified_schema_database(
        tmp_path,
        remove_check,
    )

    with pytest.raises(DatabaseIntegrityError, match="definitions"):
        database.verify_schema()


def test_verify_schema_rejects_removed_entity_unique_constraint(tmp_path):
    def remove_unique(statement):
        return statement.replace(
            "            CHECK (schema_version = 1),\n\n"
            "        UNIQUE (entity_type, canonical_identifier)",
            "            CHECK (schema_version = 1)",
        )

    database = create_modified_schema_database(
        tmp_path,
        remove_unique,
    )

    with pytest.raises(
        DatabaseIntegrityError,
        match="uniqueness",
    ):
        database.verify_schema()


def test_verify_schema_rejects_unexpected_explicit_index(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "CREATE INDEX unexpected_index ON sources(name)"
        )

    with pytest.raises(
        DatabaseIntegrityError,
        match="definitions",
    ):
        database.verify_schema()


def test_verify_schema_rejects_unexpected_view(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            "CREATE VIEW unexpected_view AS "
            "SELECT source_id FROM sources"
        )

    with pytest.raises(DatabaseIntegrityError, match="definitions"):
        database.verify_schema()


def test_verify_schema_rejects_unexpected_trigger(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER unexpected_trigger
            AFTER INSERT ON sources
            BEGIN
                SELECT 1;
            END
            """
        )

    with pytest.raises(DatabaseIntegrityError, match="definitions"):
        database.verify_schema()


def test_verify_schema_rejects_unsealed_evidence_set_record(tmp_path):
    database = make_database(tmp_path)

    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO evidence_sets (
                evidence_set_id, created_at, schema_version, is_sealed
            )
            VALUES (?, ?, ?, 0)
            """,
            (
                "unsealed-evidence-set",
                "2026-11-13T10:00:00+00:00",
                1,
            ),
        )

    with pytest.raises(
        DatabaseIntegrityError,
        match="unsealed evidence set",
    ):
        database.verify_schema()
