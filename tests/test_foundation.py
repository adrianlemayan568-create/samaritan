import sqlite3

import samaritan


def test_package_version():
    assert samaritan.__version__ == "0.1.0"


def test_sqlite_is_available():
    connection = sqlite3.connect(":memory:")

    try:
        result = connection.execute("SELECT sqlite_version()").fetchone()
        assert result is not None
        assert result[0]
    finally:
        connection.close()
