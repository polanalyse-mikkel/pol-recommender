"""pol_connector - Simple PostgreSQL/Redshift connection helper.

A thin wrapper around psycopg3 that provides a connect() function that
auto-resolves DSN from environment variables and defaults to namedtuple rows.

Usage:
    export POL_CONNECTOR_DSN="postgresql://user:pass@host:5439/db"

    import pol_connector as pc

    # Using the connect function (namedtuple rows by default)
    with pc.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (123,))
            user = cur.fetchone()
            print(user.name)  # attribute access works!

    # Using a different row factory
    from psycopg.rows import dict_row

    with pc.connect(row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS value")
            row = cur.fetchone()
            print(row['value'])  # dict access works!
"""

from __future__ import annotations

from .connection import connect

__version__ = "1.0.0"

# Public API
__all__ = [
    "connect",
]
