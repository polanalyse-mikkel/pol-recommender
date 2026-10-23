from __future__ import annotations

import os
from typing import NamedTuple, overload

from psycopg import Connection
from psycopg.abc import AdaptContext, ConnParam
from psycopg.cursor import Cursor
from psycopg.rows import Row, RowFactory, namedtuple_row


def _resolve_dsn(dsn: str = "") -> str:
    """Resolve DSN from argument or POL_CONNECTOR_DSN environment variable."""
    resolved = dsn or os.getenv("POL_CONNECTOR_DSN")
    if not resolved:
        raise ValueError(
            "DSN required: pass as argument or set POL_CONNECTOR_DSN environment variable"
        )
    return resolved


@overload
def connect(
    conninfo: str = ...,
    *,
    autocommit: bool = ...,
    prepare_threshold: int | None = ...,
    context: AdaptContext | None = ...,
    row_factory: RowFactory[Row],
    cursor_factory: type[Cursor[Row]] | None = ...,
    **kwargs: ConnParam,
) -> Connection[Row]: ...


@overload
def connect(
    conninfo: str = ...,
    *,
    autocommit: bool = ...,
    prepare_threshold: int | None = ...,
    context: AdaptContext | None = ...,
    row_factory: None = ...,
    cursor_factory: type[Cursor[NamedTuple]] | None = ...,
    **kwargs: ConnParam,
) -> Connection[NamedTuple]: ...


def connect(
    conninfo: str = "",
    *,
    autocommit: bool = False,
    prepare_threshold: int | None = 5,
    context: AdaptContext | None = None,
    row_factory: RowFactory[Row] | None = None,
    cursor_factory: type[Cursor[Row]] | None = None,
    **kwargs: ConnParam,
) -> Connection[Row] | Connection[NamedTuple]:
    """Connect to the database, resolving conninfo from
    POL_CONNECTOR_DSN if not given, and defaulting to namedtuple rows.

    Usage:
        # Default: namedtuple rows
        with connect() as conn:
            ...

        # Custom row factory
        with connect(row_factory=dict_row) as conn:
            ...
    """
    rf = row_factory if row_factory is not None else namedtuple_row
    return Connection.connect(
        _resolve_dsn(conninfo),
        autocommit=autocommit,
        prepare_threshold=prepare_threshold,
        context=context,
        row_factory=rf,  # type: ignore[arg-type]
        cursor_factory=cursor_factory,  # type: ignore[arg-type]
        **kwargs,
    )
