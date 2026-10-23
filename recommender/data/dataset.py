"""Dataset loader for getting user-item interactions from Redshift."""

from typing import List, Optional

import numpy as np
import polars as pl
from psycopg import Connection
from psycopg.rows import tuple_row

import recommender.pol_connector as pc


def generate_synthetic_interactions(
    n_users: int = 100, n_items: int = 50, interactions: int = 500, seed=42
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(0, n_users, size=interactions)
    item_ids = rng.integers(0, n_items, size=interactions)

    return pl.DataFrame({"user_id": user_ids, "item_id": item_ids})


def get_pageview_interactions(
    start_date: str,
    end_date: Optional[str] = None,
    *,
    filter_active: bool = False,
) -> pl.DataFrame:
    churn_filter = "and is_churned = false" if filter_active else ""
    query = f"""
    with users as (
        select distinct sso_id, sso_id_hash
        from politiken_ia_prv.subscriptions_mv
        where sso_id != 'none'
        {churn_filter}
        union all
        select sso_id, sso_id_hash
        from politiken_ia_prv.internal_recommendation_users
    ), items as (
        select distinct article_id
        from politiken_ia_prv.articles_mv
    )
    select
        users.sso_id::varchar as user_id,
        listagg(pv.article_id::int, ',')    as item_ids
    from politiken_ia_prv.layer_one_sso_articles pv
    join users on pv.sso_id_hash = users.sso_id_hash
    join items on pv.article_id = items.article_id
    where event_date between '{start_date}' and '{end_date}'
    group by user_id
    """
    with pc.connect(row_factory=tuple_row) as conn:
        return (
            pl.read_database(query, conn, schema_overrides={"user_id": pl.Utf8})
            .with_columns(pl.col("item_ids").str.split(","))
            .explode("item_ids")
            .rename({"item_ids": "item_id"})
            .with_columns(pl.col("item_id").cast(pl.UInt32))
        )


def get_users(filter_active: bool = True) -> List[str]:
    active_filter = "where has_active_subscription = true" if filter_active else ""
    query = f"""
    select sso_id
    from politiken_ia_prv.accounts_mv
    {active_filter}
    union all
    select sso_id
    from politiken_ia_prv.internal_recommendation_users
    """
    with pc.connect() as conn:
        users = pl.read_database(query, conn)
    return users["sso_id"].to_list()


def list_pending_editorial_recommendations() -> list[tuple[str, str]]:
    query = """
        SELECT editorial_id, editorial_name
        FROM politiken_ia_prv.recommendation_editorials
        WHERE editorial_status in ('pending', 'active')
          and GETDATE()::timestamptz <= expire_at
    """
    with pc.connect(row_factory=tuple_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return rows


def set_editorial_recommendation_status(
    conn: Connection, editorial_id: str, status: str
) -> None:
    query = """
        UPDATE politiken_ia_prv.recommendation_editorials
        SET editorial_status = %s
        WHERE editorial_id = %s
    """
    conn.execute(query, (status, editorial_id))


if __name__ == "__main__":
    # print(list_pending_editorial_recommendations())
    # print(get_pageview_interactions("2025-09-01"))
    start_date = "2025-06-01"
    end_date = "2026-06-01"
    interactions = get_pageview_interactions(start_date, end_date)
    print(interactions.head())
    print(interactions.height)
