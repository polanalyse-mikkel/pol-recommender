import inspect
import sys
from datetime import datetime, timedelta

from psycopg import sql

import recommender.pol_connector as pc


def get_candidates() -> list[int]:
    query = "select article_id from politiken_ia_prv.articles_mv"
    with pc.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def get_first_test_candidates() -> list[int]:
    """
    Candidate list for the first recommendation test starting Monday 30th June 2025.
    """
    query = """
    select art.article_id
    from politiken_ia_prv.layer_three_article_performance_all_time perf
    join politiken_ia_prv.articles_mv art
    on perf.article_id = art.article_id
    where art.first_published_date >= '2025-06-01'
      and points_total >= 120
      and is_super_template = false
      and is_liveblog = false
      and sum_pv >= 10000
      and art.article_id not in (10407101, 10439954, 10449814, 10434113)
    order by points_total desc;
    """
    with pc.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def get_second_test_candidates() -> list[int]:
    """
    Candidate list for the second recommendation test starting Monday 2nd July 2025.
    """
    query = """
    select art.article_id
    from politiken_ia_prv.layer_three_article_performance_all_time perf
    join politiken_ia_prv.articles_mv art
    on perf.article_id = art.article_id
    where art.first_published_date >= '2025-06-01'
      and points_total >= 120
      and is_liveblog = false
      and sum_pv >= 10000
      and art.article_id not in (10407101, 10439954, 10449814, 10434113)
    order by points_total desc;
    """
    with pc.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def get_first_production_5050_candidates() -> list[int]:
    """
    Candidate list for the first large scale recommendation test, starting Thursday 3rd July 2025.
    """
    query = """
    select art.article_id
    from politiken_ia_prv.layer_three_article_performance_all_time perf
    join politiken_ia_prv.articles_mv art
    on perf.article_id = art.article_id
    where art.first_published_date >= '2025-06-01'
      and points_total >= 120
      and is_liveblog = false
      and sum_pv >= 10000
      and user_needs != 'Opdatér mig'
      and art.article_id not in (10407101, 10439954, 10449814, 10434113)
    order by points_total desc;
    """
    with pc.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def get_august_2025_recommendation_candidates() -> list[int]:
    """
    Candidate list for the first full scale recommendation test, starting Thursday 12th August 2025.
    """
    query = """
    select art.article_id
    from politiken_ia_prv.layer_three_article_performance_all_time perf
    join politiken_ia_prv.articles_mv art
    on perf.article_id = art.article_id
    where art.first_published_date >= '2025-08-01'
      and points_total >= 140
      and is_liveblog = false
      and sum_pv >= 10000
      and user_needs != 'Opdatér mig'
      and art.is_ritzau = false
      and art.secondary_section != 'cykling'
    order by points_total desc;"""
    with pc.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def get_last_month_candidates() -> list[int]:
    """
    Candidate list for the last month.
    """
    date_str = (datetime.now() - timedelta(days=31)).date().strftime("%Y-%m-%d")
    query = sql.SQL("""
    select art.article_id
    from politiken_ia_prv.layer_three_article_performance_all_time perf
    join politiken_ia_prv.articles_mv art
    on perf.article_id = art.article_id
    where art.first_published_date >= %s
      and points_total >= 140
      and is_liveblog = false
      and sum_pv >= 10000
      and user_needs != 'Opdatér mig'
      and art.is_ritzau = false
      and art.primary_section != 'sport'
      and art.tertiary_section != 'roskildefestival'
    order by points_total desc;""")
    with pc.connect() as conn:
        rows = conn.execute(query, (date_str,)).fetchall()
    return [r[0] for r in rows]


def get_last_3_days_candidates() -> list[int]:
    """
    Candidate list for the last 3 days.
    """
    date_str = (datetime.now() - timedelta(days=3)).date().strftime("%Y-%m-%d")
    query = sql.SQL("""
    select art.article_id
    from politiken_ia_prv.layer_three_article_performance_all_time perf
    join politiken_ia_prv.articles_mv art
    on perf.article_id = art.article_id
    where art.first_published_date >= {}
      and points_total >= 140
      and is_liveblog = false
      and sum_pv >= 10000
      and user_needs != 'Opdatér mig'
      and art.is_ritzau = false
      and art.primary_section != 'sport'
    order by points_total desc;""").format(date_str)
    with pc.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def list_candidate_functions():
    """
    Lists all available candidate-generating functions in this module.
    """
    functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    candidate_functions = [
        name
        for name, func in functions
        if name.startswith("get_") and "candidates" in name
    ]
    return candidate_functions


if __name__ == "__main__":
    print(list_candidate_functions())
