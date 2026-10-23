from itertools import islice
from typing import Any, Mapping
from datetime import datetime
from uuid import UUID
from psycopg import Connection
from tqdm import tqdm
import orjson

from recommender.models.base import RecommenderBase
from recommender.models.editorial import EditorialRecommender
from recommender.schemas.recommendation_list import RecommendationList

BATCH_SIZE = 2_000


# =============================================================================
# PAYLOAD FORMATTERS
# =============================================================================


def format_model_payload(models: list[RecommenderBase], run_id: UUID) -> list[dict]:
    """Format model data for database insertion."""
    model_payload = []
    now = datetime.now()
    for model in models:
        editorial_id = (
            str(model.editorial_id) if isinstance(model, EditorialRecommender) else None
        )
        model_row = {
            "model_id": str(model.model_id),
            "run_id": str(run_id),
            "model_name": model.name,
            "model_category": model.category,
            "model_status": "active",
            "created_at": now.isoformat(),
            "editorial_id": editorial_id,
            "expire_at": model.expire_at.isoformat(),
            "model_hash": model.model_hash,
        }
        model_payload.append(model_row)
    return model_payload


def format_recommendation_list_payload(
    user_recommendations: dict[str, RecommendationList],
    user_states: Mapping[str, Any],
    control_group_users: set[str],
    run_id: UUID,
) -> list[dict]:
    """Format recommendation list data for database insertion."""
    recommendation_list_payload = []
    for user_id, rec_list in user_recommendations.items():
        # Get model state for this user
        user_state = user_states.get(user_id, {"model_ids": [], "model_names": []})

        # Check if this user is in the control group
        is_control = user_id in control_group_users

        # Get the model_id from the first recommendation (all recommendations in a list have the same model_id)
        # For control group, model_id will be None
        model_id = (
            str(rec_list.recommendations[0].model_id)
            if rec_list.recommendations
            else None
        )

        # Format recommendations as a list of dicts for SUPER column
        recommendations_data = [
            {
                "article_id": rec.item_id,
                "rank": rec.rank,
                "score": rec.score,
            }
            for rec in rec_list.recommendations
        ]

        list_row = {
            "list_id": str(rec_list.recommendation_list_id),
            "run_id": str(run_id),
            "model_id": model_id,
            "sso_id": user_id,
            "recommendations": orjson.dumps(recommendations_data).decode("utf-8"),
            "candidate_models": orjson.dumps(
                [str(mid) for mid in user_state["model_ids"]]
            ).decode("utf-8"),
            "candidate_model_names": orjson.dumps(user_state["model_names"]).decode(
                "utf-8"
            ),
            "num_recommendations": len(rec_list),
            "is_control": is_control,
        }
        recommendation_list_payload.append(list_row)
    return recommendation_list_payload


# =============================================================================
# RECOMMENDATION RUN OPERATIONS
# =============================================================================


def insert_recommendation_run(
    conn: Connection,
    run_id: UUID,
    training_start: str,
    training_end: str,
    num_interactions: int,
    config: dict | None = None,
    verbose: bool = False,
) -> None:
    """Insert a new recommendation run record with status 'active'."""
    config_json = orjson.dumps(config).decode("utf-8").replace("'", "''")

    with conn.cursor() as cur:
        query = f"""
        INSERT INTO politiken_ia_prv.recommendation_runs
        (run_id, run_status, training_start, training_end, num_interactions, config)
        VALUES (%s, %s, %s, %s, %s, JSON_PARSE('{config_json}'))
        """
        cur.execute(
            query,
            [
                str(run_id),
                "active",
                training_start,
                training_end,
                num_interactions,
            ],
        )
        if verbose:
            print(f"Inserted recommendation run: {run_id}")


def end_active_runs_and_models(conn: Connection) -> None:
    """
    End all currently active recommendation runs.
    Also ends all models associated with those runs to maintain consistency.
    """
    with conn.cursor() as cur:
        # End active runs
        cur.execute("""
            UPDATE politiken_ia_prv.recommendation_runs
            SET run_status = 'ended', ended_at = GETDATE()
            WHERE run_status = 'active'
        """)

        # End models that belong to the runs we just ended
        cur.execute("""
            UPDATE politiken_ia_prv.recommendation_models
            SET model_status = 'ended', modified_at = GETDATE()
            WHERE run_id IN (
                SELECT run_id
                FROM politiken_ia_prv.recommendation_runs
                WHERE run_status = 'ended'
            )
            AND model_status = 'active'
        """)


# =============================================================================
# MODEL OPERATIONS
# =============================================================================


def insert_models(
    conn: Connection, models: list[RecommenderBase], run_id: UUID, verbose: bool = False
) -> None:
    """Insert models with status 'active' for the given run."""
    payload = format_model_payload(models, run_id)
    with conn.cursor() as cur:
        it = iter(payload)
        with tqdm(
            total=len(payload),
            desc="Inserting models",
            unit="rows",
            colour="green",
            disable=not verbose,
        ) as pbar:
            while batch := list(islice(it, BATCH_SIZE)):
                placeholders = ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s,%s)"] * len(batch))
                flat = [
                    val
                    for row in batch
                    for val in (
                        row["model_id"],
                        row["run_id"],
                        row["model_name"],
                        row["model_category"],
                        row["model_status"],
                        row["created_at"],
                        row["editorial_id"],
                        row["expire_at"],
                        row["model_hash"],
                    )
                ]
                cur.execute(
                    f"INSERT INTO politiken_ia_prv.recommendation_models "
                    f"(model_id, run_id, model_name, model_category, model_status, created_at, editorial_id, expire_at, model_hash) "
                    f"VALUES {placeholders}",
                    flat,
                )
                pbar.update(len(batch))


# =============================================================================
# RECOMMENDATION LIST OPERATIONS
# =============================================================================


def insert_recommendation_lists(
    conn: Connection,
    user_recommendations: dict[str, RecommendationList],
    user_states: Mapping[str, Any],
    control_group_users: set[str],
    run_id: UUID,
    verbose: bool = False,
) -> None:
    """Insert recommendation lists for all users."""
    payload = format_recommendation_list_payload(
        user_recommendations, user_states, control_group_users, run_id
    )
    with conn.cursor() as cur:
        it = iter(payload)
        with tqdm(
            total=len(payload),
            desc="Inserting recommendation lists",
            unit="rows",
            colour="green",
            disable=not verbose,
        ) as pbar:
            while batch := list(islice(it, BATCH_SIZE)):
                # Build values with JSON_PARSE for SUPER columns
                # We need to inline JSON_PARSE() since psycopg can't handle SUPER type parameters
                values_parts = []
                params = []

                for row in batch:
                    # Escape single quotes by doubling them for SQL strings
                    recommendations_escaped = row["recommendations"].replace("'", "''")
                    candidate_models_escaped = row["candidate_models"].replace(
                        "'", "''"
                    )
                    candidate_model_names_escaped = row[
                        "candidate_model_names"
                    ].replace("'", "''")

                    # Build the VALUES clause with JSON_PARSE for SUPER columns
                    values_parts.append(
                        f"(%s,%s,%s,%s,JSON_PARSE('{recommendations_escaped}'),"
                        f"JSON_PARSE('{candidate_models_escaped}'),JSON_PARSE('{candidate_model_names_escaped}'),%s,%s)"
                    )

                    # Add non-SUPER column parameters
                    params.extend(
                        [
                            row["list_id"],
                            row["run_id"],
                            row["model_id"],
                            row["sso_id"],
                            row["num_recommendations"],
                            row["is_control"],
                        ]
                    )

                values_clause = ",".join(values_parts)

                cur.execute(
                    f"INSERT INTO politiken_ia_prv.recommendation_lists "
                    f"(list_id, run_id, model_id, sso_id, recommendations, candidate_models, candidate_model_names, num_recommendations, is_control) "
                    f"VALUES {values_clause}",
                    params,
                )
                pbar.update(len(batch))


# =============================================================================
# EDITORIAL OPERATIONS
# =============================================================================


def expire_old_editorials(conn: Connection) -> None:
    """End editorials that are active or pending but have passed their expiration date."""
    query = """
    UPDATE politiken_ia_prv.recommendation_editorials
    SET editorial_status = 'ended'
    WHERE editorial_status IN ('active', 'pending')
      AND GETDATE()::timestamptz > expire_at
    """
    conn.execute(query)


def activate_editorial(conn: Connection, editorial_id: UUID) -> None:
    """Set a specific editorial to active status."""
    query = """
    UPDATE politiken_ia_prv.recommendation_editorials
    SET editorial_status = 'active'
    WHERE editorial_id = %s
    """
    conn.execute(query, [str(editorial_id)])
