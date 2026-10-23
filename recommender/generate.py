import datetime
import random
import uuid
from typing import TypedDict

import orjson
from tqdm import tqdm

import recommender.pol_connector as pc
from recommender.data.dataset import (
    get_pageview_interactions,
    get_users,
    list_pending_editorial_recommendations,
)
from recommender.data.upload import (
    activate_editorial,
    end_active_runs_and_models,
    expire_old_editorials,
    insert_models,
    insert_recommendation_lists,
    insert_recommendation_run,
)
from recommender.models import (
    CFRecommenderFiltered,
    EditorialRecommender,
    RecommenderBase,
)
from recommender.schemas.recommendation_list import RecommendationList
from recommender.utils import candidates, s3
from recommender.utils.export import format_export_payload
from recommender.utils.sanitizers import RecImpressionSanitizer

INTERACTION_LOOKBACK_DAYS = 365
CONTROL_GROUP_FRACTION = 0


class UserState(TypedDict):
    model_ids: list[uuid.UUID]
    model_names: list[str]


def main() -> None:
    random.seed(42)
    run_id = uuid.uuid4()

    models: list[RecommenderBase] = [
        CFRecommenderFiltered(
            iterations=30,
            min_user_pv=25,
            max_user_pv=5000,
            min_item_pv=1000,
            max_item_pv=100_000,
        ),
    ]

    start_date = (
        datetime.date.today() - datetime.timedelta(days=INTERACTION_LOOKBACK_DAYS)
    ).isoformat()
    end_date = datetime.date.today().isoformat()

    # TODO: add retry logic
    interactions = get_pageview_interactions(start_date, end_date)

    users = get_users()
    sanitizer = RecImpressionSanitizer()
    candidate_list = candidates.get_last_month_candidates()

    # init pending editorial models
    editorial_recommendation_ids = list_pending_editorial_recommendations()
    for edi_id, _ in editorial_recommendation_ids:
        models.append(EditorialRecommender(edi_id))

    # fit models
    for model in models:
        model.fit(interactions)

    sampled_users = random.sample(users, int(len(users) * (1 - CONTROL_GROUP_FRACTION)))
    sampled_users_set = set(sampled_users)
    control_group_users = {
        user_id for user_id in users if user_id not in sampled_users_set
    }

    user_recommendations: dict[str, RecommendationList] = {}
    user_states: dict[str, UserState] = {}

    # Generate recommendations for sampled users
    for user_id in tqdm(
        sampled_users, desc="Generating recommendations for sampled users"
    ):
        eligible_models = [model for model in models if model.can_recommend(user_id)]
        if len(eligible_models) == 0:
            user_recommendations[user_id] = RecommendationList(recommendations=[])
            user_states[user_id] = {"model_ids": [], "model_names": []}
            continue

        model = random.choice(eligible_models)
        recs = model.recommend(
            user_id,
            k=25,
            candidates=candidate_list,
            filter_already_liked_items=True,
        )
        recs = sanitizer(user_id, recs).top_k(5)
        user_recommendations[user_id] = recs

        model_id_state = [model.model_id for model in eligible_models]
        model_name_state = [model.name for model in eligible_models]
        user_states[user_id] = {
            "model_ids": model_id_state,
            "model_names": model_name_state,
        }

    # Create empty recommendation lists for control group users
    for user_id in tqdm(control_group_users, desc="Creating control group entries"):
        user_recommendations[user_id] = RecommendationList(recommendations=[])
        eligible_models = [model for model in models if model.can_recommend(user_id)]
        model_id_state = [model.model_id for model in eligible_models]
        model_name_state = [model.name for model in eligible_models]
        user_states[user_id] = {
            "model_ids": model_id_state,
            "model_names": model_name_state,
        }

    payload = format_export_payload(user_recommendations)
    body = orjson.dumps(payload)
    with pc.connect() as conn:
        try:
            # End active runs and their models
            end_active_runs_and_models(conn)

            # Insert new recommendation run
            insert_recommendation_run(
                conn,
                run_id,
                training_start=start_date,
                training_end=end_date,
                num_interactions=len(interactions),
                verbose=True,
            )

            # Insert new models as active
            insert_models(conn, models, run_id, verbose=True)

            # Insert recommendation lists
            insert_recommendation_lists(
                conn,
                user_recommendations,
                user_states,
                control_group_users,
                run_id,
                verbose=True,
            )

            # Expire old editorials
            expire_old_editorials(conn)

            # Activate newly used editorials
            for model in models:
                if isinstance(model, EditorialRecommender):
                    activate_editorial(conn, model.editorial_id)

            # Upload to S3 for Steen
            key = f"inbox/{run_id}.json"
            s3.upload_to_s3(key, body)

            # Commit transaction
            conn.commit()

        except Exception as e:
            # Rollback on error
            conn.rollback()
            raise e


if __name__ == "__main__":
    main()
