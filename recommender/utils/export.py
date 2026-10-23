from recommender.schemas.recommendation_list import RecommendationList


def format_export_payload(user_recommendations: dict[str, RecommendationList] = {}):
    payload = {}
    for user_id, rec_list in user_recommendations.items():
        items = rec_list.map(lambda item: item.item_id)
        if len(items) > 0:
            payload[user_id] = items
    return payload


def format_exo_export_payload(
    user_recommendations: dict[str, RecommendationList],
    anonymous_recommendations: RecommendationList | None = None,
):
    """Format the export payload for EXO. Follows this JSON structure:
    {

    }
    """
    ...  # do the stuff

    return {
        "anonymous_data": anonymous_recommendations.to_list()
        if anonymous_recommendations
        else [],
    }


if __name__ == "__main__":
    from uuid import uuid4

    user_recommendations = {uuid4(): RecommendationList() for _ in range(10)}
    print(user_recommendations)
