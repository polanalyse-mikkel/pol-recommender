from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from recommender.schemas.recommendation_list import RecommendationList


def format_export_payload(user_recommendations: dict[str, RecommendationList] = {}):
    payload = {}
    for user_id, rec_list in user_recommendations.items():
        items = rec_list.map(lambda item: item.item_id)
        if len(items) > 0:
            payload[user_id] = items
    return payload


class ExportAction(str, Enum):
    flush = "flush"
    update = "update"


class ConfigPayload(BaseModel):
    action: ExportAction


class ModelBlock(BaseModel):
    model_name: str
    # [{sso_1: [art1, art2, ...]}, {sso_2: [art1, ...]}]
    art_sub_map: list[dict[str, list[int]]]


class AnonymousData(BaseModel):
    articles_list: list[int]


class ExoExportPayload(BaseModel):
    config: ConfigPayload
    subscriber_data: list[ModelBlock]
    registered_data: list[ModelBlock]
    anonymous_data: AnonymousData


def _build_model_blocks(
    user_recommendations: dict[str, RecommendationList],
    model_names: dict[UUID, str],
) -> list[ModelBlock]:
    """Group per-user recommendations into one block per model.

    Users with empty lists (e.g. control group) are omitted. Article order
    within a user follows the list order (already sorted by top_k) and
    duplicates are dropped.
    """
    # model_id -> sso_id -> [item_ids]; dicts preserve insertion order
    by_model: dict[UUID, dict[str, list[int]]] = defaultdict(dict)

    for sso_id, rec_list in user_recommendations.items():
        for rec in rec_list:
            items = by_model[rec.model_id].setdefault(sso_id, [])
            if rec.item_id not in items:
                items.append(rec.item_id)

    return [
        ModelBlock(
            model_name=model_names.get(model_id, str(model_id)),
            art_sub_map=[{sso_id: items} for sso_id, items in users.items()],
        )
        for model_id, users in by_model.items()
    ]


def format_exo_export_payload(
    subscriber_recommendations: dict[str, RecommendationList] | None = None,
    registered_recommendations: dict[str, RecommendationList] | None = None,
    anonymous_recommendations: RecommendationList | None = None,
    *,
    model_names: dict[UUID, str] | None = None,
    action: ExportAction = ExportAction.flush,
    shuffle_anonymous: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Format the export payload object.

    Args:
        subscriber_recommendations: sso_id -> recommendations for subscribers.
        registered_recommendations: sso_id -> recommendations for registered users.
        anonymous_recommendations: Global list shown to anonymous users.
        model_names: model_id -> human-readable model name.
        action: "flush" or "update".
        shuffle_anonymous: Randomize anonymous order instead of sorting by score.
        seed: Seed for the anonymous shuffle.
    """
    model_names = model_names or {}

    anon_items: list[int] = []
    if anonymous_recommendations:
        recs = anonymous_recommendations.dedupe()
        recs = (
            recs.shuffle(seed=seed)
            if shuffle_anonymous
            else recs.sort_by_score(descending=True)
        )
        anon_items = recs.map(lambda r: r.item_id)

    payload = ExoExportPayload(
        config=ConfigPayload(action=action),
        subscriber_data=_build_model_blocks(
            subscriber_recommendations or {}, model_names
        ),
        registered_data=_build_model_blocks(
            registered_recommendations or {}, model_names
        ),
        anonymous_data=AnonymousData(articles_list=anon_items),
    )
    return payload.model_dump()


if __name__ == "__main__":
    from uuid import uuid4

    import orjson

    from recommender.schemas.recommendation import Recommendation

    model_id = uuid4()
    names = {model_id: "reco_model_abc"}

    def make_list(item_ids: list[int]) -> RecommendationList:
        return RecommendationList(
            recommendations=[
                Recommendation(
                    item_id=i, score=1.0 / (n + 1), rank=n + 1, model_id=model_id
                )
                for n, i in enumerate(item_ids)
            ]
        )

    registered = {
        "sso_1": make_list([1, 2, 3]),
        "sso_2": make_list([4, 5]),
        "sso_3": RecommendationList(),
    }
    subscribers = {"sso_4": make_list([6, 7])}
    anonymous = make_list([10, 11, 12])

    print(
        format_exo_export_payload(
            subscriber_recommendations=subscribers,
            registered_recommendations=registered,
            anonymous_recommendations=anonymous,
            model_names=names,
        ),
    )
    print(
        orjson.dumps(
            format_exo_export_payload(
                subscriber_recommendations=subscribers,
                registered_recommendations=registered,
                anonymous_recommendations=anonymous,
                model_names=names,
            )
        ).decode()
    )
