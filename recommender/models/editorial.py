from typing import Any

import orjson
import polars as pl
from psycopg.rows import dict_row

import recommender.pol_connector as pc
from recommender.models.base import RecommenderBase
from recommender.schemas.recommendation import Recommendation
from recommender.schemas.recommendation_list import RecommendationList
from recommender.utils.exceptions import NotFittedError, UserNotFittedError
from recommender.utils.helpers import coerce_uuid


class EditorialRecommender(RecommenderBase):
    def __init__(self, editorial_id: str) -> None:
        super().__init__()
        self.editorial_id = coerce_uuid(editorial_id)
        self.user_items: dict[Any, set] = {}
        self._name: str = self.__class__.__name__
        self.users: set[str] = set()
        self.items: set[int] = set()
        self.is_fitted = False

        self._store_init_params(locals())

    def fit(self, interactions: pl.DataFrame) -> "EditorialRecommender":
        query = """
            SELECT
                editorial_id, editorial_name,
                users, articles, expire_at
            FROM politiken_ia_prv.recommendation_editorials
            WHERE editorial_status in ('pending', 'active')
              AND editorial_id = %s
        """
        with pc.connect(row_factory=dict_row) as conn:
            row = conn.execute(query, [str(self.editorial_id)]).fetchone()

        if not row:
            raise ValueError(
                "No pending editorial recommendations found in the database."
            )

        self._name = row["editorial_name"]
        self.users = set(orjson.loads(row["users"])) if row["users"] else set()
        self.items = set(int(x) for x in orjson.loads(row["articles"]))
        self.expire_at = row["expire_at"]

        self.user_items = {
            user_id: set(items)
            for user_id, items in interactions.group_by("user_id")
            .agg(pl.col("item_id"))
            .iter_rows()
        }

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidates: list[Any] | None = None,
        filter_already_liked_items: bool = True,
    ) -> RecommendationList:
        if not self.is_fitted:
            raise NotFittedError(
                f"{self.name} was not fitted. Call .fit(...) before recommending."
            )
        if self.user_items is None:
            raise Exception(
                r"user_items not defined despite already being, idk what is wrong, good luck ¯\_(ツ)_/¯"
            )
        if user_id not in self.users:
            raise UserNotFittedError(
                f"User id '{user_id}' was not present in the editorial recommendation list."
            )

        interacted_items = self.user_items.get(user_id, set())
        recommended_items = [
            Recommendation(item_id=item, model_id=self.model_id, rank=rank)
            for rank, item in enumerate(self.items, start=1)
            if (filter_already_liked_items and item not in interacted_items)
            or not filter_already_liked_items
        ]
        recommended_items = recommended_items[:k]
        return RecommendationList(recommendations=recommended_items)

    def can_recommend(self, user_id: str) -> bool:
        return user_id in self.users

    def save(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str) -> "RecommenderBase":
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return "editorial"
