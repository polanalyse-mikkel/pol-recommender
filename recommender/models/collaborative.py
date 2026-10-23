import os
from typing import Any, Iterable

import joblib
import numpy as np
import polars as pl
import threadpoolctl
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix

from recommender.data.preprocess import (
    construct_interaction_matrix,
    filter_interactions,
)
from recommender.models.base import RecommenderBase
from recommender.schemas.recommendation import Recommendation
from recommender.schemas.recommendation_list import RecommendationList
from recommender.utils.exceptions import NotFittedError, UserNotFittedError

threadpoolctl.threadpool_limits(1, "blas")


def _save_artifact(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def _load_artifact(path: str) -> Any:
    return joblib.load(path)


class CFRecommender(RecommenderBase):
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int | None = 42,
        dtype=np.float32,
        apply_bm25: bool = True,
        K1: int = 100,
        B: float = 0.8,
    ):
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.dtype = dtype

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            dtype=self.dtype,
        )

        self.apply_bm25 = apply_bm25
        self.K1 = K1
        self.B = B

        self.is_fitted = False
        self.user2idx: dict[Any, int] = {}
        self.users: list[Any] = []
        self.item2idx: dict[Any, int] = {}
        self.items: list[Any] = []
        self.user_items: csr_matrix | None = None

        self._store_init_params(locals())

    def bm25_weight(self, user_items: csr_matrix) -> csr_matrix:
        item_users = user_items.T.tocsr()
        weighted = bm25_weight(item_users, K1=self.K1, B=self.B)
        return weighted.T.tocsr()

    def fit(self, interactions: pl.DataFrame) -> "CFRecommender":
        self.user_items, self.users, self.items, self.user2idx, self.item2idx = (
            construct_interaction_matrix(interactions)
        )

        if self.apply_bm25:
            self.user_items = self.bm25_weight(self.user_items)

        self.model.fit(self.user_items)
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
                "CFRecommender was not fitted. Call .fit(...) before recommending."
            )
        if self.user_items is None:
            raise Exception(
                r"user_items not defined despite already being, idk what is wrong, good luck ¯\_(ツ)_/¯"
            )

        if candidates:
            candidates_idx = np.array(
                [self.item2idx[item] for item in candidates if item in self.item2idx],
                dtype=np.int32,
            )
            if len(candidates_idx) == 0:
                candidates_idx = None
        else:
            candidates_idx = None

        if user_id not in self.user2idx:
            raise UserNotFittedError(
                f"User id '{user_id}' was not present in the training set. "
                "Implement cold-start handling (e.g. fallback to popular items or content-based recommendations) "
                "or add the user's interactions and re-fit the model."
            )

        user_idx = self.user2idx[user_id]
        user_reads = self.user_items[user_idx]
        raw_recs_matrix, raw_scores = self.model.recommend(
            user_idx,
            user_reads,
            N=k,
            filter_already_liked_items=filter_already_liked_items,
            items=candidates_idx,
        )
        recs_matrix = [self.items[rec] for rec in raw_recs_matrix]
        scores = [float(s) for s in raw_scores]

        recommendations: list[Recommendation] = []
        for rank, (item_id, score) in enumerate(zip(recs_matrix, scores), start=1):
            recommendations.append(
                Recommendation(
                    item_id=item_id, score=score, rank=rank, model_id=self.model_id
                )
            )
        return RecommendationList.from_list(recommendations)

    def can_recommend(self, user_id: str) -> bool:
        return user_id in self.user2idx

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "is_fitted": self.is_fitted,
            "users": self.users,
            "items": self.items,
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "user_items": self.user_items,
            "apply_bm25": self.apply_bm25,
            "K1": self.K1,
            "B": self.B,
            "meta": {
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
                "random_state": self.random_state,
                "dtype": str(self.dtype)
                if hasattr(self.dtype, "__name__")
                else str(self.dtype),
            },
        }
        _save_artifact(path, payload)

    @classmethod
    def load(cls, path: str) -> "CFRecommender":
        payload = _load_artifact(path)
        meta = payload.get("meta", {})

        dtype = np.float32

        inst = cls(
            factors=meta.get("factors", 64),
            regularization=meta.get("regularization", 0.01),
            iterations=meta.get("iterations", 15),
            random_state=meta.get("random_state", 42),
            dtype=dtype,
            apply_bm25=payload.get("apply_bm25", True),
            K1=payload.get("K1", 100),
            B=payload.get("B", 0.8),
        )

        inst.model = payload["model"]
        inst.is_fitted = payload.get("is_fitted", True)

        inst.users = payload.get("users", [])
        inst.items = payload.get("items", [])
        inst.user2idx = payload.get("user2idx", {})
        inst.item2idx = payload.get("item2idx", {})
        inst.user_items = payload.get("user_items", None)

        inst.factors = meta.get("factors", inst.factors)
        inst.regularization = meta.get("regularization", inst.regularization)
        inst.iterations = meta.get("iterations", inst.iterations)
        inst.random_state = meta.get("random_state", inst.random_state)
        inst.dtype = dtype
        inst.apply_bm25 = payload.get("apply_bm25", inst.apply_bm25)
        inst.K1 = payload.get("K1", inst.K1)
        inst.B = payload.get("B", inst.B)

        return inst

    @property
    def name(self) -> str:
        return "ALS (Collaborative Filtering)"

    @property
    def category(self) -> str:
        return "personalized"

    def recommend_batch(
        self,
        user_ids: Iterable[Any],
        k: int = 10,
        candidates: Iterable[Any] | None = None,
        candidates_per_user: dict[Any, Iterable[Any]] | None = None,
        filter_already_liked_items: bool = True,
        skip_missing: bool = False,
        **kwargs,
    ) -> dict[Any, RecommendationList]:
        """Return recommendations for a batch of users.

        Args:
            user_ids: User IDs to recommend for.
            k: Number of recommendations per user.
            candidates: Item IDs that restrict the candidate pool (same for all users).
            candidates_per_user: Dict mapping user_id -> item IDs to use as candidates for that user.
            filter_already_liked_items: Whether to filter items the user has already interacted with.
            skip_missing: If True, unknown users get an empty RecommendationList.
        """
        if not self.is_fitted:
            raise NotFittedError(
                "CFRecommender was not fitted. Call .fit(...) before recommending."
            )
        if self.user_items is None:
            raise Exception(
                r"user_items not defined despite already being, idk what is wrong, good luck ¯\_(ツ)_/¯"
            )

        candidates_idx_global = None
        if candidates is not None:
            candidates_idx_global = [
                self.item2idx[it] for it in candidates if it in self.item2idx
            ]

        user_ids_list = list(user_ids)

        missing_users = [u for u in user_ids_list if u not in self.user2idx]
        if missing_users and not skip_missing:
            raise UserNotFittedError(
                f"User id(s) {missing_users} were not present in the training set. "
                "Use skip_missing=True to skip unknown users."
            )

        known_user_map = {
            u: self.user2idx[u] for u in user_ids_list if u in self.user2idx
        }
        if len(known_user_map) == 0:
            return {u: RecommendationList() for u in user_ids_list}

        user_idx_list = np.array(list(known_user_map.values()), dtype=np.int32)
        user_factors = self.model.user_factors[user_idx_list]
        item_factors = self.model.item_factors

        scores = user_factors.dot(item_factors.T).astype(np.float32, copy=False)

        user_recs: dict[Any, RecommendationList] = {}

        for batch_pos, (user_id, user_idx) in enumerate(known_user_map.items()):
            row_scores = scores[batch_pos]

            if candidates_per_user and user_id in candidates_per_user:
                cand_list = [it for it in candidates_per_user[user_id]]
                cand_idx = [
                    self.item2idx[it] for it in cand_list if it in self.item2idx
                ]
            else:
                cand_idx = candidates_idx_global

            if filter_already_liked_items:
                seen = self.user_items[user_idx].indices
                if seen.size:
                    row_scores[seen] = -np.inf

            if cand_idx is not None:
                if len(cand_idx) == 0:
                    topk_idx = np.array([], dtype=np.int32)
                    topk_scores = np.array([], dtype=np.float32)
                else:
                    cand_scores = row_scores[cand_idx]
                    if cand_scores.size <= k:
                        order = np.argsort(-cand_scores)
                    else:
                        part = np.argpartition(-cand_scores, k - 1)[:k]
                        order = part[np.argsort(-cand_scores[part])]
                    topk_idx = np.array([cand_idx[i] for i in order], dtype=np.int32)
                    topk_scores = cand_scores[order]
            else:
                n_items = row_scores.shape[0]
                if n_items <= k:
                    order = np.argsort(-row_scores)
                    topk_idx = order
                    topk_scores = row_scores[order]
                else:
                    part = np.argpartition(-row_scores, k - 1)[:k]
                    order = part[np.argsort(-row_scores[part])]
                    topk_idx = order
                    topk_scores = row_scores[order]

            rec_items = [self.items[i] for i in topk_idx]
            rec_scores = [float(s) for s in topk_scores]

            rec_list: list[Recommendation] = []
            for rank, (item_id, score) in enumerate(
                zip(rec_items, rec_scores), start=1
            ):
                rec_list.append(
                    Recommendation(
                        item_id=item_id, score=score, rank=rank, model_id=self.model_id
                    )
                )

            user_recs[user_id] = RecommendationList.from_list(rec_list)

        if skip_missing:
            for u in user_ids_list:
                if u not in user_recs:
                    user_recs[u] = RecommendationList()

        return user_recs


class CFRecommenderFiltered(CFRecommender):
    def __init__(
        self,
        min_user_pv: int | None = None,
        max_user_pv: int | None = None,
        min_item_pv: int | None = None,
        max_item_pv: int | None = None,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int | None = 42,
        dtype=np.float32,
        apply_bm25: bool = True,
        K1: int = 100,
        B: float = 0.8,
    ):
        super().__init__(
            factors, regularization, iterations, random_state, dtype, apply_bm25, K1, B
        )
        self.min_user_pv = min_user_pv
        self.max_user_pv = max_user_pv
        self.min_item_pv = min_item_pv
        self.max_item_pv = max_item_pv

        self._store_init_params(locals())

    def fit(self, interactions: pl.DataFrame) -> "CFRecommenderFiltered":
        filtered_interactions = filter_interactions(
            interactions,
            min_user_pv=self.min_user_pv,
            max_user_pv=self.max_user_pv,
            min_item_pv=self.min_item_pv,
            max_item_pv=self.max_item_pv,
        )

        self.user_items, self.users, self.items, self.user2idx, self.item2idx = (
            construct_interaction_matrix(filtered_interactions)
        )

        if self.apply_bm25:
            self.user_items = self.bm25_weight(self.user_items)

        self.model.fit(self.user_items)
        self.is_fitted = True
        return self
