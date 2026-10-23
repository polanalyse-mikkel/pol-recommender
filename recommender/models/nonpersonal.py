"""Non-personalized recommenders (popularity-based, trending, etc.)."""

import os
from typing import Any

import joblib
import polars as pl

from recommender.schemas.recommendation import Recommendation
from recommender.schemas.recommendation_list import RecommendationList

from .base import RecommenderBase


def _save_artifact(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def _load_artifact(path: str) -> Any:
    return joblib.load(path)


class PopularityRecommender(RecommenderBase):
    popular_items: list[Any]
    popular_scores: dict[Any, float]
    user_items: dict[Any, set]

    def __init__(self) -> None:
        super().__init__()
        self.popular_items = []
        self.popular_scores = {}
        self.user_items = {}
        
        self._store_init_params(locals())

    def fit(self, interactions: pl.DataFrame) -> "PopularityRecommender":
        item_counts = (
            interactions.group_by("item_id")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        self.popular_items = item_counts["item_id"].to_list()
        self.popular_scores = dict(item_counts.iter_rows())
        # Precompute rank index for O(1) candidate filtering
        self._item_rank: dict[Any, int] = {
            item: i for i, item in enumerate(self.popular_items)
        }
        self.user_items = {
            user_id: set(items)
            for user_id, items in interactions.group_by("user_id")
            .agg(pl.col("item_id"))
            .iter_rows()
        }
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidates: list[Any] | None = None,
        filter_already_liked_items: bool = True,
    ) -> RecommendationList:
        already_liked = (
            self.user_items.get(user_id, set()) if filter_already_liked_items else set()
        )

        if candidates is not None:
            pool = sorted(
                candidates,
                key=lambda x: self._item_rank.get(x, len(self.popular_items)),
            )
        else:
            pool = self.popular_items

        out = []
        for item in pool:
            if item not in already_liked:
                out.append(
                    Recommendation(
                        item_id=item,
                        score=self.popular_scores[item],
                        rank=len(out) + 1,
                        model_id=self.model_id,
                        metadata={},
                    )
                )
                if len(out) == k:
                    break
        return RecommendationList.from_list(out)

    def can_recommend(self, user_id: str) -> bool:
        return True

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "popular_items": self.popular_items,
                "popular_scores": self.popular_scores,
                "user_items": self.user_items,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PopularityRecommender":
        data = joblib.load(path)
        inst = cls()
        inst.popular_items = data.get("popular_items", [])
        inst.popular_scores = data.get("popular_scores", {})
        inst.user_items = data.get("user_items", {})
        return inst

    @property
    def name(self) -> str:
        return "Popularity"

    @property
    def category(self) -> str:
        return "popularity"


# class RandomRecommender(RecommenderBase):
#     """Return random items from the corpus. Useful as a lower-bound baseline."""

#     def __init__(self, seed: Optional[int] = None) -> None:
#         self.seed = seed
#         self.items: List[Any] = []

#     def fit(self, interactions: pl.DataFrame) -> "RandomRecommender":
#         self.items = interactions.select("item_id").unique().to_series().to_list()
#         return self

#     def recommend(self, user_id: Any, k: int = 10) -> RecommendationList:
#         if not self.items:
#             return RecommendationList()
#         rng = np.random.default_rng(self.seed)
#         chosen: List[Any]
#         if k >= len(self.items):
#             # shuffle and return all
#             chosen = list(rng.permutation(self.items))
#         else:
#             chosen = list(rng.choice(self.items, size=k, replace=False).tolist())

#         out: List[Recommendation] = []
#         for rank, item in enumerate(chosen, start=1):
#             # Random baseline: score is None (or uniform probability). We'll leave it None.
#             out.append(
#                 Recommendation(
#                     item_id=item,
#                     score=None,
#                     rank=rank,
#                     model=self.name,
#                     metadata={},
#                 )
#             )
#         return RecommendationList(out)

#     def save(self, path: str) -> None:
#         _save_artifact(path, {"items": self.items, "seed": self.seed})

#     @classmethod
#     def load(cls, path: str) -> "RandomRecommender":
#         data = _load_artifact(path)
#         inst = cls(seed=data.get("seed"))
#         inst.items = data.get("items", [])
#         return inst

#     @property
#     def name(self) -> str:
#         return "Random"

#     @property
#     def category(self) -> str:
#         return "popularity"


# class TrendingRecommender(RecommenderBase):
#     """
#     Items ranked by their activity within a recent time window (e.g. last N days).
#     Requires the interactions DataFrame to include a timestamp column.
#     Falls back to global popularity if timestamps are absent.
#     """

#     def __init__(self, window_days: int = 7) -> None:
#         self.window_days = window_days
#         self.ranked_items: List[Any] = []
#         self.ranked_scores: Dict[Any, float] = {}

#     def fit(self, interactions: pl.DataFrame) -> "TrendingRecommender":
#         ts_col = "timestamp"
#         if ts_col is None:
#             # fallback: global popularity
#             counts = Counter(interactions["item_id"].to_list())
#             self.ranked_items = [i for i, _ in counts.most_common()]
#             self.ranked_scores = {i: float(c) for i, c in counts.items()}
#             return self

#         # normalize timestamp to datetime if needed
#         df = interactions.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("_ts"))

#         max_ts = df.select(pl.col("_ts").max()).item()
#         if max_ts is None:
#             self.ranked_items = []
#             self.ranked_scores = {}
#             return self

#         cutoff = max_ts - datetime.timedelta(days=self.window_days)
#         recent = df.filter(pl.col("_ts") >= cutoff)
#         counts = recent.group_by("item_id").agg(pl.count("item_id").alias("c"))
#         counts = counts.sort("c", descending=True)
#         # convert to lists
#         try:
#             items_list = counts["item_id"].to_list()
#             scores_list = counts["c"].to_list()
#         except Exception:
#             # fallback if column access differs
#             items_list = counts.select("item_id").to_series().to_list()
#             scores_list = counts.select("c").to_series().to_list()
#         self.ranked_items = items_list
#         self.ranked_scores = {
#             item: float(score) for item, score in zip(items_list, scores_list)
#         }
#         return self

#     def recommend(self, user_id: Any, k: int = 10) -> RecommendationList:
#         out: List[Recommendation] = []
#         for rank, item in enumerate(self.ranked_items[:k], start=1):
#             score = self.ranked_scores.get(item, None)
#             out.append(
#                 Recommendation(
#                     item_id=item,
#                     score=score,
#                     rank=rank,
#                     model=self.name,
#                     metadata={},
#                 )
#             )
#         return RecommendationList(out)

#     def save(self, path: str) -> None:
#         _save_artifact(
#             path,
#             {
#                 "ranked_items": self.ranked_items,
#                 "ranked_scores": self.ranked_scores,
#                 "window_days": self.window_days,
#             },
#         )

#     @classmethod
#     def load(cls, path: str) -> "TrendingRecommender":
#         data = _load_artifact(path)
#         inst = cls(window_days=data.get("window_days", 7))
#         inst.ranked_items = data.get("ranked_items", [])
#         inst.ranked_scores = data.get("ranked_scores", {})
#         return inst

#     @property
#     def name(self) -> str:
#         return f"Trending (last {self.window_days}d)"

#     @property
#     def category(self) -> str:
#         return "popularity"


# class TimeDecayPopularityRecommender(RecommenderBase):
#     """
#     Popularity with exponential time decay.
#     Score(item) = sum_{events} exp(-ln(2) * age_days / half_life_days)
#     Prefer recent activity but still account for older events.
#     """

#     def __init__(self, half_life_days: float = 7.0) -> None:
#         self.half_life_days = half_life_days
#         self.scored_items: List[Any] = []
#         self.scored_scores: Dict[Any, float] = {}

#     def fit(self, interactions: pl.DataFrame) -> "TimeDecayPopularityRecommender":
#         ts_col = "timestamp"
#         if ts_col not in interactions.columns:
#             # fallback to simple popularity
#             counts = Counter(interactions["item_id"].to_list())
#             self.scored_items = [i for i, _ in counts.most_common()]
#             self.scored_scores = {i: float(c) for i, c in counts.items()}
#             return self

#         df = interactions.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("_ts"))

#         # get max timestamp robustly (polars -> pandas/pyt datetime)
#         max_ts_raw = df.select(pl.col("_ts").max()).to_series()[0]
#         if max_ts_raw is None:
#             self.scored_items = []
#             self.scored_scores = {}
#             return self

#         # convert max timestamp to numpy datetime64 (ns)
#         max_ts_np = np.datetime64(max_ts_raw, "ns")

#         # get events as pandas DataFrame and ensure numpy datetime64[ns] for _ts
#         events = df.select(["item_id", "_ts"]).to_pandas()
#         ts_arr = events["_ts"].to_numpy(dtype="datetime64[ns]")

#         # compute age in days (float)
#         age_days = (max_ts_np - ts_arr).astype("timedelta64[D]").astype(float)
#         age_days = np.clip(age_days, 0.0, None)

#         lam = np.log(2) / float(max(self.half_life_days, 1e-9))
#         scores = np.exp(-lam * age_days)

#         agg: Dict[Any, float] = {}
#         for item, s in zip(events["item_id"], scores):
#             agg[item] = agg.get(item, 0.0) + float(s)

#         # sort by score desc
#         sorted_items = [
#             item for item, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
#         ]
#         self.scored_items = sorted_items
#         self.scored_scores = {item: float(score) for item, score in agg.items()}
#         return self

#     def recommend(self, user_id: Any, k: int = 10) -> RecommendationList:
#         out: List[Recommendation] = []
#         for rank, item in enumerate(self.scored_items[:k], start=1):
#             score = self.scored_scores.get(item, None)
#             out.append(
#                 Recommendation(
#                     item_id=item,
#                     score=score,
#                     rank=rank,
#                     model=self.name,
#                     metadata={},
#                 )
#             )
#         return RecommendationList(out)

#     def save(self, path: str) -> None:
#         _save_artifact(
#             path,
#             {
#                 "scored_items": self.scored_items,
#                 "scored_scores": self.scored_scores,
#                 "half_life_days": self.half_life_days,
#             },
#         )

#     @classmethod
#     def load(cls, path: str) -> "TimeDecayPopularityRecommender":
#         data = _load_artifact(path)
#         inst = cls(half_life_days=data.get("half_life_days", 7.0))
#         inst.scored_items = data.get("scored_items", [])
#         inst.scored_scores = data.get("scored_scores", {})
#         return inst

#     @property
#     def name(self) -> str:
#         return f"TimeDecayPopularity(half_life={self.half_life_days}d)"

#     @property
#     def category(self) -> str:
#         return "popularity"


# class CategoryTopKRecommender(RecommenderBase):
#     """
#     Top-N within each category. Interactions must include a category column named
#     either 'item_category' or 'category'. If missing, falls back to global popularity.

#     `recommend(user_id, k)` returns top-k global items by default; to request
#     top items for a specific category, call recommend_for_category(category, k).
#     """

#     def __init__(self) -> None:
#         self.top_per_category: Dict[Any, List[Any]] = {}
#         self.global_popular: List[Any] = []
#         self.global_scores: Dict[Any, float] = {}

#     def fit(self, interactions: pl.DataFrame) -> "CategoryTopKRecommender":
#         cat_col = None
#         for c in ("item_category", "category", "cat"):
#             if c in interactions.columns:
#                 cat_col = c
#                 break

#         if cat_col is None:
#             counts = Counter(interactions["item_id"].to_list())
#             self.global_popular = [i for i, _ in counts.most_common()]
#             self.global_scores = {i: float(c) for i, c in counts.items()}
#             self.top_per_category = {}
#             return self

#         df = interactions.group_by([cat_col, "item_id"]).agg(
#             pl.count("item_id").alias("c")
#         )
#         # collect per category ordered list
#         # We'll iterate sorted by category then by count desc
#         df_sorted = df.sort([cat_col, pl.col("c").reverse()])
#         # build mapping
#         for row in df_sorted.iter_rows():
#             # row: (category, item_id, c)
#             cat = row[0]
#             item = row[1]
#             self.top_per_category.setdefault(cat, []).append(item)

#         # also compute global popularity fallback
#         counts = Counter(interactions["item_id"].to_list())
#         self.global_popular = [i for i, _ in counts.most_common()]
#         self.global_scores = {i: float(c) for i, c in counts.items()}
#         return self

#     def recommend(self, user_id: Any, k: int = 10) -> RecommendationList:
#         # default: top global
#         out: List[Recommendation] = []
#         for rank, item in enumerate(self.global_popular[:k], start=1):
#             score = self.global_scores.get(item, None)
#             out.append(
#                 Recommendation(
#                     item_id=item,
#                     score=score,
#                     rank=rank,
#                     model=self.name,
#                     metadata={},
#                 )
#             )
#         return RecommendationList(out)

#     def recommend_for_category(self, category: Any, k: int = 10) -> RecommendationList:
#         items = self.top_per_category.get(category, self.global_popular)[:k]
#         out: List[Recommendation] = []
#         for rank, item in enumerate(items, start=1):
#             score = self.global_scores.get(item, None)
#             out.append(
#                 Recommendation(
#                     item_id=item,
#                     score=score,
#                     rank=rank,
#                     model=self.name,
#                     metadata={"category": category},
#                 )
#             )
#         return RecommendationList(out)

#     def save(self, path: str) -> None:
#         _save_artifact(
#             path,
#             {
#                 "top_per_category": self.top_per_category,
#                 "global_popular": self.global_popular,
#                 "global_scores": self.global_scores,
#             },
#         )

#     @classmethod
#     def load(cls, path: str) -> "CategoryTopKRecommender":
#         data = _load_artifact(path)
#         inst = cls()
#         inst.top_per_category = data.get("top_per_category", {})
#         inst.global_popular = data.get("global_popular", [])
#         inst.global_scores = data.get("global_scores", {})
#         return inst

#     @property
#     def name(self) -> str:
#         return "CategoryTopK"

#     @property
#     def category(self) -> str:
#         return "popularity"


# class FreshnessRecommender(RecommenderBase):
#     """
#     Recommend newest items based on the most recent event timestamp per item.
#     Requires a timestamp column; falls back to popularity if missing.
#     """

#     def __init__(self) -> None:
#         self.newest_items: List[Any] = []
#         self.newest_timestamps: Dict[Any, Any] = {}

#     def fit(self, interactions: pl.DataFrame) -> "FreshnessRecommender":
#         ts_col = "timestamp"
#         if ts_col is None:
#             counts = Counter(interactions["item_id"].to_list())
#             self.newest_items = [i for i, _ in counts.most_common()]
#             self.newest_timestamps = {}
#             return self

#         df = interactions.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("_ts"))
#         last_seen = df.group_by("item_id").agg(pl.col("_ts").max().alias("last_ts"))
#         last_seen = last_seen.sort("last_ts", descending=True)
#         try:
#             items = last_seen["item_id"].to_list()
#             timestamps = last_seen["last_ts"].to_list()
#         except Exception:
#             items = last_seen.select("item_id").to_series().to_list()
#             timestamps = last_seen.select("last_ts").to_series().to_list()
#         self.newest_items = items
#         # store timestamps as ISO strings for easy serialization
#         self.newest_timestamps = {
#             item: (ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
#             for item, ts in zip(items, timestamps)
#         }
#         return self

#     def recommend(self, user_id: Any, k: int = 10) -> RecommendationList:
#         out: List[Recommendation] = []
#         for rank, item in enumerate(self.newest_items[:k], start=1):
#             ts = self.newest_timestamps.get(item, None)
#             out.append(
#                 Recommendation(
#                     item_id=item,
#                     score=None,
#                     rank=rank,
#                     model=self.name,
#                     metadata={"last_seen": ts},
#                 )
#             )
#         return RecommendationList(out)

#     def save(self, path: str) -> None:
#         _save_artifact(
#             path,
#             {
#                 "newest_items": self.newest_items,
#                 "newest_timestamps": self.newest_timestamps,
#             },
#         )

#     @classmethod
#     def load(cls, path: str) -> "FreshnessRecommender":
#         data = _load_artifact(path)
#         inst = cls()
#         inst.newest_items = data.get("newest_items", [])
#         inst.newest_timestamps = data.get("newest_timestamps", {})
#         return inst

#     @property
#     def name(self) -> str:
#         return "Freshness"

#     @property
#     def category(self) -> str:
#         return "popularity"


# class NonPersonalizedRecommenderRoulette(RecommenderBase):
#     def __init__(self) -> None:
#         self.models = [
#             PopularityRecommender(),
#             RandomRecommender(),
#             TrendingRecommender(),
#             TimeDecayPopularityRecommender(),
#             FreshnessRecommender(),
#         ]
#         self.active_model: Optional[RecommenderBase] = None

#     def fit(self, interactions: pl.DataFrame) -> "NonPersonalizedRecommenderRoulette":
#         for model in self.models:
#             model.fit(interactions)
#         return self

#     def recommend(self, user_id: Any, k: int = 10) -> RecommendationList:
#         self.active_model = random.choice(self.models)
#         if self.active_model:
#             recs = self.active_model.recommend(user_id, k=k)
#             return recs
#         raise Exception("Something went wrong")

#     def save(self, path: str) -> None:
#         """Save model to disk"""
#         raise NotImplementedError()

#     @classmethod
#     def load(cls, path: str) -> "RecommenderBase":
#         """Load model from disk"""
#         raise NotImplementedError()

#     @property
#     def name(self) -> str:
#         return "Roulette"

#     @property
#     def category(self) -> str:
#         return "popularity"
