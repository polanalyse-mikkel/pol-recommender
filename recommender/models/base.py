import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import orjson
import polars as pl

from recommender.schemas.recommendation_collection import RecommendationCollection
from recommender.schemas.recommendation_list import RecommendationList


class RecommenderBase(ABC):
    def __init__(self):
        self.model_id = uuid.uuid4()
        self.expire_at: datetime = datetime.now() + timedelta(days=1)
        self._init_params: dict[str, Any] = {}

    @abstractmethod
    def fit(self, interactions: pl.DataFrame) -> "RecommenderBase":
        raise NotImplementedError()

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidates: list[Any] | None = None,
        filter_already_liked_items: bool = True,
    ) -> RecommendationList:
        raise NotImplementedError()

    @abstractmethod
    def can_recommend(self, user_id: str) -> bool:
        raise NotImplementedError()

    def recommend_batch(
        self, user_ids: list[str], k: int = 10, **kwargs
    ) -> dict[str, RecommendationList]:
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, k=k, **kwargs)
        return results

    def recommend_batch_as_collection(
        self, user_ids: list[str], k: int = 10, **kwargs
    ) -> RecommendationCollection:
        """Recommend for multiple users and return as a RecommendationCollection."""
        batch_result = self.recommend_batch(user_ids, k=k, **kwargs)
        return RecommendationCollection.from_batch_result(
            batch_result,
            model_id=self.model_id,
            model_name=self.name,
            metadata={"k": k, **kwargs},
        )

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "RecommenderBase":
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def category(self) -> str:
        return "no category"

    def fit_timed(self, *args, **kwargs) -> "RecommenderBase":
        start_time = time.time()
        out = self.fit(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{self.name} fit method executed in {elapsed_time:.4f} seconds")
        return out

    def recommend_timed(self, *args, **kwargs) -> RecommendationList:
        start_time = time.time()
        out = self.recommend(*args, **kwargs)
        self._recommend_time = getattr(self, "_recommend_time", 0.0) + (
            time.time() - start_time
        )
        return out

    @property
    def recommend_time(self) -> float:
        return getattr(self, "_recommend_time", 0.0)

    def _store_init_params(self, local_vars: dict[str, Any]) -> None:
        """Store constructor parameters for hashing.

        Call this at the end of __init__ with locals().
        This captures all constructor parameters.
        """
        # Exclude 'self' and any special variables
        params = {k: v for k, v in local_vars.items() if k not in ("self", "__class__")}

        # Convert non-serializable types to strings
        serialized_params = {}
        for key, value in params.items():
            if hasattr(value, "__name__"):
                serialized_params[key] = value.__name__
            elif type(value).__module__ == "numpy":
                serialized_params[key] = str(value)
            else:
                serialized_params[key] = value

        self._init_params = serialized_params

    @property
    def model_hash(self) -> str:
        """Generate a deterministic hash from class name and constructor parameters.

        This hash identifies models that are functionally equivalent.
        Subclasses must call self._store_init_params(locals()) at the end of __init__.
        """
        # Create hashable representation
        hash_data = {
            "class_name": self.__class__.__name__,
            "params": self._init_params,
        }

        # Serialize to JSON for consistent hashing
        serialized = orjson.dumps(hash_data, option=orjson.OPT_SORT_KEYS)

        # Generate SHA256 hash
        return hashlib.sha256(serialized).hexdigest()
