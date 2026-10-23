from __future__ import annotations

import uuid
from typing import Any

import polars as pl
from pydantic import BaseModel, Field, field_validator

from recommender.schemas.recommendation_list import RecommendationList
from recommender.utils.helpers import coerce_uuid


class RecommendationCollection(BaseModel):
    """Collection of recommendation lists for a single model.

    Attributes:
        recommendation_lists: Dict mapping user_id (str) -> RecommendationList.
        model_id: Unique identifier for the model instance.
        model_name: Name of the model/algorithm.
        metadata: Additional metadata about the collection.
    """

    model_config = {"validate_assignment": True}

    recommendation_lists: dict[str, RecommendationList] = Field(default_factory=dict)
    model_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    model_name: str | None = None
    model_category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_id", mode="before")
    @classmethod
    def coerce_model_id(cls, v: uuid.UUID | str | None) -> uuid.UUID:
        return coerce_uuid(v)

    def __len__(self) -> int:
        return len(self.recommendation_lists)

    def __iter__(self):  # type: ignore[override]
        return iter(self.recommendation_lists.items())

    def __getitem__(self, user_id: str) -> RecommendationList:
        return self.recommendation_lists[user_id]

    def __contains__(self, user_id: str) -> bool:
        return user_id in self.recommendation_lists

    def __repr__(self) -> str:
        return (
            f"RecommendationCollection(model_id={str(self.model_id)[:8]}..., "
            f"model_name={self.model_name!r}, users={len(self)})"
        )

    def add(self, user_id: str, rec_list: RecommendationList) -> None:
        """Add a recommendation list for a user."""
        self.recommendation_lists[user_id] = rec_list

    def get(
        self, user_id: str, default: RecommendationList | None = None
    ) -> RecommendationList | None:
        """Get recommendation list for a user."""
        return self.recommendation_lists.get(user_id, default)

    def user_ids(self) -> list[str]:
        """Get all user IDs in the collection."""
        return list(self.recommendation_lists.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict representation."""
        return {
            "model_id": str(self.model_id),
            "model_name": self.model_name,
            "metadata": self.metadata,
            "recommendation_lists": {
                str(user_id): rec_list.model_dump()
                for user_id, rec_list in self.recommendation_lists.items()
            },
        }

    def to_polars(self) -> pl.DataFrame:
        """Convert all recommendation lists to a single Polars DataFrame."""
        dfs = []
        for user_id, rec_list in self.recommendation_lists.items():
            df = rec_list.to_polars()
            df = df.with_columns(
                pl.lit(str(user_id)).alias("user_id"),
                pl.lit(str(self.model_id)).alias("model_id"),
                pl.lit(self.model_name).alias("model_name"),
            )
            dfs.append(df)

        if not dfs:
            return pl.DataFrame(
                {
                    "user_id": pl.Series([], dtype=pl.Utf8),
                    "model_id": pl.Series([], dtype=pl.Utf8),
                    "model_name": pl.Series([], dtype=pl.Utf8),
                    "item_id": pl.Series([], dtype=pl.Utf8),
                    "score": pl.Series([], dtype=pl.Float64),
                    "rank": pl.Series([], dtype=pl.Float64),
                    "model": pl.Series([], dtype=pl.Utf8),
                    "metadata": pl.Series([], dtype=pl.Object),
                    "recommendation_id": pl.Series([], dtype=pl.Utf8),
                    "recommendation_list_id": pl.Series([], dtype=pl.Utf8),
                }
            )

        return pl.concat(dfs)

    def filter_users(self, user_ids: list[str]) -> RecommendationCollection:
        """Return a new collection with only specified users."""
        filtered_lists = {
            uid: rec_list
            for uid, rec_list in self.recommendation_lists.items()
            if uid in user_ids
        }
        return RecommendationCollection(
            recommendation_lists=filtered_lists,
            model_id=self.model_id,
            model_name=self.model_name,
            metadata=dict(self.metadata),
        )

    def total_recommendations(self) -> int:
        """Get total number of recommendations across all users."""
        return sum(len(rec_list) for rec_list in self.recommendation_lists.values())

    @classmethod
    def from_batch_result(
        cls,
        batch_result: dict[str, RecommendationList],
        model_id: uuid.UUID | str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecommendationCollection:
        """Create from a batch recommendation result."""
        return cls(
            recommendation_lists=batch_result,
            model_id=model_id,  # type: ignore[arg-type]
            model_name=model_name,
            metadata=metadata or {},
        )
