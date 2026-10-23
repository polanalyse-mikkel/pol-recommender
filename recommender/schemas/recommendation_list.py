from __future__ import annotations

import math
import random
import uuid
from typing import Any, Callable

import polars as pl
from pydantic import BaseModel, Field, field_validator

from recommender.schemas.recommendation import Recommendation
from recommender.utils.helpers import coerce_uuid


class RecommendationList(BaseModel):
    """Container for recommendations with utility methods.

    Attributes:
        recommendations: List of Recommendation objects.
        recommendation_list_id: Unique identifier.
    """

    model_config = {"validate_assignment": True}

    recommendations: list[Recommendation] = Field(default_factory=list)
    recommendation_list_id: uuid.UUID = Field(default_factory=uuid.uuid4)

    @field_validator("recommendation_list_id", mode="before")
    @classmethod
    def coerce_list_id(cls, v: uuid.UUID | str | None) -> uuid.UUID:
        return coerce_uuid(v)

    def __len__(self) -> int:
        return len(self.recommendations)

    def __iter__(self):  # type: ignore[override]
        return iter(self.recommendations)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return RecommendationList(
                recommendations=self.recommendations[idx],
                recommendation_list_id=self.recommendation_list_id,
            )
        return self.recommendations[idx]

    def __repr__(self) -> str:
        top_items = [r.item_id for r in self.recommendations[:5]]
        return f"RecommendationList(id={str(self.recommendation_list_id)}, len={len(self)}, top_items={top_items})"

    @classmethod
    def from_list(
        cls,
        recs: list[Recommendation],
        recommendation_list_id: uuid.UUID | str | None = None,
    ) -> RecommendationList:
        return cls(
            recommendations=list(recs),
            recommendation_list_id=recommendation_list_id,  # type: ignore[arg-type]
        )

    @classmethod
    def from_dicts(
        cls,
        dicts: list[dict[str, Any]],
        recommendation_list_id: uuid.UUID | str | None = None,
    ) -> RecommendationList:
        items = [Recommendation(**d) for d in dicts]
        return cls(
            recommendations=items,
            recommendation_list_id=recommendation_list_id,  # type: ignore[arg-type]
        )

    def to_list(self) -> list[Recommendation]:
        return list(self.recommendations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_list_id": str(self.recommendation_list_id),
            "recommendations": [r.model_dump() for r in self.recommendations],
        }

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.model_dump() for r in self.recommendations]

    def to_json(self, **json_kwargs) -> str:
        return self.model_dump_json(**json_kwargs)

    def to_polars(self) -> pl.DataFrame:
        dicts = self.to_dicts()
        if not dicts:
            return pl.DataFrame(
                {
                    "item_id": pl.Series([], dtype=pl.Utf8),
                    "score": pl.Series([], dtype=pl.Float64),
                    "rank": pl.Series([], dtype=pl.Float64),
                    "model": pl.Series([], dtype=pl.Utf8),
                    "metadata": pl.Series([], dtype=pl.Object),
                    "recommendation_list_id": pl.Series([], dtype=pl.Utf8),
                }
            )
        df = pl.DataFrame(dicts)
        if "recommendation_list_id" not in df.columns:
            df = df.with_columns(
                pl.lit(str(self.recommendation_list_id)).alias("recommendation_list_id")
            )
        return df

    def top_k(self, k: int) -> RecommendationList:
        return self.sort_by_score(descending=True)[:k]

    def sort_by_score(self, descending: bool = True) -> RecommendationList:
        def key_fn(rec: Recommendation):
            s = rec.score
            if s is None or (isinstance(s, float) and not math.isfinite(s)):
                return float("-inf") if descending else float("inf")
            return float(s)

        sorted_recs = sorted(self.recommendations, key=key_fn, reverse=descending)
        return RecommendationList(
            recommendations=sorted_recs,
            recommendation_list_id=self.recommendation_list_id,
        )

    def shuffle(
        self, seed: int | None = None, reset_rank: bool = True
    ) -> RecommendationList:
        """Return recommendations in random order.

        Args:
            seed: Random seed for reproducibility.
            reset_rank: Update ranks to reflect new positions.
        """
        rng = random.Random(seed)
        shuffled = list(self.recommendations)
        rng.shuffle(shuffled)

        if reset_rank:
            shuffled = [
                rec.copy_with(rank=float(i + 1)) for i, rec in enumerate(shuffled)
            ]

        return RecommendationList(
            recommendations=shuffled, recommendation_list_id=self.recommendation_list_id
        )

    def filter(self, predicate: Callable[[Recommendation], bool]) -> RecommendationList:
        return RecommendationList(
            recommendations=[r for r in self.recommendations if predicate(r)],
            recommendation_list_id=self.recommendation_list_id,
        )

    def map(self, fn: Callable[[Recommendation], Any]) -> list[Any]:
        return [fn(r) for r in self.recommendations]

    def append(self, rec: Recommendation) -> None:
        self.recommendations.append(rec)

    def extend(self, recs: list[Recommendation]) -> None:
        self.recommendations.extend(recs)

    def as_item_score_pairs(self) -> list[tuple[Any, Any]]:
        return [(r.item_id, r.score) for r in self.recommendations]

    def dedupe(
        self,
        key: Callable[[Recommendation], Any] = lambda r: r.item_id,
        keep: str = "first",
    ) -> RecommendationList:
        """Remove duplicates by key."""
        assert keep in {"first", "last"}
        seen = {}
        if keep == "first":
            for r in self.recommendations:
                k = key(r)
                if k not in seen:
                    seen[k] = r
        else:
            for r in self.recommendations:
                seen[key(r)] = r
        return RecommendationList(
            recommendations=list(seen.values()),
            recommendation_list_id=self.recommendation_list_id,
        )

    def merge(
        self,
        other: RecommendationList,
        resolve: Callable[[Recommendation, Recommendation], Recommendation]
        | None = None,
    ) -> RecommendationList:
        """Merge with another RecommendationList.

        Args:
            other: Another RecommendationList.
            resolve: Conflict resolution function. Default keeps higher score.
        """
        by_id: dict[Any, Recommendation] = {}
        for r in self.recommendations:
            by_id[r.item_id] = r
        for r in other:
            if r.item_id not in by_id:
                by_id[r.item_id] = r
            else:
                if resolve is not None:
                    by_id[r.item_id] = resolve(by_id[r.item_id], r)
                else:
                    a, b = by_id[r.item_id], r
                    a_score = (
                        float("-inf")
                        if a.score is None
                        or (isinstance(a.score, float) and not math.isfinite(a.score))
                        else a.score
                    )
                    b_score = (
                        float("-inf")
                        if b.score is None
                        or (isinstance(b.score, float) and not math.isfinite(b.score))
                        else b.score
                    )
                    by_id[r.item_id] = a if a_score >= b_score else b
        merged_list = list(by_id.values())
        return RecommendationList(
            recommendations=merged_list,
            recommendation_list_id=self.recommendation_list_id,
        )

    def open_all_websites(self) -> None:
        """Open all recommended articles in the default browser."""
        for rec in self.recommendations:
            rec.open_website()
