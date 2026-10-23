from __future__ import annotations

import math
import uuid
import webbrowser
from functools import total_ordering
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_serializer

from recommender.utils.helpers import coerce_uuid, sanitize_numeric


@total_ordering
class Recommendation(BaseModel):
    """Single recommendation with validation and serialization.

    Attributes:
        item_id: Recommended item identifier.
        score: Numeric score (higher is better). NaN/Inf converted to None.
        rank: Position (1-based).
        model: Model/algorithm name.
        metadata: Additional metadata.
        recommendation_id: Unique identifier.
    """

    model_config = {"validate_assignment": True}

    item_id: int
    score: float | None = None
    rank: float
    model_id: uuid.UUID
    metadata: dict[str, Any] = Field(default_factory=dict)
    recommendation_id: uuid.UUID = Field(default_factory=uuid.uuid4)

    @field_validator("score", "rank", mode="before")
    @classmethod
    def sanitize_numeric_fields(cls, v: Any) -> float | None:
        return sanitize_numeric(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def ensure_metadata_dict(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        try:
            return dict(v)
        except Exception:
            return {}

    @field_validator("recommendation_id", mode="before")
    @classmethod
    def coerce_recommendation_id(cls, v: uuid.UUID | str | None) -> uuid.UUID:
        return coerce_uuid(v)

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        return {
            "recommendation_id": str(self.recommendation_id),
            "item_id": self.item_id,
            "score": self.score,
            "rank": self.rank,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }

    def open_website(self) -> None:
        """Open the Politiken article in the default browser."""
        url = f"https://politiken.dk/{self.item_id}"
        webbrowser.open(url)

    def copy_with(
        self,
        *,
        item_id: int | None = None,
        score: float | None = None,
        rank: float | None = None,
        model_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
        new_id: bool = False,
    ) -> Recommendation:
        """Return a copy with updated fields."""
        rid = uuid.uuid4() if new_id else self.recommendation_id
        return Recommendation(
            item_id=self.item_id if item_id is None else item_id,
            score=score if score is not None else self.score,
            rank=rank if rank is not None else self.rank,
            model_id=model_id if model_id is not None else self.model_id,
            metadata=metadata if metadata is not None else dict(self.metadata),
            recommendation_id=rid,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Recommendation):
            return NotImplemented
        return (
            self.item_id == other.item_id
            and self.score == other.score
            and self.rank == other.rank
            and self.model_id == other.model_id
            and self.metadata == other.metadata
        )

    def __lt__(self, other: Recommendation) -> bool:
        if not isinstance(other, Recommendation):
            raise NotImplementedError
        a = (
            self.score
            if (self.score is not None and math.isfinite(self.score))
            else float("-inf")
        )
        b = (
            other.score
            if (other.score is not None and math.isfinite(other.score))
            else float("-inf")
        )
        if a != b:
            return a < b
        return str(self.item_id) < str(other.item_id)

    def __hash__(self) -> int:
        metadata_tuple = tuple(sorted(self.metadata.items()))
        return hash(
            (self.item_id, self.score, self.rank, self.model_id, metadata_tuple)
        )

    def __repr__(self) -> str:
        return (
            f"Recommendation(item_id={self.item_id!r}, score={self.score!r}, "
            f"rank={self.rank!r}, model={self.model_id!r}, metadata_keys={list(self.metadata.keys())})"
        )
