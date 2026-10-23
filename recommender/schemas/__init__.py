"""Recommendation schemas using Pydantic models for easy dict and JSON handling."""

from recommender.schemas.recommendation import Recommendation
from recommender.schemas.recommendation_collection import RecommendationCollection
from recommender.schemas.recommendation_list import RecommendationList

__all__ = ["Recommendation", "RecommendationList", "RecommendationCollection"]
