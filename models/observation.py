"""Observation model for the Content Optimization RL Environment."""

from pydantic import BaseModel, Field
from typing import List


class Observation(BaseModel):
    """
    Represents the current state visible to the agent.

    The observation includes the original transcript, the current draft,
    all quality scores, and metadata about the episode progress.
    """

    original_content: str = Field(
        ..., description="The original unmodified content provided at reset."
    )
    current_draft: str = Field(
        ..., description="The current version of the content after applied actions."
    )
    seo_score: float = Field(
        ..., ge=0.0, le=1.0, description="SEO quality score (0-1)."
    )
    readability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Readability score (0-1)."
    )
    engagement_score: float = Field(
        ..., ge=0.0, le=1.0, description="Engagement quality score (0-1)."
    )
    sentiment_score: float = Field(
        ..., ge=0.0, le=1.0, description="Sentiment alignment score (0-1)."
    )
    step_count: int = Field(
        ..., ge=0, description="Number of steps taken so far in the episode."
    )
    actions_taken: List[str] = Field(
        default_factory=list, description="History of action names taken this episode."
    )
    task_id: str = Field(
        ..., description="Identifier for the current task being solved."
    )
