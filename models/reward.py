"""Reward model for the Content Optimization RL Environment."""

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """
    Structured reward returned after each step.

    Breaks down the total reward into components so agents (and humans)
    can understand what drove the reward signal.
    """

    total: float = Field(
        ..., description="Total scalar reward for this step."
    )
    seo_delta: float = Field(
        0.0, description="Reward component from SEO score change."
    )
    readability_delta: float = Field(
        0.0, description="Reward component from readability score change."
    )
    engagement_delta: float = Field(
        0.0, description="Reward component from engagement score change."
    )
    sentiment_delta: float = Field(
        0.0, description="Reward component from sentiment score change."
    )
    repetition_penalty: float = Field(
        0.0, description="Penalty for repeating the same action consecutively."
    )
    no_improvement_penalty: float = Field(
        0.0, description="Penalty when the action produced no measurable improvement."
    )
    degradation_penalty: float = Field(
        0.0, description="Penalty when content quality decreased."
    )
