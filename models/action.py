"""Action model for the Content Optimization RL Environment."""

from enum import Enum
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Discrete action types available to the agent."""

    REWRITE_HEADLINE = "rewrite_headline"
    ADD_KEYWORDS = "add_keywords"
    IMPROVE_READABILITY = "improve_readability"
    SHORTEN_CONTENT = "shorten_content"
    CHANGE_TONE = "change_tone"
    OPTIMIZE_CTA = "optimize_cta"
    GENERATE_REPLY = "generate_reply"
    NO_OP = "no_op"


class Action(BaseModel):
    """
    Represents an action the agent can take to modify content.

    Each action targets a specific aspect of the content (SEO, readability,
    engagement, or tone) and optionally includes parameters.
    """

    action_type: ActionType = Field(
        ..., description="The type of action to perform."
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Optional parameters for the action (e.g., target keywords, desired tone).",
    )
