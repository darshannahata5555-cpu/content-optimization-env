"""
Internal mutable state for the Content Optimization RL Environment.

This is the *internal* state container — distinct from the Observation
model that is returned to the agent.  ContentState holds everything the
environment needs between steps, while Observation is a sanitised,
read-only snapshot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Sample content corpus used by reset() for each task
# ---------------------------------------------------------------------------

TASK_CONTENT: Dict[str, Dict[str, str]] = {
    # Task 1 – Easy: headline SEO optimization
    "headline_seo": {
        "original": (
            "Our New Product is Here\n\n"
            "We are excited to announce our new product that will change "
            "the way you work.  It has many features and is very affordable. "
            "Buy it now from our website."
        ),
        "target_keywords": "productivity,software,tool,workflow,automation",
        "target_tone": "professional",
    },
    # Task 2 – Medium: blog readability + keyword usage
    "blog_readability": {
        "original": (
            "Why Businesses Need Digital Transformation\n\n"
            "In today's rapidly evolving technological landscape, businesses "
            "that fail to adopt digital transformation strategies risk falling "
            "behind their competitors and losing significant market share in "
            "an increasingly interconnected and digitally-driven global "
            "economy that demands agility, innovation, and a customer-centric "
            "approach to service delivery and product development across "
            "every vertical and horizontal within the organization, from "
            "human resources and finance to marketing and supply chain "
            "management, necessitating a comprehensive overhaul of legacy "
            "systems, processes, and cultural mindsets that may have served "
            "the organization well in previous decades but are now "
            "fundamentally inadequate for the challenges and opportunities "
            "presented by the fourth industrial revolution."
        ),
        "target_keywords": "digital transformation,business,technology,innovation,cloud,AI",
        "target_tone": "informative",
    },
    # Task 3 – Hard: ORM reply generation
    "orm_reply": {
        "original": (
            "Customer Review: 'I am extremely disappointed with your "
            "service.  The delivery was late, the product was damaged, and "
            "nobody in your support team seemed to care.  I will never buy "
            "from you again.  Terrible experience overall.'\n\n"
            "Draft Reply: 'Sorry about that.'"
        ),
        "target_keywords": "apology,resolution,customer satisfaction,support,improvement",
        "target_tone": "empathetic",
    },
}


@dataclass
class ContentState:
    """Mutable internal environment state."""

    # Content ---------------------------------------------------------------
    original_content: str = ""
    current_draft: str = ""
    target_keywords: List[str] = field(default_factory=list)
    target_tone: str = "professional"

    # Scores (0 → 1) -------------------------------------------------------
    seo_score: float = 0.0
    readability_score: float = 0.0
    engagement_score: float = 0.0
    sentiment_score: float = 0.0

    # Episode bookkeeping ---------------------------------------------------
    step_count: int = 0
    max_steps: int = 10
    done: bool = False
    task_id: str = "headline_seo"
    actions_taken: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @classmethod
    def from_task(cls, task_id: str, max_steps: int = 10) -> "ContentState":
        """Create a fresh state for a given task."""
        task_data = TASK_CONTENT[task_id]
        keywords = [k.strip() for k in task_data["target_keywords"].split(",")]
        return cls(
            original_content=task_data["original"],
            current_draft=task_data["original"],
            target_keywords=keywords,
            target_tone=task_data["target_tone"],
            task_id=task_id,
            max_steps=max_steps,
        )

    def snapshot(self) -> "ContentState":
        """Return a deep copy of the state (useful for reward comparison)."""
        return copy.deepcopy(self)

    def composite_score(self) -> float:
        """Weighted average of all four quality metrics."""
        return (
            0.30 * self.seo_score
            + 0.25 * self.readability_score
            + 0.25 * self.engagement_score
            + 0.20 * self.sentiment_score
        )
