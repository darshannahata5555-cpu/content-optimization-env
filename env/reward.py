"""
Reward computation for the Content Optimization RL Environment.

The reward is a per-step signal that encourages the agent to improve
all quality dimensions while penalising stagnation, repetition, and
degradation.
"""

from __future__ import annotations

from env.state import ContentState
from models.reward import Reward


# ---------------------------------------------------------------------------
# Weights (positive incentives)
# ---------------------------------------------------------------------------
W_SEO = 1.0
W_READ = 1.0
W_ENGAGE = 1.0
W_SENT = 1.0

# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------
PENALTY_REPETITION = -0.05       # repeated action in a row
PENALTY_NO_IMPROVEMENT = -0.02   # step produced zero delta
PENALTY_DEGRADATION = -0.10      # any metric got worse


def compute_reward(
    old_state: ContentState,
    new_state: ContentState,
    action_name: str,
) -> Reward:
    """
    Compare *old_state* and *new_state* and produce a structured Reward.

    Parameters
    ----------
    old_state : ContentState
        Snapshot taken **before** the action was applied.
    new_state : ContentState
        State **after** the action was applied and metrics recomputed.
    action_name : str
        Name of the action just taken (used for repetition penalty).

    Returns
    -------
    Reward
        A Pydantic reward with decomposed components.
    """

    # --- per-metric deltas -------------------------------------------------
    seo_d = new_state.seo_score - old_state.seo_score
    read_d = new_state.readability_score - old_state.readability_score
    engage_d = new_state.engagement_score - old_state.engagement_score
    sent_d = new_state.sentiment_score - old_state.sentiment_score

    # --- penalties ---------------------------------------------------------
    rep_pen = 0.0
    if (
        len(old_state.actions_taken) >= 1
        and old_state.actions_taken[-1] == action_name
    ):
        rep_pen = PENALTY_REPETITION

    total_delta = seo_d + read_d + engage_d + sent_d
    no_imp_pen = PENALTY_NO_IMPROVEMENT if abs(total_delta) < 1e-6 else 0.0

    deg_pen = 0.0
    if total_delta < -1e-6:
        deg_pen = PENALTY_DEGRADATION

    # --- total reward ------------------------------------------------------
    total = (
        W_SEO * seo_d
        + W_READ * read_d
        + W_ENGAGE * engage_d
        + W_SENT * sent_d
        + rep_pen
        + no_imp_pen
        + deg_pen
    )

    return Reward(
        total=round(total, 6),
        seo_delta=round(seo_d, 6),
        readability_delta=round(read_d, 6),
        engagement_delta=round(engage_d, 6),
        sentiment_delta=round(sent_d, 6),
        repetition_penalty=round(rep_pen, 6),
        no_improvement_penalty=round(no_imp_pen, 6),
        degradation_penalty=round(deg_pen, 6),
    )
