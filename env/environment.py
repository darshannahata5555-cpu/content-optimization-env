"""
Content Optimization RL Environment — main environment class.

Follows the OpenEnv specification:
  - reset()         → Observation
  - step(action)    → (Observation, Reward, bool, dict)
  - state()         → dict   (full serialisable state)

The environment is fully deterministic, self-contained, and requires
no external API calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.observation import Observation
from models.action import Action, ActionType
from models.reward import Reward
from env.state import ContentState, TASK_CONTENT
from env.actions import apply_action
from env.reward import compute_reward
from env.graders import grade_task, recompute_metrics


# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------

class ContentOptimizationEnv:
    """
    OpenEnv-compatible RL environment for iterative content optimisation.

    The agent takes discrete actions (rewrite headline, add keywords, …)
    and receives per-step rewards based on measurable quality deltas.
    """

    # Class-level metadata
    TASK_IDS: List[str] = list(TASK_CONTENT.keys())
    ACTIONS: List[str] = [a.value for a in ActionType]
    MAX_STEPS_DEFAULT: int = 10
    SCORE_THRESHOLD: float = 0.90

    def __init__(
        self,
        task_id: str = "headline_seo",
        max_steps: int = MAX_STEPS_DEFAULT,
    ) -> None:
        if task_id not in TASK_CONTENT:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from {self.TASK_IDS}"
            )
        self._task_id = task_id
        self._max_steps = max_steps
        self._state: ContentState = ContentState()
        self._cumulative_reward: float = 0.0
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment to the initial state for the configured task.

        Returns
        -------
        Observation
            The initial observation.
        """
        self._state = ContentState.from_task(self._task_id, self._max_steps)
        recompute_metrics(self._state)
        self._cumulative_reward = 0.0
        self._episode_rewards = []
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : Action
            The action the agent wants to take.

        Returns
        -------
        observation : Observation
        reward : Reward
        done : bool
        info : dict
        """
        if self._state.done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping again."
            )

        # 1. Snapshot before
        old_state = self._state.snapshot()

        # 2. Apply action to content
        self._state.current_draft = apply_action(
            action_type=action.action_type,
            draft=self._state.current_draft,
            target_keywords=self._state.target_keywords,
            target_tone=self._state.target_tone,
            parameters=action.parameters,
        )

        # 3. Recompute all metrics
        recompute_metrics(self._state)

        # 4. Bookkeeping
        self._state.step_count += 1
        self._state.actions_taken.append(action.action_type.value)

        # 5. Compute reward
        reward = compute_reward(old_state, self._state, action.action_type.value)
        self._cumulative_reward += reward.total
        self._episode_rewards.append(reward.total)

        # 6. Check termination
        composite = self._state.composite_score()
        if (
            self._state.step_count >= self._state.max_steps
            or composite >= self.SCORE_THRESHOLD
        ):
            self._state.done = True

        # 7. Build info dict
        task_grade = grade_task(self._state)
        info: Dict[str, Any] = {
            "composite_score": round(composite, 4),
            "task_grade": round(task_grade, 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "action_applied": action.action_type.value,
        }

        return self._make_observation(), reward, self._state.done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the full serialisable state of the environment.

        This can be used for checkpointing / debugging.
        """
        return {
            "task_id": self._state.task_id,
            "original_content": self._state.original_content,
            "current_draft": self._state.current_draft,
            "seo_score": self._state.seo_score,
            "readability_score": self._state.readability_score,
            "engagement_score": self._state.engagement_score,
            "sentiment_score": self._state.sentiment_score,
            "step_count": self._state.step_count,
            "max_steps": self._state.max_steps,
            "done": self._state.done,
            "actions_taken": list(self._state.actions_taken),
            "target_keywords": list(self._state.target_keywords),
            "target_tone": self._state.target_tone,
            "composite_score": round(self._state.composite_score(), 4),
            "task_grade": round(grade_task(self._state), 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "episode_rewards": [round(r, 4) for r in self._episode_rewards],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        return Observation(
            original_content=self._state.original_content,
            current_draft=self._state.current_draft,
            seo_score=self._state.seo_score,
            readability_score=self._state.readability_score,
            engagement_score=self._state.engagement_score,
            sentiment_score=self._state.sentiment_score,
            step_count=self._state.step_count,
            actions_taken=list(self._state.actions_taken),
            task_id=self._state.task_id,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def available_tasks(cls) -> List[str]:
        return list(TASK_CONTENT.keys())

    @classmethod
    def available_actions(cls) -> List[str]:
        return [a.value for a in ActionType]

    def __repr__(self) -> str:
        return (
            f"ContentOptimizationEnv(task={self._task_id!r}, "
            f"step={self._state.step_count}/{self._state.max_steps}, "
            f"done={self._state.done})"
        )
