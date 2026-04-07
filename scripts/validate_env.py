"""
Self-validation script for the Content Optimization RL Environment.

This script checks all 4 OpenEnv requirements:
  [PASS] A. openenv.yaml exists with required fields
  [PASS] B. Environment class has reset(), step(), state()
  [PASS] C. step() returns (observation, reward, done, info)
  [PASS] D. Typed Pydantic models exist

Usage:
    python scripts/validate_env.py
"""

from __future__ import annotations

import io
import os
import sys
import traceback

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((status, name))
    msg = f"  {status} {name}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)


def main() -> None:
    print("=" * 60)
    print("  OpenEnv Self-Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # A. openenv.yaml exists with required fields
    # ------------------------------------------------------------------
    print("\n--- A. openenv.yaml ---")
    yaml_path = os.path.join(PROJECT_ROOT, "openenv.yaml")
    yaml_exists = os.path.isfile(yaml_path)
    check("openenv.yaml exists", yaml_exists)

    if yaml_exists:
        try:
            import yaml
            with open(yaml_path) as f:
                manifest = yaml.safe_load(f)

            check(
                "has 'name' field",
                "name" in manifest,
                manifest.get("name", "MISSING"),
            )
            check(
                "has 'description' field",
                "description" in manifest,
                (manifest.get("description", "MISSING") or "")[:60],
            )
            check(
                "has 'entry_point' field",
                "entry_point" in manifest,
                manifest.get("entry_point", "MISSING"),
            )
        except ImportError:
            # No pyyaml — do basic string check
            with open(yaml_path) as f:
                content = f.read()
            check("has 'name' field", "name:" in content)
            check("has 'description' field", "description:" in content)
            check("has 'entry_point' field", "entry_point:" in content)

    # ------------------------------------------------------------------
    # B. Environment class has reset(), step(), state()
    # ------------------------------------------------------------------
    print("\n--- B. Environment class ---")
    try:
        from env.environment import ContentOptimizationEnv

        check("ContentOptimizationEnv importable", True)

        env = ContentOptimizationEnv(task_id="headline_seo")
        check("Constructor works", True)

        check("has reset()", callable(getattr(env, "reset", None)))
        check("has step()", callable(getattr(env, "step", None)))
        check("has state()", callable(getattr(env, "state", None)))
    except Exception as e:
        check("ContentOptimizationEnv importable", False, str(e))

    # ------------------------------------------------------------------
    # C. step() returns (observation, reward, done, info)
    # ------------------------------------------------------------------
    print("\n--- C. step() return format ---")
    try:
        from models.action import Action, ActionType
        from models.observation import Observation
        from models.reward import Reward

        env = ContentOptimizationEnv(task_id="headline_seo")
        obs = env.reset()
        check("reset() returns Observation", isinstance(obs, Observation))

        action = Action(action_type=ActionType.REWRITE_HEADLINE)
        result = env.step(action)

        check("step() returns a tuple", isinstance(result, tuple))
        check("step() returns 4 items", len(result) == 4, f"got {len(result)}")

        obs2, reward, done, info = result
        check("item 0 is Observation", isinstance(obs2, Observation))
        check("item 1 is Reward", isinstance(reward, Reward))
        check("item 2 is bool (done)", isinstance(done, bool))
        check("item 3 is dict (info)", isinstance(info, dict))

        # Verify reward has total field
        check("reward.total is float", isinstance(reward.total, float), f"{reward.total}")

        # Verify observation scores are in [0, 1]
        for field in ["seo_score", "readability_score", "engagement_score", "sentiment_score"]:
            val = getattr(obs2, field)
            check(f"obs.{field} in [0,1]", 0.0 <= val <= 1.0, f"{val:.4f}")

        # Verify state() works
        state = env.state()
        check("state() returns dict", isinstance(state, dict))
        check("state has 'task_id'", "task_id" in state)
        check("state has 'composite_score'", "composite_score" in state)
    except Exception as e:
        check("step() execution failed", False, str(e))
        traceback.print_exc()

    # ------------------------------------------------------------------
    # D. Typed Pydantic models exist
    # ------------------------------------------------------------------
    print("\n--- D. Pydantic models ---")
    try:
        from models.observation import Observation

        check("models/observation.py → Observation", True)
        check("  is Pydantic BaseModel", hasattr(Observation, "model_fields"))
    except ImportError as e:
        check("models/observation.py → Observation", False, str(e))

    try:
        from models.action import Action, ActionType

        check("models/action.py → Action", True)
        check("models/action.py → ActionType enum", True)
        check(
            f"  {len(ActionType)} actions defined",
            len(ActionType) == 8,
            ", ".join(a.value for a in ActionType),
        )
    except ImportError as e:
        check("models/action.py → Action", False, str(e))

    try:
        from models.reward import Reward

        check("models/reward.py → Reward", True)
        check("  has 'total' field", "total" in Reward.model_fields)
    except ImportError as e:
        check("models/reward.py → Reward", False, str(e))

    # ------------------------------------------------------------------
    # E. Bonus: run all 3 tasks
    # ------------------------------------------------------------------
    print("\n--- E. Full episode test (all 3 tasks) ---")
    from env.environment import ContentOptimizationEnv
    from models.action import Action, ActionType

    PLANS = {
        "headline_seo": [ActionType.REWRITE_HEADLINE, ActionType.ADD_KEYWORDS, ActionType.OPTIMIZE_CTA],
        "blog_readability": [ActionType.IMPROVE_READABILITY, ActionType.ADD_KEYWORDS, ActionType.OPTIMIZE_CTA],
        "orm_reply": [ActionType.GENERATE_REPLY, ActionType.CHANGE_TONE, ActionType.OPTIMIZE_CTA],
    }

    for task_id in ["headline_seo", "blog_readability", "orm_reply"]:
        try:
            env = ContentOptimizationEnv(task_id=task_id)
            obs = env.reset()
            for act_type in PLANS[task_id]:
                action = Action(action_type=act_type)
                obs, reward, done, info = env.step(action)
            grade = info["task_grade"]
            check(f"Task '{task_id}' runs OK", True, f"grade={grade:.3f}")
        except Exception as e:
            check(f"Task '{task_id}' runs OK", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(1 for s, _ in results if s == PASS)
    failed = sum(1 for s, _ in results if s == FAIL)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {passed}/{total} checks passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print(f"  -- ALL PASSED")
    print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
