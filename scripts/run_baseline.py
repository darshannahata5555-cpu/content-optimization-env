"""
Baseline inference script – runs an OpenAI model against the
Content Optimization RL Environment on all 3 tasks.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python scripts/run_baseline.py

The script interacts with the environment via the step() loop,
asking the model which action to take at each step, and logs
rewards, scores, and final grades.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import ContentOptimizationEnv
from models.action import Action, ActionType
from models.observation import Observation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AVAILABLE_ACTIONS = ContentOptimizationEnv.available_actions()

SYSTEM_PROMPT = """\
You are an expert content optimization agent.  You are given the current
state of a piece of content and must choose the single best action to
improve it.  Respond with ONLY a JSON object:

{{"action": "<action_name>"}}

Available actions: {actions}

Strategy tips:
- "rewrite_headline" improves SEO for the headline.
- "add_keywords" boosts keyword density / SEO.
- "improve_readability" simplifies long sentences and vocabulary.
- "shorten_content" removes filler words.
- "change_tone" aligns tone (professional / empathetic / informative).
- "optimize_cta" adds a strong call-to-action.
- "generate_reply" builds or enhances an ORM-style reply.
- "no_op" does nothing (use only if content is already optimal).

Choose wisely — repeated identical actions are penalised.
""".format(actions=", ".join(AVAILABLE_ACTIONS))


def build_user_message(obs: Observation) -> str:
    return (
        f"Task: {obs.task_id}\n"
        f"Step: {obs.step_count}\n"
        f"SEO: {obs.seo_score:.3f}  |  Readability: {obs.readability_score:.3f}  |  "
        f"Engagement: {obs.engagement_score:.3f}  |  Sentiment: {obs.sentiment_score:.3f}\n"
        f"Actions taken so far: {obs.actions_taken}\n\n"
        f"--- Current Draft ---\n{obs.current_draft}\n--- End Draft ---\n\n"
        f"Which action should be taken next?"
    )


def parse_action(response_text: str) -> ActionType:
    """Extract the action name from the model's response."""
    # Try JSON parse first
    try:
        data = json.loads(response_text)
        name = data.get("action", "no_op")
    except json.JSONDecodeError:
        # Fallback: look for a known action name in the text
        name = "no_op"
        for a in AVAILABLE_ACTIONS:
            if a in response_text.lower():
                name = a
                break
    return ActionType(name)


def call_openai(messages: List[dict], api_key: str) -> str:
    """Call OpenAI chat completion (works with openai>=1.0)."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        print(
            "[WARN] openai package not installed. "
            "Falling back to heuristic agent."
        )
        return ""


# ---------------------------------------------------------------------------
# Heuristic fallback agent (no API key needed)
# ---------------------------------------------------------------------------

HEURISTIC_PLANS: Dict[str, List[str]] = {
    "headline_seo": [
        "rewrite_headline",
        "add_keywords",
        "optimize_cta",
        "improve_readability",
        "shorten_content",
        "change_tone",
        "add_keywords",
        "rewrite_headline",
        "optimize_cta",
        "no_op",
    ],
    "blog_readability": [
        "improve_readability",
        "shorten_content",
        "add_keywords",
        "change_tone",
        "optimize_cta",
        "improve_readability",
        "shorten_content",
        "add_keywords",
        "change_tone",
        "no_op",
    ],
    "orm_reply": [
        "generate_reply",
        "change_tone",
        "optimize_cta",
        "add_keywords",
        "improve_readability",
        "shorten_content",
        "generate_reply",
        "change_tone",
        "optimize_cta",
        "no_op",
    ],
}


def heuristic_action(task_id: str, step: int) -> ActionType:
    plan = HEURISTIC_PLANS.get(task_id, HEURISTIC_PLANS["headline_seo"])
    idx = min(step, len(plan) - 1)
    return ActionType(plan[idx])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_task(task_id: str, api_key: str | None, use_llm: bool) -> Dict[str, Any]:
    """Run a full episode on one task. Returns summary dict."""
    env = ContentOptimizationEnv(task_id=task_id, max_steps=10)
    obs = env.reset()

    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")
    print(f"  Initial scores → SEO={obs.seo_score:.3f}  READ={obs.readability_score:.3f}  "
          f"ENG={obs.engagement_score:.3f}  SENT={obs.sentiment_score:.3f}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    step_log: List[dict] = []

    while True:
        # Decide action
        if use_llm and api_key:
            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})
            raw_response = call_openai(messages, api_key)
            if raw_response:
                action_type = parse_action(raw_response)
                messages.append({"role": "assistant", "content": raw_response})
            else:
                action_type = heuristic_action(task_id, obs.step_count)
        else:
            action_type = heuristic_action(task_id, obs.step_count)

        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)

        total_reward += reward.total
        step_log.append({
            "step": obs.step_count,
            "action": action_type.value,
            "reward": round(reward.total, 4),
            "composite": info["composite_score"],
            "task_grade": info["task_grade"],
        })

        print(
            f"  Step {obs.step_count:2d} | {action_type.value:22s} | "
            f"reward={reward.total:+.4f} | composite={info['composite_score']:.3f} | "
            f"grade={info['task_grade']:.3f}"
        )

        if done:
            break

    final_state = env.state()
    summary = {
        "task_id": task_id,
        "steps": obs.step_count,
        "final_seo": obs.seo_score,
        "final_readability": obs.readability_score,
        "final_engagement": obs.engagement_score,
        "final_sentiment": obs.sentiment_score,
        "composite_score": final_state["composite_score"],
        "task_grade": final_state["task_grade"],
        "cumulative_reward": round(total_reward, 4),
        "step_log": step_log,
    }

    print(f"\n  ✓ Episode finished — grade={summary['task_grade']:.3f}  "
          f"reward={summary['cumulative_reward']:+.4f}")

    return summary


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    use_llm = bool(api_key)

    if use_llm:
        print("🤖 Using OpenAI model for action selection.")
    else:
        print("📋 OPENAI_API_KEY not set — using heuristic agent.")

    tasks = ContentOptimizationEnv.available_tasks()
    results: List[Dict[str, Any]] = []

    for task_id in tasks:
        summary = run_task(task_id, api_key, use_llm)
        results.append(summary)

    # ----- Final report ----------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL REPORT")
    print(f"{'='*60}")
    print(f"  {'Task':<20s} {'Grade':>8s} {'Reward':>10s} {'Steps':>6s}")
    print(f"  {'-'*46}")
    for r in results:
        print(
            f"  {r['task_id']:<20s} {r['task_grade']:>8.3f} "
            f"{r['cumulative_reward']:>+10.4f} {r['steps']:>6d}"
        )

    avg_grade = sum(r["task_grade"] for r in results) / len(results)
    total_r = sum(r["cumulative_reward"] for r in results)
    print(f"  {'-'*46}")
    print(f"  {'AVERAGE':<20s} {avg_grade:>8.3f} {total_r:>+10.4f}")
    print()

    # Save JSON report
    report_path = os.path.join(PROJECT_ROOT, "baseline_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  📄 Full report saved to {report_path}")


if __name__ == "__main__":
    main()
