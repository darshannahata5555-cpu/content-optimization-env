"""
Inference Script - Content Optimization RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import ContentOptimizationEnv
from models.action import Action, ActionType
from models.observation import Observation

# ---------------------------------------------------------------------------
# Environment Variables (MANDATORY)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "content-optimization-env"
MAX_STEPS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.7  # normalized score in [0, 1]

AVAILABLE_ACTIONS = ContentOptimizationEnv.available_actions()

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert content optimization agent. You iteratively improve
    content by choosing one action per step.

    Respond with ONLY a JSON object: {{"action": "<action_name>"}}

    Available actions: {actions}

    Action descriptions:
    - rewrite_headline: Rewrite headline with target keywords, optimize length for SEO
    - add_keywords: Insert missing target keywords naturally into the body
    - improve_readability: Break long sentences, simplify vocabulary, add paragraph breaks
    - shorten_content: Remove filler phrases and redundancies
    - change_tone: Adjust tone toward target (professional / empathetic / informative)
    - optimize_cta: Add or improve call-to-action at end of content
    - generate_reply: Build or enhance an ORM-style reply to a customer review
    - no_op: Do nothing (only if content is already optimal)

    Strategy:
    - Avoid repeating the same action consecutively (penalty applies).
    - Focus on the weakest score first.
    - For ORM tasks, prioritize generate_reply and change_tone.
    - For SEO tasks, prioritize rewrite_headline and add_keywords.
    - For readability tasks, prioritize improve_readability and shorten_content.
    """
).strip().format(actions=", ".join(AVAILABLE_ACTIONS))


# ---------------------------------------------------------------------------
# Logging helpers (MANDATORY stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction via OpenAI Client
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Observation, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {obs.task_id}
        Step: {step}
        SEO: {obs.seo_score:.3f} | Readability: {obs.readability_score:.3f} | Engagement: {obs.engagement_score:.3f} | Sentiment: {obs.sentiment_score:.3f}
        Actions taken: {obs.actions_taken}

        --- Current Draft ---
        {obs.current_draft}
        --- End Draft ---

        Previous steps:
        {history_block}

        Choose the best action to improve the content. Reply with JSON only.
        """
    ).strip()


def parse_action(response_text: str) -> ActionType:
    """Extract the action name from the model's response."""
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


def get_model_action(client: OpenAI, obs: Observation, step: int, history: List[str]) -> ActionType:
    """Ask the LLM which action to take next."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text) if text else ActionType.NO_OP
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ActionType.NO_OP


# ---------------------------------------------------------------------------
# Heuristic fallback agent (when no API key is available)
# ---------------------------------------------------------------------------

HEURISTIC_PLANS = {
    "headline_seo": [
        "rewrite_headline", "add_keywords", "optimize_cta",
        "improve_readability", "shorten_content", "change_tone",
        "add_keywords", "rewrite_headline", "optimize_cta", "no_op",
    ],
    "blog_readability": [
        "improve_readability", "shorten_content", "add_keywords",
        "change_tone", "optimize_cta", "improve_readability",
        "shorten_content", "add_keywords", "change_tone", "no_op",
    ],
    "orm_reply": [
        "generate_reply", "change_tone", "optimize_cta",
        "add_keywords", "improve_readability", "shorten_content",
        "generate_reply", "change_tone", "optimize_cta", "no_op",
    ],
}


def heuristic_action(task_id: str, step: int) -> ActionType:
    plan = HEURISTIC_PLANS.get(task_id, HEURISTIC_PLANS["headline_seo"])
    idx = min(step, len(plan) - 1)
    return ActionType(plan[idx])


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_id: str, client: Optional[OpenAI], use_llm: bool) -> None:
    """Run a full episode on one task with proper [START]/[STEP]/[END] logging."""

    env = ContentOptimizationEnv(task_id=task_id, max_steps=MAX_STEPS)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            # Choose action
            if use_llm and client:
                action_type = get_model_action(client, obs, step, history)
            else:
                action_type = heuristic_action(task_id, step - 1)

            action = Action(action_type=action_type)
            obs, reward_obj, done, info = env.step(action)

            reward = reward_obj.total
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_type.value,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action_type.value} -> reward {reward:+.2f}"
            )

            if done:
                break

        # Score is the task grade (already in [0, 1])
        score = info["task_grade"]
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Exception during episode: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Set up OpenAI client
    use_llm = bool(HF_TOKEN)
    client = None

    if use_llm:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    else:
        print("[DEBUG] HF_TOKEN not set -- using heuristic agent", flush=True)

    # Run all 3 tasks
    tasks = ContentOptimizationEnv.available_tasks()
    for task_id in tasks:
        run_task(task_id, client, use_llm)


if __name__ == "__main__":
    main()
