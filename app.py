"""
app.py — Gradio demo UI + OpenEnv REST API for the Content Optimization RL
Environment.

This gives the HF Space:
  1. A persistent Gradio web UI for interactive use.
  2. REST API endpoints (/reset, /step, /state) for the OpenEnv automated
     checker (Phase 1 validation).
"""

import io
import sys
import traceback
import difflib

# Fix encoding on containers
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import gradio as gr

from env.environment import ContentOptimizationEnv
from env.actions import apply_action
from env.graders import recompute_metrics
from env.reward import compute_reward
from env.state import ContentState
from models.action import Action, ActionType
from models.observation import Observation
from models.reward import Reward

# ---------------------------------------------------------------------------
# FastAPI app (primary — Gradio will be mounted onto this)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Content Optimization RL Environment",
    description="OpenEnv-compatible RL environment for content optimization",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global env instances for API usage (keyed by session, simple global for now)
# ---------------------------------------------------------------------------
API_ENV = None


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(content={"error": message}, status_code=status_code)


async def _read_json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return body if isinstance(body, dict) else {}


def _parse_task_id(body: Dict[str, Any]) -> str:
    return body.get("task_id") or body.get("task") or "headline_seo"


def _parse_max_steps(body: Dict[str, Any]) -> int:
    raw_value = body.get("max_steps", 10)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 10


def _build_reset_payload() -> Dict[str, Any]:
    observation = API_ENV.reset()
    return observation.model_dump()


def _build_step_payload(action_data: Dict[str, Any]) -> Dict[str, Any]:
    action_type_raw = action_data.get("action_type") or action_data.get("action") or "no_op"
    parameters = action_data.get("parameters", {})

    if isinstance(action_type_raw, dict):
        action_type_raw = action_type_raw.get("action_type", "no_op")
    if isinstance(action_type_raw, list):
        action_type_raw = action_type_raw[0] if action_type_raw else "no_op"

    action = Action(
        action_type=ActionType(str(action_type_raw)),
        parameters=parameters if isinstance(parameters, dict) else {},
    )
    obs, reward, done, info = API_ENV.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


# ---------------------------------------------------------------------------
# OpenEnv REST API endpoints
# ---------------------------------------------------------------------------


@app.post("/reset")
@app.post("/openenv/reset")
async def api_reset(request: Request):
    """
    Reset the environment.

    Expected JSON body:
        {"task_id": "headline_seo"}  (optional, defaults to headline_seo)

    Returns: Observation as JSON
    """
    global API_ENV
    body = await _read_json_body(request)
    task_id = _parse_task_id(body)
    max_steps = _parse_max_steps(body)

    try:
        API_ENV = ContentOptimizationEnv(task_id=task_id, max_steps=max_steps)
        return JSONResponse(content=_build_reset_payload(), status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/step")
@app.post("/openenv/step")
async def api_step(request: Request):
    """
    Take one step in the environment.

    Expected JSON body:
        {"action": {"action_type": "rewrite_headline", "parameters": {}}}
        or simply:
        {"action_type": "rewrite_headline"}

    Returns: {"observation": ..., "reward": ..., "done": bool, "info": dict}
    """
    global API_ENV
    if API_ENV is None:
        return _json_error("Environment not initialized. Call POST /reset first.")

    body = await _read_json_body(request)

    try:
        if "action" in body and isinstance(body["action"], dict):
            action_data = body["action"]
        else:
            action_data = body
        return JSONResponse(content=_build_step_payload(action_data), status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/state")
@app.post("/state")
@app.get("/openenv/state")
@app.post("/openenv/state")
async def api_state():
    """
    Return the full serialisable environment state.

    Returns: dict with all environment state fields.
    """
    global API_ENV
    if API_ENV is None:
        return _json_error("Environment not initialized. Call POST /reset first.")

    try:
        state = API_ENV.state()
        return JSONResponse(content=state, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "environment": "content-optimization-env",
            "version": "1.0.0",
            "tasks": ContentOptimizationEnv.available_tasks(),
            "actions": ContentOptimizationEnv.available_actions(),
        },
        status_code=200,
    )


@app.get("/metadata")
async def metadata():
    """OpenEnv metadata endpoint."""
    return JSONResponse(
        content={
            "name": "content-optimization-env",
            "display_name": "Content Optimization RL Environment",
            "description": "RL environment for optimizing content with iterative actions and per-step rewards.",
            "version": "1.0.0",
            "mode": "simulation",
        },
        status_code=200,
    )


@app.get("/schema")
async def schema():
    """Return action, observation, and state schemas."""
    return JSONResponse(
        content={
            "action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
            "state": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "original_content": {"type": "string"},
                    "current_draft": {"type": "string"},
                    "seo_score": {"type": "number"},
                    "readability_score": {"type": "number"},
                    "engagement_score": {"type": "number"},
                    "sentiment_score": {"type": "number"},
                    "step_count": {"type": "integer"},
                    "max_steps": {"type": "integer"},
                    "done": {"type": "boolean"},
                    "actions_taken": {"type": "array", "items": {"type": "string"}},
                    "target_keywords": {"type": "array", "items": {"type": "string"}},
                    "target_tone": {"type": "string"},
                    "composite_score": {"type": "number"},
                    "task_grade": {"type": "number"},
                    "cumulative_reward": {"type": "number"},
                    "episode_rewards": {"type": "array", "items": {"type": "number"}},
                },
                "required": [
                    "task_id",
                    "original_content",
                    "current_draft",
                    "seo_score",
                    "readability_score",
                    "engagement_score",
                    "sentiment_score",
                    "step_count",
                    "max_steps",
                    "done",
                    "actions_taken",
                    "target_keywords",
                    "target_tone",
                    "composite_score",
                    "task_grade",
                    "cumulative_reward",
                    "episode_rewards",
                ],
            },
            "reward": Reward.model_json_schema(),
        },
        status_code=200,
    )


@app.post("/mcp")
async def mcp():
    """Minimal JSON-RPC probe response for validator compatibility."""
    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": None,
            "result": {
                "name": "content-optimization-env",
                "status": "healthy",
            },
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Gradio UI functions (unchanged)
# ---------------------------------------------------------------------------
GRADIO_ENV = None
CUSTOM_STATE = None
CUSTOM_INITIAL_DRAFT = ""
CUSTOM_CUMULATIVE_REWARD = 0.0
CUSTOM_REWARDS = []

TASKS = ["headline_seo", "blog_readability", "orm_reply"]
ACTIONS = [a.value for a in ActionType]

ACTION_RATIONALE = {
    "rewrite_headline": "Improves headline SEO and keyword targeting.",
    "add_keywords": "Adds missing target keywords to improve discoverability.",
    "improve_readability": "Breaks complex text into simpler, clearer structure.",
    "shorten_content": "Removes filler and redundant phrasing.",
    "change_tone": "Aligns tone with your chosen communication style.",
    "optimize_cta": "Strengthens conversion by improving the call-to-action.",
    "generate_reply": "Generates a stronger support/ORM-style response.",
    "no_op": "Keeps the current draft unchanged.",
}


def _parse_keywords(raw_keywords: str) -> list[str]:
    parts = [p.strip() for p in (raw_keywords or "").split(",")]
    return [p for p in parts if p]


def _format_diff(before: str, after: str, max_lines: int = 120) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    if not lines:
        return "No text changes yet."
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (diff truncated)"]
    return "\n".join(lines)


def _format_custom_summary(
    state: ContentState,
    reward: float,
    action_name: str,
) -> str:
    grade = state.composite_score()
    done_val = str(state.done).lower()
    rationale = ACTION_RATIONALE.get(action_name, "Action applied.")
    return (
        f"Step: {state.step_count}/{state.max_steps}\n"
        f"Action: {action_name}\n"
        f"Rationale: {rationale}\n"
        f"Reward: {reward:+.4f}\n"
        f"Cumulative Reward: {CUSTOM_CUMULATIVE_REWARD:+.4f}\n"
        f"SEO: {state.seo_score:.3f} | Readability: {state.readability_score:.3f} | "
        f"Engagement: {state.engagement_score:.3f} | Sentiment: {state.sentiment_score:.3f}\n"
        f"Composite Grade: {grade:.3f}\n"
        f"Done: {done_val}"
    )


def _suggest_custom_action(state: ContentState) -> ActionType:
    if state.step_count == 0 and state.target_tone == "empathetic":
        return ActionType.GENERATE_REPLY

    scores = {
        "seo": state.seo_score,
        "readability": state.readability_score,
        "engagement": state.engagement_score,
        "sentiment": state.sentiment_score,
    }
    weakest = min(scores, key=scores.get)

    candidate_map = {
        "seo": [ActionType.REWRITE_HEADLINE, ActionType.ADD_KEYWORDS],
        "readability": [ActionType.IMPROVE_READABILITY, ActionType.SHORTEN_CONTENT],
        "engagement": [ActionType.OPTIMIZE_CTA, ActionType.ADD_KEYWORDS],
        "sentiment": [ActionType.CHANGE_TONE, ActionType.GENERATE_REPLY],
    }
    candidates = candidate_map[weakest]
    last_action = state.actions_taken[-1] if state.actions_taken else None

    for action in candidates:
        if action.value != last_action:
            return action
    return candidates[0]


def custom_reset_env(
    input_text: str,
    keywords_csv: str,
    tone: str,
    max_steps: int,
) -> tuple[str, str, str, str, str]:
    global CUSTOM_STATE, CUSTOM_INITIAL_DRAFT, CUSTOM_CUMULATIVE_REWARD, CUSTOM_REWARDS

    draft = (input_text or "").strip()
    if not draft:
        err = "ERROR: Please provide input text."
        return err, "", "", "", err

    target_keywords = _parse_keywords(keywords_csv)
    if not target_keywords:
        target_keywords = ["content", "quality", "improvement"]

    CUSTOM_STATE = ContentState(
        original_content=draft,
        current_draft=draft,
        target_keywords=target_keywords,
        target_tone=tone,
        task_id="custom_input",
        max_steps=max_steps,
    )
    recompute_metrics(CUSTOM_STATE)

    CUSTOM_INITIAL_DRAFT = draft
    CUSTOM_CUMULATIVE_REWARD = 0.0
    CUSTOM_REWARDS = []

    summary = _format_custom_summary(CUSTOM_STATE, 0.0, "reset")
    return summary, draft, draft, _format_diff(draft, draft), "Custom session initialized."


def custom_step_env(action_name: str) -> tuple[str, str, str, str, str]:
    global CUSTOM_STATE, CUSTOM_CUMULATIVE_REWARD, CUSTOM_REWARDS

    if CUSTOM_STATE is None:
        err = "ERROR: Initialize custom session first."
        return err, "", "", "", err

    if CUSTOM_STATE.done:
        summary = _format_custom_summary(CUSTOM_STATE, 0.0, action_name)
        return summary, CUSTOM_STATE.current_draft, CUSTOM_STATE.current_draft, "No changes. Episode already done.", "Episode finished."

    before = CUSTOM_STATE.current_draft
    old_state = CUSTOM_STATE.snapshot()
    action_type = ActionType(action_name)
    after = apply_action(
        action_type=action_type,
        draft=CUSTOM_STATE.current_draft,
        target_keywords=CUSTOM_STATE.target_keywords,
        target_tone=CUSTOM_STATE.target_tone,
        parameters={},
    )
    CUSTOM_STATE.current_draft = after
    recompute_metrics(CUSTOM_STATE)
    CUSTOM_STATE.step_count += 1
    CUSTOM_STATE.actions_taken.append(action_name)

    reward_obj = compute_reward(old_state, CUSTOM_STATE, action_name)
    CUSTOM_CUMULATIVE_REWARD += reward_obj.total
    CUSTOM_REWARDS.append(reward_obj.total)

    composite = CUSTOM_STATE.composite_score()
    if CUSTOM_STATE.step_count >= CUSTOM_STATE.max_steps or composite >= 0.90:
        CUSTOM_STATE.done = True

    summary = _format_custom_summary(CUSTOM_STATE, reward_obj.total, action_name)
    rationale = ACTION_RATIONALE.get(action_name, "Action applied.")
    return summary, before, after, _format_diff(before, after), rationale


def custom_auto_optimize() -> tuple[str, str, str, str, str]:
    global CUSTOM_STATE
    if CUSTOM_STATE is None:
        err = "ERROR: Initialize custom session first."
        return err, "", "", "", err

    logs = []
    while not CUSTOM_STATE.done:
        action = _suggest_custom_action(CUSTOM_STATE)
        summary, _, _, _, rationale = custom_step_env(action.value)
        logs.append(f"step={CUSTOM_STATE.step_count} action={action.value} rationale={rationale}")
        if CUSTOM_STATE.step_count >= CUSTOM_STATE.max_steps:
            break

    final_summary = summary if logs else _format_custom_summary(CUSTOM_STATE, 0.0, "no_op")
    return (
        final_summary,
        CUSTOM_INITIAL_DRAFT,
        CUSTOM_STATE.current_draft,
        _format_diff(CUSTOM_INITIAL_DRAFT, CUSTOM_STATE.current_draft),
        "\n".join(logs) if logs else "No optimization steps executed.",
    )


def reset_env(task_id: str) -> str:
    """Reset the environment with the selected task."""
    global GRADIO_ENV
    GRADIO_ENV = ContentOptimizationEnv(task_id=task_id, max_steps=10)
    obs = GRADIO_ENV.reset()
    state = GRADIO_ENV.state()

    output = (
        f"=== Environment Reset ===\n"
        f"Task: {task_id}\n\n"
        f"--- Scores ---\n"
        f"  SEO:         {obs.seo_score:.3f}\n"
        f"  Readability: {obs.readability_score:.3f}\n"
        f"  Engagement:  {obs.engagement_score:.3f}\n"
        f"  Sentiment:   {obs.sentiment_score:.3f}\n"
        f"  Composite:   {state['composite_score']:.3f}\n\n"
        f"--- Current Draft ---\n{obs.current_draft}\n"
    )
    return output


def step_env(action_name: str) -> str:
    """Take one step in the environment."""
    global GRADIO_ENV
    if GRADIO_ENV is None:
        return "ERROR: Please reset the environment first!"

    if GRADIO_ENV._state.done:
        state = GRADIO_ENV.state()
        return (
            f"Episode is DONE.\n"
            f"Final Grade: {state['task_grade']:.3f}\n"
            f"Cumulative Reward: {state['cumulative_reward']:+.4f}\n\n"
            f"Click Reset to start a new episode."
        )

    try:
        action_type = ActionType(action_name)
    except ValueError:
        return f"ERROR: Unknown action '{action_name}'"

    action = Action(action_type=action_type)
    obs, reward, done, info = GRADIO_ENV.step(action)

    status = "DONE" if done else "RUNNING"
    output = (
        f"=== Step {obs.step_count} | {action_name} | {status} ===\n\n"
        f"--- Reward ---\n"
        f"  Total:       {reward.total:+.4f}\n"
        f"  SEO delta:   {reward.seo_delta:+.4f}\n"
        f"  Read delta:  {reward.readability_delta:+.4f}\n"
        f"  Engage delta:{reward.engagement_delta:+.4f}\n"
        f"  Sent delta:  {reward.sentiment_delta:+.4f}\n"
        f"  Penalties:   rep={reward.repetition_penalty:.2f} "
        f"noimpr={reward.no_improvement_penalty:.2f} "
        f"degrad={reward.degradation_penalty:.2f}\n\n"
        f"--- Scores ---\n"
        f"  SEO:         {obs.seo_score:.3f}\n"
        f"  Readability: {obs.readability_score:.3f}\n"
        f"  Engagement:  {obs.engagement_score:.3f}\n"
        f"  Sentiment:   {obs.sentiment_score:.3f}\n"
        f"  Composite:   {info['composite_score']:.3f}\n"
        f"  Task Grade:  {info['task_grade']:.3f}\n\n"
        f"--- Current Draft ---\n{obs.current_draft}\n"
    )

    if done:
        state = GRADIO_ENV.state()
        output += (
            f"\n=== EPISODE FINISHED ===\n"
            f"Final Grade: {state['task_grade']:.3f}\n"
            f"Cumulative Reward: {state['cumulative_reward']:+.4f}\n"
        )

    return output


def run_full_episode(task_id: str) -> str:
    """Run a full heuristic episode and return the log."""
    plans = {
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

    env = ContentOptimizationEnv(task_id=task_id, max_steps=10)
    obs = env.reset()

    lines = [
        f"[START] task={task_id} env=content-optimization-env model=heuristic",
        f"  Initial: SEO={obs.seo_score:.3f} READ={obs.readability_score:.3f} "
        f"ENG={obs.engagement_score:.3f} SENT={obs.sentiment_score:.3f}",
        "",
    ]

    plan = plans.get(task_id, plans["headline_seo"])
    for i, act_name in enumerate(plan):
        action = Action(action_type=ActionType(act_name))
        obs, reward, done, info = env.step(action)
        lines.append(
            f"[STEP] step={i+1} action={act_name} "
            f"reward={reward.total:.2f} done={str(done).lower()} error=null"
        )
        if done:
            break

    state = env.state()
    rewards_str = ",".join(f"{r:.2f}" for r in state["episode_rewards"])
    success = state["task_grade"] >= 0.7
    lines.append("")
    lines.append(
        f"[END] success={str(success).lower()} steps={state['step_count']} "
        f"score={state['task_grade']:.3f} rewards={rewards_str}"
    )
    lines.append("")
    lines.append(f"Final Grade: {state['task_grade']:.3f}")
    lines.append(f"Cumulative Reward: {state['cumulative_reward']:+.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Content Optimization RL Environment",
) as demo:
    gr.Markdown(
        "# Content Optimization RL Environment\n"
        "An OpenEnv-compatible RL environment for iterative content optimization.\n"
        "Choose a task, then step through actions to improve the content."
    )

    with gr.Tab("Interactive"):
        with gr.Row():
            task_dd = gr.Dropdown(choices=TASKS, value="headline_seo", label="Task")
            reset_btn = gr.Button("Reset Environment", variant="primary")
        with gr.Row():
            action_dd = gr.Dropdown(choices=ACTIONS, value="rewrite_headline", label="Action")
            step_btn = gr.Button("Step", variant="secondary")
        output_box = gr.Textbox(label="Output", lines=25, max_lines=40)

        reset_btn.click(fn=reset_env, inputs=task_dd, outputs=output_box)
        step_btn.click(fn=step_env, inputs=action_dd, outputs=output_box)

    with gr.Tab("Auto-Run (Heuristic)"):
        gr.Markdown("Run a full episode with the built-in heuristic agent.")
        with gr.Row():
            auto_task = gr.Dropdown(choices=TASKS, value="headline_seo", label="Task")
            auto_btn = gr.Button("Run Full Episode", variant="primary")
        auto_output = gr.Textbox(label="Episode Log", lines=25, max_lines=40)
        auto_btn.click(fn=run_full_episode, inputs=auto_task, outputs=auto_output)

    with gr.Tab("Custom Input"):
        gr.Markdown("Paste your own content, set goals, and optimize step-by-step.")
        custom_text = gr.Textbox(
            label="Input Content",
            lines=10,
            value="Announcing our new platform update. It has many features and should help teams work better.",
        )
        with gr.Row():
            custom_keywords = gr.Textbox(
                label="Target Keywords (comma separated)",
                value="productivity,workflow,automation,team collaboration",
            )
            custom_tone = gr.Dropdown(
                choices=["professional", "informative", "empathetic"],
                value="professional",
                label="Target Tone",
            )
            custom_max_steps = gr.Slider(minimum=1, maximum=15, value=10, step=1, label="Max Steps")
        with gr.Row():
            custom_reset_btn = gr.Button("Initialize Custom Session", variant="primary")
            custom_action = gr.Dropdown(choices=ACTIONS, value="rewrite_headline", label="Manual Action")
            custom_step_btn = gr.Button("Apply Action", variant="secondary")
            custom_auto_btn = gr.Button("Auto Optimize", variant="secondary")

        custom_summary = gr.Textbox(label="Session Summary", lines=8)
        custom_rationale = gr.Textbox(label="Reasoning / Action Log", lines=6)
        with gr.Row():
            custom_before = gr.Textbox(label="Before", lines=12)
            custom_after = gr.Textbox(label="After", lines=12)
        custom_diff = gr.Textbox(label="Unified Diff", lines=14)

        custom_reset_btn.click(
            fn=custom_reset_env,
            inputs=[custom_text, custom_keywords, custom_tone, custom_max_steps],
            outputs=[custom_summary, custom_before, custom_after, custom_diff, custom_rationale],
        )
        custom_step_btn.click(
            fn=custom_step_env,
            inputs=custom_action,
            outputs=[custom_summary, custom_before, custom_after, custom_diff, custom_rationale],
        )
        custom_auto_btn.click(
            fn=custom_auto_optimize,
            inputs=[],
            outputs=[custom_summary, custom_before, custom_after, custom_diff, custom_rationale],
        )

    with gr.Tab("About"):
        gr.Markdown(
            """
### Actions
| Action | Effect |
|---|---|
| `rewrite_headline` | Rewrites headline with target keywords |
| `add_keywords` | Inserts missing keywords into body |
| `improve_readability` | Breaks long sentences, simplifies vocab |
| `shorten_content` | Removes filler phrases |
| `change_tone` | Adjusts tone (professional/empathetic/informative) |
| `optimize_cta` | Adds strong call-to-action |
| `generate_reply` | Builds ORM-style reply |
| `no_op` | Does nothing |

### Tasks
- **headline_seo** (Easy) — Optimize a product headline for SEO
- **blog_readability** (Medium) — Improve a dense blog post
- **orm_reply** (Hard) — Generate an empathetic reply to a negative review

### Reward
Per-step reward = metric deltas + penalties for repetition/degradation

### API Endpoints
- `POST /reset` — Reset environment (body: `{"task_id": "headline_seo"}`)
- `POST /step` — Take action (body: `{"action_type": "rewrite_headline"}`)
- `GET /state` — Get full environment state
- `GET /health` — Health check
            """
        )


# ---------------------------------------------------------------------------
# Mount Gradio onto FastAPI and launch
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
