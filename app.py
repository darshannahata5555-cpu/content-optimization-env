"""
app.py — Gradio demo UI for the Content Optimization RL Environment.

This gives the HF Space a persistent web server so it stays "Running".
Users can interact with the environment through the browser.
"""

import io
import sys
import json

# Fix encoding on containers
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr

from env.environment import ContentOptimizationEnv
from models.action import Action, ActionType

# ---------------------------------------------------------------------------
# Global env instance (reset per session via button)
# ---------------------------------------------------------------------------
ENV = None


def reset_env(task_id: str) -> str:
    """Reset the environment with the selected task."""
    global ENV
    ENV = ContentOptimizationEnv(task_id=task_id, max_steps=10)
    obs = ENV.reset()
    state = ENV.state()

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
    global ENV
    if ENV is None:
        return "ERROR: Please reset the environment first!"

    if ENV._state.done:
        state = ENV.state()
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
    obs, reward, done, info = ENV.step(action)

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
        state = ENV.state()
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
# Gradio UI
# ---------------------------------------------------------------------------

TASKS = ["headline_seo", "blog_readability", "orm_reply"]
ACTIONS = [a.value for a in ActionType]

with gr.Blocks(
    title="Content Optimization RL Environment",
    theme=gr.themes.Soft(),
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
            """
        )

demo.launch(server_name="0.0.0.0", server_port=7860)
