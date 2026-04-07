---
title: Content Optimization Env
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Content Optimization RL Environment

An **OpenEnv-compatible** reinforcement learning environment where an AI agent iteratively improves content (blog posts, social media copy, ORM replies) using step-by-step discrete actions and receives **per-step rewards** based on measurable quality improvements.

---

## 🏗️ Architecture

```
content-optimization-rl/
├── env/                        # Core environment logic
│   ├── environment.py          # Main RL env (reset / step / state)
│   ├── actions.py              # 8 deterministic action transformations
│   ├── reward.py               # Per-step reward computation
│   ├── graders.py              # Metric scorers + task graders
│   └── state.py                # Internal state & sample content
│
├── models/                     # Pydantic data models
│   ├── observation.py          # Observation schema
│   ├── action.py               # Action & ActionType enum
│   └── reward.py               # Reward schema (decomposed)
│
├── scripts/
│   └── run_baseline.py         # Baseline agent (OpenAI or heuristic)
│
├── openenv.yaml                # OpenEnv manifest
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the heuristic baseline (no API key needed)

```bash
python scripts/run_baseline.py
```

### 3. Run with OpenAI model

```bash
export OPENAI_API_KEY=sk-...
python scripts/run_baseline.py
```

---

## 🎮 How the RL Loop Works

```
┌──────────┐     action      ┌──────────────┐
│  Agent   │ ──────────────► │ Environment  │
│          │                 │              │
│          │ ◄────────────── │  step()      │
│          │  obs, reward,   │              │
│          │  done, info     │              │
└──────────┘                 └──────────────┘
```

### Episode lifecycle

1. **`reset()`** — environment creates fresh content for the selected task & computes initial scores
2. **`step(action)`** — agent picks an action → environment transforms content → recomputes metrics → returns reward
3. Repeat until **done** (max steps or quality threshold reached)
4. **`state()`** — inspect full internal state at any time

### Example

```python
from env.environment import ContentOptimizationEnv
from models.action import Action, ActionType

env = ContentOptimizationEnv(task_id="headline_seo")
obs = env.reset()

print(f"Initial SEO: {obs.seo_score:.3f}")

action = Action(action_type=ActionType.REWRITE_HEADLINE)
obs, reward, done, info = env.step(action)

print(f"Reward: {reward.total:+.4f}")
print(f"New SEO: {obs.seo_score:.3f}")
print(f"Grade: {info['task_grade']:.3f}")
```

---

## 📋 Action Space

| Action | Effect |
|---|---|
| `rewrite_headline` | Rewrites the headline with target keywords, optimises length |
| `add_keywords` | Inserts missing keywords naturally into the body |
| `improve_readability` | Breaks long sentences, simplifies vocabulary |
| `shorten_content` | Removes filler phrases and redundancies |
| `change_tone` | Adjusts tone toward the task's target (professional / empathetic / informative) |
| `optimize_cta` | Adds or replaces the call-to-action |
| `generate_reply` | Builds / enhances an ORM-style reply |
| `no_op` | Does nothing |

---

## 📊 Observation (State)

| Field | Type | Description |
|---|---|---|
| `original_content` | `str` | Unmodified content from reset |
| `current_draft` | `str` | Current version after actions |
| `seo_score` | `float [0,1]` | SEO quality |
| `readability_score` | `float [0,1]` | Readability quality |
| `engagement_score` | `float [0,1]` | Engagement quality |
| `sentiment_score` | `float [0,1]` | Sentiment alignment |
| `step_count` | `int` | Steps taken |
| `actions_taken` | `list[str]` | Action history |
| `task_id` | `str` | Current task identifier |

---

## 🎯 Reward Function

Per-step reward = **sum of metric deltas + penalties**

```
reward = (ΔSEO + ΔReadability + ΔEngagement + ΔSentiment)
       + repetition_penalty   (-0.05 if same action twice)
       + no_improvement_penalty (-0.02 if Δ ≈ 0)
       + degradation_penalty  (-0.10 if Δ < 0)
```

Rewards are **smooth and per-step** — not just a final score.

---

## 🏆 Tasks & Graders

### Task 1 – Headline SEO (Easy)
- **Goal:** Optimise a product headline for search engines
- **Grader:** keyword presence in headline (40%) + headline length (30%) + overall SEO (30%)
- **Success:** grade ≥ 0.8

### Task 2 – Blog Readability (Medium)
- **Goal:** Improve a dense blog post for readability & keyword coverage
- **Grader:** readability score (50%) + keyword density (30%) + paragraph structure (20%)
- **Success:** grade ≥ 0.8

### Task 3 – ORM Reply (Hard)
- **Goal:** Generate & refine an empathetic reply to a negative review
- **Grader:** sentiment alignment (40%) + helpfulness markers (30%) + tone correctness (30%)
- **Success:** grade ≥ 0.85

---

## ⏹️ Episode Termination

An episode ends when **either** condition is met:
- `step_count >= 10` (max steps)
- `composite_score >= 0.90` (quality threshold)

---

## 🔍 Scoring Details

All scoring is **deterministic** and **keyword/heuristic-based** — no heavy external APIs:

- **SEO:** keyword presence, headline length, keyword density, first-paragraph length
- **Readability:** avg sentence length, avg word length, paragraph count, jargon ratio
- **Engagement:** CTA phrases, active verbs, power words, content length
- **Sentiment:** tone-appropriate keywords, absence of negatives, politeness markers

---

## 📝 License

MIT
