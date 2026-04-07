"""
Graders & scoring functions for the Content Optimization RL Environment.

Provides:
- Four metric scorers (SEO, readability, engagement, sentiment)
- Three task-specific graders that return a 0.0 → 1.0 score
- A utility to recompute *all* metrics on a ContentState

All scorers are deterministic and keyword/heuristic-based — no external
API calls required.
"""

from __future__ import annotations

import math
import re
from typing import List

from env.state import ContentState


# ====================================================================
#  Individual metric scorers (0 → 1)
# ====================================================================

def score_seo(text: str, target_keywords: List[str]) -> float:
    """
    SEO score based on:
      - keyword presence (50 %)
      - headline length 40-65 chars (20 %)
      - keyword density 1-3 % (20 %)
      - meta-description-length first paragraph (10 %)
    """
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    word_count = max(len(words), 1)

    # -- keyword presence --
    found = sum(1 for kw in target_keywords if kw.lower() in lower)
    kw_ratio = found / max(len(target_keywords), 1)

    # -- headline length (first line) --
    headline = text.split("\n", 1)[0].strip()
    hl_len = len(headline)
    if 40 <= hl_len <= 65:
        hl_score = 1.0
    elif 30 <= hl_len < 40 or 65 < hl_len <= 80:
        hl_score = 0.6
    else:
        hl_score = 0.2

    # -- keyword density --
    kw_word_hits = sum(
        lower.count(kw.lower()) for kw in target_keywords
    )
    density = kw_word_hits / word_count * 100  # percent
    if 1.0 <= density <= 3.0:
        density_score = 1.0
    elif 0.5 <= density < 1.0 or 3.0 < density <= 5.0:
        density_score = 0.5
    else:
        density_score = 0.1

    # -- first paragraph length (≈ meta description proxy) --
    first_para = text.split("\n\n")[0]
    fp_len = len(first_para)
    if 120 <= fp_len <= 160:
        meta_score = 1.0
    elif 80 <= fp_len < 120 or 160 < fp_len <= 200:
        meta_score = 0.6
    else:
        meta_score = 0.2

    raw = 0.50 * kw_ratio + 0.20 * hl_score + 0.20 * density_score + 0.10 * meta_score
    return round(min(max(raw, 0.0), 1.0), 4)


def score_readability(text: str) -> float:
    """
    Simplified readability score:
      - average sentence length (target: 12-20 words) — 40 %
      - average word length (target: 4-6 chars) — 20 %
      - paragraph count (>= 3 is good) — 20 %
      - no excessive jargon / long words — 20 %
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text)

    if not sentences or not words:
        return 0.1

    avg_sent_len = len(words) / len(sentences)
    if 12 <= avg_sent_len <= 20:
        sent_score = 1.0
    elif 8 <= avg_sent_len < 12 or 20 < avg_sent_len <= 28:
        sent_score = 0.5
    else:
        sent_score = 0.15

    avg_word_len = sum(len(w) for w in words) / len(words)
    if 4 <= avg_word_len <= 6:
        word_score = 1.0
    elif 3 <= avg_word_len < 4 or 6 < avg_word_len <= 8:
        word_score = 0.5
    else:
        word_score = 0.15

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_score = min(len(paragraphs) / 3.0, 1.0)

    long_words = [w for w in words if len(w) > 12]
    jargon_ratio = len(long_words) / len(words)
    jargon_score = max(1.0 - jargon_ratio * 10, 0.0)

    raw = 0.40 * sent_score + 0.20 * word_score + 0.20 * para_score + 0.20 * jargon_score
    return round(min(max(raw, 0.0), 1.0), 4)


def score_engagement(text: str) -> float:
    """
    Engagement heuristics:
      - has a call-to-action phrase (25 %)
      - uses active voice indicators (25 %)
      - uses power/action words (25 %)
      - content length is substantial but not excessive (25 %)
    """
    lower = text.lower()

    # CTA phrases
    cta_phrases = [
        "get started", "sign up", "learn more", "download", "try",
        "request", "explore", "discover", "join", "subscribe",
        "contact", "reach out", "demo", "free guide", "take the next step",
    ]
    cta_hits = sum(1 for p in cta_phrases if p in lower)
    cta_score = min(cta_hits / 2.0, 1.0)

    # Active voice — presence of strong verbs
    active_verbs = [
        "transform", "boost", "drive", "create", "build", "deliver",
        "achieve", "improve", "enhance", "empower", "accelerate",
        "resolve", "implement", "discover", "unlock",
    ]
    verb_hits = sum(1 for v in active_verbs if v in lower)
    active_score = min(verb_hits / 3.0, 1.0)

    # Power words
    power_words = [
        "proven", "exclusive", "essential", "ultimate", "powerful",
        "breakthrough", "guaranteed", "remarkable", "actionable",
        "comprehensive", "invaluable", "dedicated", "committed",
    ]
    power_hits = sum(1 for w in power_words if w in lower)
    power_score = min(power_hits / 2.0, 1.0)

    # Length (sweet spot: 100-400 words)
    word_count = len(re.findall(r"\b\w+\b", text))
    if 100 <= word_count <= 400:
        length_score = 1.0
    elif 50 <= word_count < 100 or 400 < word_count <= 600:
        length_score = 0.6
    else:
        length_score = 0.2

    raw = 0.25 * cta_score + 0.25 * active_score + 0.25 * power_score + 0.25 * length_score
    return round(min(max(raw, 0.0), 1.0), 4)


def score_sentiment(text: str, target_tone: str) -> float:
    """
    Sentiment alignment via keyword signals:
      - tone-appropriate words (50 %)
      - absence of negative/unwanted indicators (30 %)
      - overall politeness markers (20 %)
    """
    lower = text.lower()

    # Tone-specific positive indicators
    tone_words = {
        "empathetic": [
            "apologize", "sorry", "understand", "frustrating", "regret",
            "care", "concern", "committed", "resolve", "right",
            "feedback", "improve", "valued", "support",
        ],
        "professional": [
            "professional", "effective", "efficient", "solution",
            "implement", "strategy", "optimize", "results", "performance",
            "deliver", "quality", "excellence",
        ],
        "informative": [
            "learn", "discover", "understand", "guide", "research",
            "data", "evidence", "insight", "knowledge", "comprehensive",
            "analysis", "strategy", "approach",
        ],
    }

    words_for_tone = tone_words.get(target_tone, tone_words["professional"])
    tone_hits = sum(1 for w in words_for_tone if w in lower)
    tone_score = min(tone_hits / 4.0, 1.0)

    # Negative indicators (bad in any tone)
    negative = [
        "terrible", "awful", "worst", "hate", "stupid", "idiot",
        "never buy", "rip off", "scam",
    ]
    neg_hits = sum(1 for n in negative if n in lower)
    neg_score = max(1.0 - neg_hits * 0.3, 0.0)

    # Politeness
    polite = ["please", "thank", "appreciate", "grateful", "welcome", "kindly"]
    polite_hits = sum(1 for p in polite if p in lower)
    polite_score = min(polite_hits / 2.0, 1.0)

    raw = 0.50 * tone_score + 0.30 * neg_score + 0.20 * polite_score
    return round(min(max(raw, 0.0), 1.0), 4)


# ====================================================================
#  Recompute all metrics for a ContentState (mutates in place)
# ====================================================================

def recompute_metrics(state: ContentState) -> None:
    """Recompute and update all four score fields on *state*."""
    state.seo_score = score_seo(state.current_draft, state.target_keywords)
    state.readability_score = score_readability(state.current_draft)
    state.engagement_score = score_engagement(state.current_draft)
    state.sentiment_score = score_sentiment(
        state.current_draft, state.target_tone
    )


# ====================================================================
#  Task-level graders (0 → 1)
# ====================================================================

def grade_task(state: ContentState) -> float:
    """
    Return a deterministic 0.0 → 1.0 grade for the current task
    based on task-specific success criteria.
    """
    graders = {
        "headline_seo": _grade_headline_seo,
        "blog_readability": _grade_blog_readability,
        "orm_reply": _grade_orm_reply,
    }
    grader = graders.get(state.task_id, _grade_default)
    return grader(state)


# --- Task 1: headline SEO -------------------------------------------------

def _grade_headline_seo(state: ContentState) -> float:
    """
    Criteria:
      - keyword presence in headline (40 %)
      - headline length 40-65 chars    (30 %)
      - overall SEO score              (30 %)
    """
    headline = state.current_draft.split("\n", 1)[0].strip().lower()

    # keyword presence in headline
    found_in_hl = sum(
        1 for kw in state.target_keywords if kw.lower() in headline
    )
    kw_hl_score = min(found_in_hl / max(min(len(state.target_keywords), 3), 1), 1.0)

    # headline length
    hl_len = len(headline)
    if 40 <= hl_len <= 65:
        len_score = 1.0
    elif 30 <= hl_len < 40 or 65 < hl_len <= 80:
        len_score = 0.5
    else:
        len_score = 0.1

    grade = 0.40 * kw_hl_score + 0.30 * len_score + 0.30 * state.seo_score
    return round(min(max(grade, 0.0), 1.0), 4)


# --- Task 2: blog readability ---------------------------------------------

def _grade_blog_readability(state: ContentState) -> float:
    """
    Criteria:
      - readability score          (50 %)
      - keyword density 1-3 %     (30 %)
      - paragraph structure        (20 %)
    """
    lower = state.current_draft.lower()
    words = re.findall(r"\b\w+\b", lower)
    word_count = max(len(words), 1)

    kw_hits = sum(lower.count(kw.lower()) for kw in state.target_keywords)
    density = kw_hits / word_count * 100
    if 1.0 <= density <= 3.0:
        density_score = 1.0
    elif 0.5 <= density < 1.0 or 3.0 < density <= 5.0:
        density_score = 0.5
    else:
        density_score = 0.1

    paragraphs = [p.strip() for p in state.current_draft.split("\n\n") if p.strip()]
    para_score = min(len(paragraphs) / 4.0, 1.0)

    grade = (
        0.50 * state.readability_score
        + 0.30 * density_score
        + 0.20 * para_score
    )
    return round(min(max(grade, 0.0), 1.0), 4)


# --- Task 3: ORM reply ----------------------------------------------------

def _grade_orm_reply(state: ContentState) -> float:
    """
    Criteria:
      - sentiment alignment     (40 %)
      - helpfulness markers     (30 %)
      - tone correctness        (30 %)
    """
    lower = state.current_draft.lower()

    # Helpfulness: does it offer resolution?
    help_markers = [
        "resolve", "resolution", "escalat", "reach out", "contact",
        "24 hours", "support team", "follow up", "make things right",
    ]
    help_hits = sum(1 for m in help_markers if m in lower)
    help_score = min(help_hits / 3.0, 1.0)

    # Tone correctness: empathetic markers
    empathy_words = [
        "apologize", "sorry", "understand", "frustrat", "sincerely",
        "committed", "valued", "care",
    ]
    emp_hits = sum(1 for e in empathy_words if e in lower)
    tone_score = min(emp_hits / 3.0, 1.0)

    grade = (
        0.40 * state.sentiment_score
        + 0.30 * help_score
        + 0.30 * tone_score
    )
    return round(min(max(grade, 0.0), 1.0), 4)


# --- Fallback --------------------------------------------------------------

def _grade_default(state: ContentState) -> float:
    """Generic grader — simple composite score."""
    return state.composite_score()
