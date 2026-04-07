"""
Deterministic action transformations for the Content Optimization RL Environment.

Each function takes the current draft and state metadata and returns
a *new* draft string.  All transformations are deterministic —
no randomness, no API calls — making the environment fully reproducible.
"""

from __future__ import annotations

import re
import textwrap
from typing import List

from models.action import ActionType


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def apply_action(
    action_type: ActionType,
    draft: str,
    target_keywords: List[str],
    target_tone: str,
    parameters: dict | None = None,
) -> str:
    """Apply *action_type* to *draft* and return the modified text."""
    params = parameters or {}
    dispatch = {
        ActionType.REWRITE_HEADLINE: _rewrite_headline,
        ActionType.ADD_KEYWORDS: _add_keywords,
        ActionType.IMPROVE_READABILITY: _improve_readability,
        ActionType.SHORTEN_CONTENT: _shorten_content,
        ActionType.CHANGE_TONE: _change_tone,
        ActionType.OPTIMIZE_CTA: _optimize_cta,
        ActionType.GENERATE_REPLY: _generate_reply,
        ActionType.NO_OP: _no_op,
    }
    handler = dispatch[action_type]
    return handler(draft, target_keywords, target_tone, params)


# ---------------------------------------------------------------------------
# Individual action implementations
# ---------------------------------------------------------------------------

def _rewrite_headline(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Rewrite the first line (headline) to include target keywords and
    optimize length for SEO (50-60 chars ideal)."""
    lines = draft.split("\n", 1)
    headline = lines[0].strip()
    body = lines[1] if len(lines) > 1 else ""

    # Build a better headline from keywords
    primary_kw = keywords[:3] if keywords else ["content"]
    new_headline = f"Top {primary_kw[0].title()}"
    if len(primary_kw) > 1:
        new_headline += f" {primary_kw[1].title()}"
    if len(primary_kw) > 2:
        new_headline += f" & {primary_kw[2].title()}"
    new_headline += ": A Complete Guide for 2025"

    return new_headline + "\n" + body


def _add_keywords(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Sprinkle missing target keywords naturally throughout the content."""
    lower_draft = draft.lower()
    missing = [kw for kw in keywords if kw.lower() not in lower_draft]

    if not missing:
        return draft  # all keywords already present

    # Append a short paragraph that weaves in missing keywords
    kw_sentence_parts = []
    for i, kw in enumerate(missing[:4]):
        if i == 0:
            kw_sentence_parts.append(
                f"This guide covers essential aspects of {kw}"
            )
        else:
            kw_sentence_parts.append(f"{kw}")

    addition = ", ".join(kw_sentence_parts) + " to help you succeed."
    paragraphs = draft.rstrip().split("\n\n")
    # Insert before the last paragraph
    if len(paragraphs) > 1:
        paragraphs.insert(-1, addition)
    else:
        paragraphs.append(addition)

    return "\n\n".join(paragraphs)


def _improve_readability(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Break long sentences, simplify vocabulary, add paragraph breaks."""
    # Split overly long sentences (> 30 words) at commas or conjunctions
    sentences = re.split(r"(?<=[.!?])\s+", draft)
    improved: list[str] = []

    for sent in sentences:
        words = sent.split()
        if len(words) > 30:
            # Try to split at a conjunction near the middle
            mid = len(words) // 2
            split_point = None
            for offset in range(6):
                for idx in (mid + offset, mid - offset):
                    if 0 <= idx < len(words) and words[idx].lower() in (
                        "and", "but", "or", "that", "which", "while",
                        "because", "although", "however", "therefore",
                    ):
                        split_point = idx
                        break
                if split_point is not None:
                    break

            if split_point and split_point > 3:
                first_half = " ".join(words[:split_point]).rstrip(",") + "."
                second_half = " ".join(words[split_point:])
                # Capitalise start of new sentence
                if second_half and second_half[0].islower():
                    second_half = second_half[0].upper() + second_half[1:]
                improved.append(first_half)
                improved.append(second_half)
            else:
                improved.append(sent)
        else:
            improved.append(sent)

    # Simple vocabulary replacements
    text = " ".join(improved)
    simplifications = {
        "necessitating": "requiring",
        "fundamentally inadequate": "not enough",
        "comprehensive overhaul": "complete update",
        "leveraging": "using",
        "utilize": "use",
        "commence": "start",
        "terminate": "end",
        "endeavor": "effort",
        "facilitate": "help",
        "ascertain": "find out",
    }
    for old, new in simplifications.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Add paragraph breaks every ~80 words if content is a single block
    paragraphs = text.split("\n\n")
    final_paragraphs = []
    for para in paragraphs:
        words = para.split()
        if len(words) > 80:
            chunks = [words[i: i + 60] for i in range(0, len(words), 60)]
            final_paragraphs.extend(" ".join(c) for c in chunks)
        else:
            final_paragraphs.append(para)

    return "\n\n".join(final_paragraphs)


def _shorten_content(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Remove filler phrases, redundancies, and trim to essentials."""
    fillers = [
        "very ", "really ", "just ", "actually ", "basically ",
        "in order to ", "as a matter of fact, ", "it goes without saying that ",
        "at the end of the day, ", "for all intents and purposes, ",
        "it is important to note that ", "it should be noted that ",
    ]
    text = draft
    for filler in fillers:
        text = re.sub(re.escape(filler), "", text, flags=re.IGNORECASE)

    # Trim excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _change_tone(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Adjust the tone toward the target (professional / empathetic / informative)."""
    text = draft

    if tone == "empathetic":
        # Add empathetic openers to reply-style content
        if not text.lower().startswith("we sincerely"):
            empathetic_opener = (
                "We sincerely apologize for your experience. "
                "We understand how frustrating this must be, and we want to "
                "make things right. "
            )
            # If it looks like a reply (starts after a quote), prepend to reply part
            if "Draft Reply:" in text:
                parts = text.split("Draft Reply:", 1)
                text = parts[0] + "Draft Reply: " + empathetic_opener + parts[1].strip().lstrip("'\"")
            else:
                text = empathetic_opener + text

        # Replace dismissive language
        text = text.replace("Sorry about that.", "")
        text = text.replace("sorry about that.", "")
        text = re.sub(r"\s{2,}", " ", text)

    elif tone == "professional":
        # Remove overly casual language
        casual = {
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "have to",
            "kinda": "somewhat",
            "lots of": "numerous",
            "awesome": "excellent",
            "cool": "effective",
        }
        for old, new in casual.items():
            text = re.sub(r"\b" + re.escape(old) + r"\b", new, text, flags=re.IGNORECASE)

    elif tone == "informative":
        # Add transitional phrases for an educational feel
        transitions = [
            ("First,", "First and foremost,"),
            ("Also,", "Additionally,"),
            ("So,", "Therefore,"),
        ]
        for old, new in transitions:
            text = text.replace(old, new)

    return text.strip()


def _optimize_cta(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Add or improve a call-to-action at the end of the content."""
    # Remove any existing weak CTA
    weak_ctas = [
        "Buy it now from our website.",
        "Click here.",
        "Learn more.",
        "Buy now.",
    ]
    text = draft
    for cta in weak_ctas:
        text = text.replace(cta, "")

    text = text.rstrip()

    # Add a strong, tone-appropriate CTA
    if tone == "empathetic":
        cta = (
            "\n\nPlease reach out to our dedicated support team at your "
            "convenience — we are committed to resolving this and restoring "
            "your confidence in our service."
        )
    elif tone == "informative":
        cta = (
            "\n\nReady to take the next step? Download our free guide and "
            "discover actionable strategies you can implement today."
        )
    else:  # professional / default
        cta = (
            "\n\nGet started today — explore our solutions and see how they "
            "can transform your workflow. Request a free demo now."
        )

    return text + cta


def _generate_reply(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Generate or enhance an ORM-style reply within the draft."""
    if "Draft Reply:" in draft:
        parts = draft.split("Draft Reply:", 1)
        review_section = parts[0]
        current_reply = parts[1].strip().strip("'\"")
    else:
        review_section = ""
        current_reply = draft

    # Build a proper reply
    reply_parts = []

    if "apologize" not in current_reply.lower() and "sorry" not in current_reply.lower():
        reply_parts.append(
            "Thank you for taking the time to share your feedback."
        )
        reply_parts.append(
            "We sincerely apologize for the inconvenience you experienced."
        )

    if "resolution" not in current_reply.lower() and "resolve" not in current_reply.lower():
        reply_parts.append(
            "We have escalated your case to our senior support team, and "
            "they will reach out to you within 24 hours with a resolution."
        )

    if "improve" not in current_reply.lower():
        reply_parts.append(
            "Your feedback is invaluable — it helps us identify areas for "
            "improvement so we can serve you and all our customers better."
        )

    if reply_parts:
        enhanced_reply = " ".join(reply_parts)
        if current_reply and current_reply != "Sorry about that.":
            enhanced_reply = current_reply.rstrip(".") + ". " + enhanced_reply
    else:
        enhanced_reply = current_reply

    if review_section:
        return review_section + "Draft Reply: " + enhanced_reply
    return enhanced_reply


def _no_op(
    draft: str, keywords: List[str], tone: str, params: dict
) -> str:
    """Do nothing — return the draft unchanged."""
    return draft
