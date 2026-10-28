"""Group words into readable subtitle lines.

The grouping algorithm is deliberately independent of the TTS engine.  It aims
for at most ``max_words`` words per line, but prefers to end a line after
sentence-ending punctuation, then clause punctuation, and respects explicit
newlines from the input text.  A conservative width estimate keeps lines inside
the video frame even when words are unusually long.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from brainrot_tts.models import WordTiming

_STRONG_END = re.compile(r"(?:[.!?…]+|[。！？]+)$")
_CLAUSE_END = re.compile(r"(?:[,;:，、；：]|—|–)$")
_CLOSERS = "\"'”’)]}»›"

# Common abbreviations that look like sentence endings to a regex.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "no",
    "inc",
    "ltd",
    "co",
    "approx",
    "dept",
}

_SPACE_WIDTH_EM = 0.28


@dataclass(slots=True)
class SubtitleLine:
    """A group of words shown together on screen."""

    words: list[WordTiming]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


def _strip_closers(text: str) -> str:
    return text.rstrip(_CLOSERS)


def is_sentence_end(word: WordTiming) -> bool:
    text = _strip_closers(word.text)
    if not text or not _STRONG_END.search(text):
        return False
    base = re.sub(r"(?:[.!?…]+|[。！？]+)$", "", text).strip().lower()
    if not base:
        return False
    if base in _ABBREVIATIONS:
        return False
    # Initials such as "J." and dotted abbreviations such as "U.S.".
    if len(base) == 1 and base.isalpha():
        return False
    return not re.fullmatch(r"(?:[a-z]\.)+[a-z]", base)


def is_clause_end(word: WordTiming) -> bool:
    text = _strip_closers(word.text)
    return bool(text) and bool(_CLAUSE_END.search(text))


def estimate_text_width_em(text: str) -> float:
    """Approximate rendered width in ``em`` units for a bold sans font.

    Overestimating is safer than underestimating: libass still performs smart
    wrapping as a final fallback, but most lines should already fit.
    """

    width = 0.0
    for char in text:
        category = unicodedata.category(char)
        if category in {"Mn", "Me", "Cf"}:
            continue
        if char.isspace():
            width += _SPACE_WIDTH_EM
            continue
        east_asian = unicodedata.east_asian_width(char)
        if east_asian in {"W", "F"}:
            width += 1.0
        elif char.isdigit():
            width += 0.55
        elif char.isalpha():
            width += 0.60 if char.isupper() else 0.50
        elif category.startswith("P"):
            width += 0.30
        else:
            width += 0.60
    return width


def _width_budget_em(max_width_px: float | None, font_size: float | None) -> float | None:
    if max_width_px is None or not font_size or font_size <= 0:
        return None
    # A small safety margin for bold glyphs, kerning differences and font
    # substitution (e.g. Arial -> Nimbus Sans).
    return max_width_px / font_size


def group_words(
    words: Sequence[WordTiming],
    *,
    max_words: int = 4,
    max_width_px: float | None = None,
    font_size: float | None = None,
    max_line_seconds: float | None = 8.0,
) -> list[SubtitleLine]:
    """Split words into display lines.

    ``max_width_px`` and ``font_size`` enable width-aware grouping.  When
    omitted only the word count (and punctuation/newlines) is considered.
    """

    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    remaining = [word for word in words if word.text]
    if not remaining:
        return []

    budget_em = _width_budget_em(max_width_px, font_size)
    lines: list[SubtitleLine] = []
    index = 0
    while index < len(remaining):
        limit = min(len(remaining), index + max_words)

        # Never cross a hard newline from the source text.
        for candidate in range(index + 1, limit):
            if remaining[candidate].break_before:
                limit = candidate
                break

        # Width-aware limit.
        take = limit - index
        if budget_em is not None:
            used = 0.0
            width_take = 0
            for candidate in range(index, limit):
                word_width = estimate_text_width_em(remaining[candidate].text)
                if width_take:
                    word_width += _SPACE_WIDTH_EM
                if width_take and used + word_width > budget_em:
                    break
                used += word_width
                width_take += 1
            take = max(1, width_take)

        # Duration-aware limit: very slow speech should not leave one line on
        # screen for an absurdly long time if a nearby word boundary exists.
        if max_line_seconds is not None and take > 1:
            start = remaining[index].start
            for candidate in range(index + 1, index + take):
                if remaining[candidate].start - start > max_line_seconds:
                    take = candidate - index
                    break

        # Prefer the last sentence boundary inside the window, as long as it
        # leaves at least two words.  This turns "Hello, world! This is" into
        # "Hello, world!" + "This is".
        chosen: int | None = None
        for candidate in range(index + take - 1, index - 1, -1):
            if is_sentence_end(remaining[candidate]):
                chosen = candidate - index + 1
                break
        if chosen is None:
            # Fall back to clause punctuation near the end of the window.
            for candidate in range(index + take - 1, index + 1, -1):
                if is_clause_end(remaining[candidate]):
                    chosen = candidate - index + 1
                    break
        if chosen is not None:
            take = chosen

        lines.append(SubtitleLine(list(remaining[index : index + take])))
        index += take

    return lines
