from __future__ import annotations

import pytest

from brainrot_tts.errors import ConfigError, SubtitleError
from brainrot_tts.models import WordTiming
from brainrot_tts.subtitles.ass import (
    SubtitleStyle,
    escape_ass_text,
    format_ass_time,
    parse_color,
    render_ass,
)
from brainrot_tts.subtitles.grouping import SubtitleLine


def test_escape_preserves_override_like_text() -> None:
    assert escape_ass_text(r"{\an8}") == "\\{\\\u200ban8\\}"
    assert escape_ass_text(r"literal \N here") == "literal \\\u200bN here"
    assert escape_ass_text(r"a\b {c}") == "a\\\u200bb \\{c\\}"
    assert escape_ass_text("line\nbreak") == "line break"


def test_escape_keeps_unicode_and_punctuation() -> None:
    text = "“Café”, naïve — 日本語! 😀 3.14%"
    assert escape_ass_text(text) == text


def test_format_ass_time() -> None:
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(1.234) == "0:00:01.23"
    assert format_ass_time(61.239) == "0:01:01.24"
    assert format_ass_time(3661.999) == "1:01:02.00"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#FFFFFF", "&H00FFFFFF"),
        ("#ff0000", "&H000000FF"),
        ("#0f0", "&H0000FF00"),
        ("00ff00", "&H0000FF00"),
        ("&H80FF0000", "&H80FF0000"),
        ("&H00D4FF", "&H0000D4FF"),
        ("yellow", "&H0000FFFF"),
        ("black", "&H00000000"),
    ],
)
def test_parse_color(value: str, expected: str) -> None:
    assert parse_color(value) == expected


def test_parse_color_alpha_and_errors() -> None:
    assert parse_color("#FFFFFF", default_alpha=0x66) == "&H66FFFFFF"
    with pytest.raises(ConfigError):
        parse_color("not-a-colour")
    with pytest.raises(ConfigError):
        parse_color("")


def _style(**kwargs: object) -> SubtitleStyle:
    values: dict[str, object] = {
        "font": "Arial",
        "font_size": 60,
        "text_color": "&H00FFFFFF",
        "highlight_color": "&H0000D4FF",
        "outline_color": "&H00000000",
        "outline_width": 3,
        "box": False,
        "box_color": "&H00000000",
        "box_padding": 12,
        "position": "bottom",
        "margin_bottom": 80,
        "margin_lr": 30,
    }
    values.update(kwargs)
    return SubtitleStyle(**values)  # type: ignore[arg-type]


def test_render_ass_highlights_exactly_one_word() -> None:
    words = [
        WordTiming("The", 0.0, 0.3),
        WordTiming("system", 0.3, 0.7),
        WordTiming("works.", 0.7, 1.2),
    ]
    ass = render_ass(
        [SubtitleLine(words)],
        _style(),
        play_res_x=1080,
        play_res_y=1920,
        hold_seconds=0.2,
    )
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "BrainrotText" in ass
    assert "BrainrotBox" not in ass
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 3
    assert r"{\c&H00D4FF&}The{\c&HFFFFFF&} system works." in dialogues[0]
    assert r"The {\c&H00D4FF&}system{\c&HFFFFFF&} works." in dialogues[1]
    assert r"The system {\c&H00D4FF&}works.{\c&HFFFFFF&}" in dialogues[2]
    assert "0:00:00.00,0:00:00.30" in dialogues[0]
    assert "0:00:00.30,0:00:00.70" in dialogues[1]
    assert "0:00:00.70,0:00:01.40" in dialogues[2]


def test_render_ass_box_layer() -> None:
    words = [WordTiming("Hello", 0.0, 0.4), WordTiming("world", 0.4, 0.9)]
    ass = render_ass(
        [SubtitleLine(words)],
        _style(box=True, corner_radius=12),
        play_res_x=720,
        play_res_y=1280,
    )
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 3  # one box event + one text event per word
    assert dialogues[0].startswith("Dialogue: 0,")
    assert dialogues[1].startswith("Dialogue: 1,")
    assert "BrainrotBox" in ass
    # The box is an ASS vector drawing with a rounded rectangle path.
    assert "\\p1" in dialogues[0]
    assert "\\an7\\pos(" in dialogues[0]
    assert " b " in dialogues[0]
    assert dialogues[1].endswith(r"{\c&H00D4FF&}Hello{\c&HFFFFFF&} world")


def test_render_ass_heavy_weight_uses_outline_pass() -> None:
    words = [WordTiming("Hello", 0.0, 0.4), WordTiming("world", 0.4, 0.9)]
    ass = render_ass(
        [SubtitleLine(words)],
        _style(font_stroke=1.5),
        play_res_x=720,
        play_res_y=1280,
    )
    assert "BrainrotOutline" in ass
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    # one outline pass + one text event per word
    assert len(dialogues) == 3
    assert dialogues[0].startswith("Dialogue: 1,")
    assert dialogues[0].endswith("Hello world")
    assert dialogues[1].startswith("Dialogue: 2,")
    # Both fill and stroke colour switch for the active word.
    assert r"{\c&H00D4FF&\3c&H00D4FF&}Hello{\c&HFFFFFF&\3c&HFFFFFF&}" in dialogues[1]


def test_render_ass_position_and_escaping() -> None:
    style = _style(position="top", margin_bottom=55)
    ass = render_ass(
        [SubtitleLine([WordTiming(r"{\an8}", 0.0, 0.5)])],
        style,
        play_res_x=1920,
        play_res_y=1080,
    )
    assert ",8,30,30,55,1" in ass
    assert "\\{\\\u200ban8\\}" in ass


def test_render_ass_rejects_empty() -> None:
    with pytest.raises(SubtitleError):
        render_ass([], _style(), play_res_x=100, play_res_y=100)
