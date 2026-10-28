from __future__ import annotations

from brainrot_tts.models import WordTiming
from brainrot_tts.subtitles.grouping import (
    estimate_text_width_em,
    group_words,
    is_clause_end,
    is_sentence_end,
)


def words(*texts: str, duration: float = 0.4) -> list[WordTiming]:
    return [WordTiming(text, i * duration, (i + 1) * duration) for i, text in enumerate(texts)]


def line_texts(lines: list) -> list[str]:
    return [line.text for line in lines]


def test_grouping_prefers_sentence_boundary() -> None:
    result = group_words(words("Hello,", "world!", "This", "is", "a", "test."), max_words=4)
    assert line_texts(result) == ["Hello, world!", "This is a test."]


def test_grouping_respects_word_limit() -> None:
    result = group_words(words("one", "two", "three", "four", "five", "six"), max_words=3)
    assert line_texts(result) == ["one two three", "four five six"]


def test_grouping_respects_hard_newline() -> None:
    ws = words("first", "line", "second", "line")
    ws[2].break_before = True
    result = group_words(ws, max_words=4)
    assert line_texts(result) == ["first line", "second line"]


def test_grouping_clause_and_abbreviation() -> None:
    assert is_sentence_end(WordTiming("world!", 0, 1))
    assert is_sentence_end(WordTiming("end.”", 0, 1))
    assert not is_sentence_end(WordTiming("Dr.", 0, 1))
    assert not is_sentence_end(WordTiming("J.", 0, 1))
    assert is_clause_end(WordTiming("word,", 0, 1))
    assert is_clause_end(WordTiming("word —", 0, 1))
    result = group_words(words("Dr.", "Smith", "went", "home.", "Then", "left."), max_words=4)
    assert line_texts(result) == ["Dr. Smith went home.", "Then left."]


def test_grouping_width_budget_prevents_long_lines() -> None:
    ws = words("supercalifragilisticexpialidocious", "extraordinarily", "long", "words")
    result = group_words(ws, max_words=4, max_width_px=300, font_size=60)
    assert len(result) >= 2
    assert all(line.words for line in result)


def test_grouping_single_word_and_cjk() -> None:
    assert line_texts(group_words(words("Hello."))) == ["Hello."]
    result = group_words(words("你好。", "世界！", "这是", "测试。"), max_words=2)
    assert line_texts(result) == ["你好。 世界！", "这是 测试。"]


def test_grouping_prefers_sentence_end_at_start_of_short_line() -> None:
    # The first word is a sentence end; a one-word line is better than mixing
    # it with the next sentence when width forced a break here.
    result = group_words(
        words("TTS.", "It", "should"),
        max_words=4,
        max_width_px=100,
        font_size=60,
        max_line_seconds=None,
    )
    assert line_texts(result)[0] == "TTS."


def test_width_estimate_monotonic() -> None:
    assert estimate_text_width_em("iiii") < estimate_text_width_em("WWWW")
    assert estimate_text_width_em("日本") > estimate_text_width_em("ab")
