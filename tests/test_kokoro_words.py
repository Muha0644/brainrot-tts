from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from brainrot_tts.tts.kokoro import (
    _estimate_word_timings,
    _WordDraft,
    draft_words_from_tokens,
    materialize_words,
    split_plain_words,
)


def tok(text: str, whitespace: str, start: float | None, end: float | None):
    return SimpleNamespace(text=text, whitespace=whitespace, start_ts=start, end_ts=end)


def test_tokens_to_words_attaches_punctuation_and_newlines() -> None:
    tokens = [
        tok('"', "", 0.00, 0.05),
        tok("Hello", "", 0.05, 0.40),
        tok(",", " ", 0.40, 0.45),
        tok("world", "", 0.45, 0.80),
        tok("!", " ", 0.80, 0.85),
        tok("\n\nNew", " ", 0.85, 1.10),
        tok("paragraph", " ", 1.10, 1.40),
        tok("—", " ", 1.40, 1.50),
        tok("with", " ", 1.50, 1.70),
    ]
    drafts = draft_words_from_tokens(tokens)
    assert [d.text for d in drafts] == ['"Hello,', "world!", "New", "paragraph —", "with"]
    assert drafts[2].break_before is True
    assert drafts[0].start == pytest.approx(0.0)
    assert drafts[0].end == pytest.approx(0.45)
    assert drafts[3].start == pytest.approx(1.10)
    assert drafts[3].end == pytest.approx(1.50)


def test_tokens_do_not_leak_newlines_into_words() -> None:
    drafts = draft_words_from_tokens([tok("\n\nNew", " ", 0.0, 0.2), tok("line", " ", 0.2, 0.4)])
    assert [d.text for d in drafts] == ["New", "line"]
    assert drafts[0].break_before is True


def test_offset_is_applied_to_token_timings() -> None:
    drafts = draft_words_from_tokens([tok("Hello", " ", 0.7, 1.0)], offset=10.0)
    assert drafts[0].start == pytest.approx(10.7)
    assert drafts[0].end == pytest.approx(11.0)


def test_estimate_fallback_distributes_audio_duration() -> None:
    drafts = [_WordDraft("one", None, None, False), _WordDraft("three", None, None, False)]
    words = _estimate_word_timings(drafts, offset=5.0, duration=2.0)
    assert words[0].start == pytest.approx(5.0)
    assert words[0].end == pytest.approx(words[1].start)
    assert words[1].end == pytest.approx(7.0)
    assert words[1].start >= words[0].end


def test_materialize_clamps_overlaps_and_keeps_order() -> None:
    drafts = [
        _WordDraft("a", 0.0, 0.5, False),
        _WordDraft("b", 0.4, 0.9, False),
    ]
    result = materialize_words(drafts, 0.0, 1.0)
    assert result[1].start >= result[0].end
    assert result[1].end >= result[1].start


def test_split_plain_words() -> None:
    assert split_plain_words("one  two\nthree") == [
        ("one", False),
        ("two", False),
        ("three", True),
    ]


def test_voice_validation() -> None:
    from brainrot_tts.errors import TTSError
    from brainrot_tts.models import TTSOptions
    from brainrot_tts.tts.kokoro import KokoroTTS

    backend = KokoroTTS()
    backend.validate(TTSOptions(voice="af_heart", language="en-us", speed=1.0))
    with pytest.raises(TTSError):
        backend.validate(TTSOptions(voice="not a voice", language="en-us"))
    with pytest.raises(TTSError, match="Unknown Kokoro voice"):
        backend.validate(TTSOptions(voice="af_not_a_voice", language="en-us"))
    with pytest.raises(TTSError):
        backend.validate(TTSOptions(voice="af_heart", language="xx"))
    with pytest.raises(TTSError):
        backend.validate(TTSOptions(voice="af_heart", language="en-us", speed=0))


def test_huggingface_cache_root_env(tmp_path, monkeypatch) -> None:
    from brainrot_tts.tts.kokoro import _huggingface_cache_root

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert _huggingface_cache_root() == tmp_path / "hub"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert _huggingface_cache_root() == tmp_path / "hfhome" / "hub"
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _huggingface_cache_root() == tmp_path / "xdg" / "huggingface" / "hub"


def test_offline_mode_enabled_only_when_all_files_cached(tmp_path, monkeypatch) -> None:
    from brainrot_tts.tts.kokoro import KokoroTTS

    cache = tmp_path / "hub"
    repo = cache / "models--hexgrad--Kokoro-82M"
    snapshot = repo / "snapshots" / "abc123"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snapshot / "voices").mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "kokoro-v1_0.pth").write_bytes(b"model")
    (snapshot / "voices" / "af_heart.pt").write_bytes(b"voice")

    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    backend = KokoroTTS()
    backend._enable_offline_if_cached("af_heart")
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    monkeypatch.delenv("HF_HUB_OFFLINE")
    backend._enable_offline_if_cached("af_missing")
    assert "HF_HUB_OFFLINE" not in os.environ
