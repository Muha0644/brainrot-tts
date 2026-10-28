from __future__ import annotations

from pathlib import Path

import pytest

from brainrot_tts.errors import ConfigError
from brainrot_tts.models import VideoInfo
from brainrot_tts.pipeline import (
    PipelineConfig,
    _default_font_size,
    _read_input_text,
    _resolve_style,
    _resolve_video_offset,
)


def test_default_font_size_scales_with_height() -> None:
    assert _default_font_size(1080) == 67
    assert _default_font_size(1920) == 119
    assert _default_font_size(100) == 24
    assert _default_font_size(10000) == 180


def test_random_offset_is_deterministic_with_seed() -> None:
    first = _resolve_video_offset(-1, 10.0, seed=42)
    second = _resolve_video_offset(-1, 10.0, seed=42)
    assert first == second
    assert 0 <= first < 10.0


def test_offset_validation() -> None:
    with pytest.raises(ConfigError):
        _resolve_video_offset(-2, 10.0, None)
    with pytest.raises(ConfigError):
        _resolve_video_offset(10.0, 10.0, None)
    assert _resolve_video_offset(0, 10.0, None) == 0


def test_resolve_style_auto_sizes_and_parses_colours() -> None:
    config = PipelineConfig(
        input_path=Path("in.txt"),
        video_path=Path("bg.mp4"),
        output_path=Path("out.mp4"),
        work_dir=Path("work"),
        background_opacity=1.0,
    )
    video = VideoInfo(Path("bg.mp4"), 1080, 1920, 10.0, 30.0)
    style = _resolve_style(config, video)
    assert style.font_size == 119
    assert style.margin_bottom == 154
    assert style.margin_lr == 43
    assert style.text_color == "&H00FFFFFF"
    assert style.highlight_color == "&H0000D4FF"
    assert style.box_color == "&H00000000"
    assert style.box is True


def test_read_input_text_rejects_empty_and_accepts_utf8(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ConfigError):
        _read_input_text(path)
    path.write_text("Café 日本語 😀", encoding="utf-8")
    assert _read_input_text(path) == "Café 日本語 😀"
    path.write_bytes(b"text\x00binary")
    with pytest.raises(ConfigError, match="NUL"):
        _read_input_text(path)


def test_ensure_tools_reports_missing_ffmpeg(monkeypatch) -> None:
    import shutil

    from brainrot_tts.video.ffmpeg import ensure_tools

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ConfigError, match="ffmpeg"):
        ensure_tools()


def test_ffmpeg_error_includes_command_and_stderr() -> None:
    from brainrot_tts.errors import FFmpegError

    error = FFmpegError(
        "FFmpeg failed",
        ["ffmpeg", "-i", "input file.mp4", "out.mp4"],
        returncode=1,
        stderr="line one\nUnknown encoder\n",
    )
    message = str(error)
    assert "exit code: 1" in message
    assert "ffmpeg -i 'input file.mp4' out.mp4" in message
    assert "Unknown encoder" in message
