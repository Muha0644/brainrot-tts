from __future__ import annotations

from pathlib import Path

import pytest

from brainrot_tts.errors import ConfigError
from brainrot_tts.video.ffmpeg import build_compose_command


def base_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "background": Path("/inputs/bg.mp4"),
        "audio": Path("/work/audio.wav"),
        "ass_filename": "subtitles.ass",
        "output": Path("/work/out.part.mp4"),
        "duration": 5.2,
        "offset": 1.5,
        "crf": 20,
        "preset": "medium",
    }
    values.update(overrides)
    return values


def test_compose_command_loops_trims_and_burns_ass() -> None:
    command = build_compose_command(**base_kwargs())  # type: ignore[arg-type]
    assert command[0] == "ffmpeg"
    loop_index = command.index("-stream_loop")
    input_index = command.index("-i")
    assert loop_index < input_index
    assert command[loop_index + 1] == "-1"
    assert command[command.index("-ss") + 1] == "1.500"
    assert command[command.index("-t") + 1] == "5.200"
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "ass=subtitles.ass" in filter_graph
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in filter_graph
    assert command[command.index("-map") + 1] == "[v]"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[-1] == "/work/out.part.mp4"
    assert command[-3:-1] == ["-f", "mp4"]
    assert "-nostdin" in command
    assert "+faststart" in command


def test_compose_command_rejects_bad_values() -> None:
    with pytest.raises(ConfigError):
        build_compose_command(**base_kwargs(preset="nope"))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        build_compose_command(**base_kwargs(crf=60))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        build_compose_command(**base_kwargs(duration=0))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        build_compose_command(**base_kwargs(offset=-1))  # type: ignore[arg-type]
