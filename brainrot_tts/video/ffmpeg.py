"""FFmpeg/FFprobe wrappers for probing media and composing the final MP4.

No Python video-rendering library is involved: subtitles are burned in by
FFmpeg's ``ass`` filter, and the background is looped/trimmed with
``-stream_loop`` + ``-t``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from brainrot_tts.errors import ConfigError, FFmpegError
from brainrot_tts.models import VideoInfo

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: Resize odd dimensions by at most one pixel; libx264/yuv420p needs even
#: dimensions.  Aspect ratio is preserved (no letterboxing/pillarboxing).
_EVEN_SCALE = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

X264_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


def ensure_tools() -> None:
    """Fail early with an actionable message if FFmpeg is not installed."""

    missing = [tool for tool in (FFMPEG, FFPROBE) if shutil.which(tool) is None]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(
            f"Required command(s) not found: {joined}. Install FFmpeg (which includes "
            "ffprobe) and make sure it is on PATH. Examples: "
            "`sudo apt install ffmpeg`, `brew install ffmpeg`, or download from "
            "https://ffmpeg.org/download.html."
        )


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    description: str,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command without a shell and raise :class:`FFmpegError` on failure."""

    logger.debug("Running: %s", shlex.join(command))
    try:
        result = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise ConfigError(f"Could not execute {command[0]!r}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not execute {command[0]!r}: {exc}") from exc

    if verbose and result.stderr.strip():
        logger.debug("Command stderr:\n%s", result.stderr.strip())
    if result.returncode != 0:
        raise FFmpegError(
            description,
            command,
            returncode=result.returncode,
            stderr=result.stderr,
        )
    return result


def _parse_rate(value: str | None) -> float | None:
    if not value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        num = float(numerator)
        den = float(denominator) if denominator else 1.0
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num / den


def _stream_rotation(stream: dict[str, object]) -> int:
    rotation = 0
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                with contextlib.suppress(TypeError, ValueError):
                    rotation = int(round(float(item["rotation"])))
    tags = stream.get("tags")
    if rotation == 0 and isinstance(tags, dict):
        rotate = tags.get("rotate")
        if rotate is not None:
            with contextlib.suppress(TypeError, ValueError):
                rotation = int(round(float(rotate)))
    return rotation % 360


def probe_video(path: Path) -> VideoInfo:
    """Read dimensions, duration, frame rate and rotation of a video."""

    path = path.expanduser()
    command = [
        FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = run_command(command, description=f"FFprobe could not read video: {path}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(
            f"FFprobe returned invalid JSON for {path}", command, stderr=result.stdout
        ) from exc
    if not isinstance(data, dict):
        raise FFmpegError(f"FFprobe returned unexpected JSON for {path}", command)

    streams = [stream for stream in (data.get("streams") or []) if isinstance(stream, dict)]
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ConfigError(f"{path} does not contain a video stream.")

    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Could not determine video dimensions for {path}.") from exc

    rotation = _stream_rotation(video_stream)
    if abs(rotation) % 180 == 90:
        width, height = height, width

    duration_value = (data.get("format") or {}).get("duration")
    if duration_value in (None, "N/A"):
        duration_value = video_stream.get("duration")
    try:
        duration = float(str(duration_value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Could not determine video duration for {path}.") from exc
    if duration <= 0:
        raise ConfigError(f"Video duration is zero or invalid for {path}.")

    fps = _parse_rate(video_stream.get("avg_frame_rate")) or _parse_rate(
        video_stream.get("r_frame_rate")
    )

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        rotation=rotation,
    )


def probe_duration(path: Path) -> float:
    """Return a media file's duration in seconds."""

    command = [
        FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration:stream=duration",
        str(path),
    ]
    result = run_command(command, description=f"FFprobe could not read media: {path}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(
            f"FFprobe returned invalid JSON for {path}", command, stderr=result.stdout
        ) from exc
    if not isinstance(data, dict):
        raise FFmpegError(f"FFprobe returned unexpected JSON for {path}", command)

    candidates: list[object] = [(data.get("format") or {}).get("duration")]
    candidates.extend(
        stream.get("duration") for stream in (data.get("streams") or []) if isinstance(stream, dict)
    )
    for candidate in candidates:
        if candidate in (None, "N/A"):
            continue
        try:
            duration = float(str(candidate))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    raise ConfigError(f"Could not determine duration for {path}.")


def build_compose_command(
    *,
    background: Path,
    audio: Path,
    ass_filename: str,
    output: Path,
    duration: float,
    offset: float = 0.0,
    crf: int = 20,
    preset: str = "medium",
) -> list[str]:
    """Build the final FFmpeg argv (also used by tests)."""

    if preset not in X264_PRESETS:
        raise ConfigError(
            f"Invalid x264 preset {preset!r}. Choose one of: {', '.join(sorted(X264_PRESETS))}"
        )
    if not 0 <= crf <= 51:
        raise ConfigError("CRF must be between 0 and 51.")
    if duration <= 0:
        raise ConfigError("Output duration must be positive.")
    if offset < 0:
        raise ConfigError("Video offset must be zero or positive.")

    filter_graph = f"[0:v]{_EVEN_SCALE},ass={ass_filename}[v]"
    return [
        FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "warning",
        "-y",
        # -stream_loop must precede the input; -ss seeks into the (looped)
        # background so --video-offset also works on short clips.
        "-stream_loop",
        "-1",
        "-ss",
        f"{offset:.3f}",
        "-i",
        str(background),
        "-i",
        str(audio),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output),
    ]


def compose_video(
    *,
    background: Path,
    audio: Path,
    ass_path: Path,
    output: Path,
    duration: float,
    offset: float = 0.0,
    crf: int = 20,
    preset: str = "medium",
    verbose: bool = False,
) -> None:
    """Burn subtitles into the looped/trimmed background with narration audio."""

    background = background.expanduser().resolve()
    audio = audio.expanduser().resolve()
    ass_path = ass_path.expanduser().resolve()
    output = output.expanduser().resolve()

    # The ASS filter is referenced by a simple relative filename and FFmpeg runs
    # with cwd=the work directory.  This avoids fragile filtergraph escaping of
    # Windows drive letters, colons and quotes.
    command = build_compose_command(
        background=background,
        audio=audio,
        ass_filename=ass_path.name,
        output=output,
        duration=duration,
        offset=offset,
        crf=crf,
        preset=preset,
    )
    run_command(
        command,
        cwd=ass_path.parent,
        description=(
            "FFmpeg failed while composing the final video. The command above can be "
            "re-run manually to debug; generated files were kept."
        ),
        verbose=verbose,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise FFmpegError(
            "FFmpeg reported success but no output file was produced.",
            command,
        )
