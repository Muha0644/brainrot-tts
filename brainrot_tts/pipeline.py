"""End-to-end text -> narration -> ASS -> MP4 pipeline."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brainrot_tts.errors import ConfigError, SubtitleError
from brainrot_tts.models import TTSOptions, TTSResult, VideoInfo, WordTiming
from brainrot_tts.subtitles.ass import SubtitleStyle, parse_color, render_ass
from brainrot_tts.subtitles.grouping import SubtitleLine, group_words
from brainrot_tts.tts import create_backend
from brainrot_tts.video.ffmpeg import (
    compose_video,
    ensure_tools,
    probe_duration,
    probe_video,
)

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
DEFAULT_WORDS_PER_LINE = 4
DEFAULT_HOLD_SECONDS = 0.40


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    video_path: Path
    output_path: Path
    work_dir: Path

    backend: str = "kokoro"
    voice: str = "af_heart"
    language: str = "en-us"
    speed: float = 1.0
    device: str = "auto"

    font: str = "Arial"
    font_size: int | None = None
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    outline_color: str = "#000000"
    outline_width: float | None = None
    box: bool = True
    background_color: str = "#000000"
    background_opacity: float = 1.0
    box_padding: float | None = None
    position: str = "bottom"
    margin_bottom: int | None = None
    words_per_line: int = DEFAULT_WORDS_PER_LINE

    video_offset: float = 0.0
    seed: int | None = None
    crf: int = 20
    preset: str = "medium"

    clean: bool = False
    regenerate: bool = False
    overwrite: bool = False
    verbose: bool = False
    extra: dict[str, object] = field(default_factory=dict)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ConfigError(f"Could not write {path}: {exc}") from exc
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _default_font_size(video_height: int) -> int:
    return max(24, min(180, round(video_height * 0.062)))


def _resolve_style(config: PipelineConfig, video: VideoInfo) -> SubtitleStyle:
    font_size = (
        config.font_size if config.font_size is not None else _default_font_size(video.height)
    )
    outline_width = (
        config.outline_width
        if config.outline_width is not None
        else max(2.0, round(font_size * 0.045, 1))
    )
    box_padding = (
        config.box_padding
        if config.box_padding is not None
        else max(6.0, round(font_size * 0.16, 1))
    )
    margin_bottom = (
        config.margin_bottom
        if config.margin_bottom is not None
        else max(20, round(video.height * 0.08))
    )
    margin_lr = max(16, round(video.width * 0.04))

    box_alpha = round((1.0 - config.background_opacity) * 255)
    box_color = parse_color(config.background_color, default_alpha=box_alpha)

    return SubtitleStyle(
        font=config.font,
        font_size=int(font_size),
        text_color=parse_color(config.text_color),
        highlight_color=parse_color(config.highlight_color),
        outline_color=parse_color(config.outline_color),
        outline_width=float(outline_width),
        box=config.box,
        box_color=box_color,
        box_padding=float(box_padding),
        position=config.position,
        margin_bottom=int(margin_bottom),
        margin_lr=int(margin_lr),
    )


def _resolve_video_offset(value: float, video_duration: float, seed: int | None) -> float:
    if value == -1:
        upper = max(0.0, video_duration - 0.01)
        offset = random.Random(seed).uniform(0.0, upper) if upper > 0 else 0.0
        logger.info("Random background offset: %.2fs (of %.2fs)", offset, video_duration)
        return offset
    if value < 0:
        raise ConfigError("--video-offset must be -1 (random) or a non-negative number.")
    if value >= video_duration:
        raise ConfigError(
            f"--video-offset ({value:g}s) must be smaller than the background duration "
            f"({video_duration:.2f}s). Use -1 for a random offset."
        )
    return value


def _read_input_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid UTF-8 text. Re-save it as UTF-8 and try again."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"Could not read input text {path}: {exc}") from exc
    if not text.strip():
        raise ConfigError(f"Input text is empty: {path}")
    if "\x00" in text:
        raise ConfigError(f"{path} contains NUL bytes and does not look like a UTF-8 text file.")
    return text


def _timing_payload(result: TTSResult) -> dict[str, object]:
    return {
        "version": 1,
        "backend": result.backend,
        "duration": round(result.duration, 4),
        "sample_rate": result.sample_rate,
        "audio": result.audio_path.name,
        "words": [word.to_dict() for word in result.words],
    }


def _load_timing(path: Path) -> tuple[list[WordTiming], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("timing file root must be a JSON object")
    words_data = data.get("words")
    if not isinstance(words_data, list) or not words_data:
        raise ValueError("timing file contains no words")
    words = [WordTiming.from_dict(item) for item in words_data]
    duration = float(data.get("duration") or 0.0)
    if duration <= 0:
        duration = max(word.end for word in words)
    return words, duration


def _load_cached_narration(
    work_dir: Path,
    audio_path: Path,
    timing_path: Path,
    cache_key: dict[str, object],
    *,
    verbose: bool,
) -> tuple[list[WordTiming], float] | None:
    manifest_path = work_dir / "manifest.json"
    if not (manifest_path.is_file() and audio_path.is_file() and timing_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable narration manifest: %s", exc)
        return None
    if not isinstance(manifest, dict):
        logger.warning("Ignoring invalid narration manifest (root is not an object).")
        return None
    if manifest.get("cache_version") != CACHE_VERSION or manifest.get("cache_key") != cache_key:
        logger.info("Narration cache is stale (text or TTS options changed); regenerating.")
        return None

    try:
        words, timing_duration = _load_timing(timing_path)
        audio_duration = probe_duration(audio_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid cached narration (%s); regenerating.", exc)
        return None
    if abs(audio_duration - timing_duration) > 0.25:
        logger.warning(
            "Cached audio/timing duration mismatch (%.2fs vs %.2fs); regenerating.",
            audio_duration,
            timing_duration,
        )
        return None

    logger.info(
        "Reusing cached narration from %s (%.2fs, %d words).",
        audio_path,
        audio_duration,
        len(words),
    )
    if verbose:
        logger.debug("First word: %s | last word: %s", words[0], words[-1])
    return words, audio_duration


_WORK_FILES = (
    "audio.wav",
    "audio.wav.tmp",
    "subtitles.ass",
    "subtitles.ass.tmp",
    "timing.json",
    "timing.json.tmp",
    "manifest.json",
    "manifest.json.tmp",
)


def _unlink_generated(path: Path, protected: set[Path]) -> None:
    """Delete a generated file unless it is one of the user's input/output files."""

    if path.resolve() in protected:
        return
    path.unlink(missing_ok=True)


def _remove_cache_files(work_dir: Path, protected: set[Path]) -> None:
    for name in (
        "audio.wav",
        "audio.wav.tmp",
        "timing.json",
        "timing.json.tmp",
        "manifest.json",
        "manifest.json.tmp",
    ):
        _unlink_generated(work_dir / name, protected)


def _clean_work_dir(work_dir: Path, protected: set[Path]) -> None:
    """Remove this tool's generated filenames, then the directory if empty."""

    for name in _WORK_FILES:
        try:
            _unlink_generated(work_dir / name, protected)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", work_dir / name, exc)
    # Leave a non-empty custom work dir in place.
    with contextlib.suppress(OSError):
        work_dir.rmdir()


def _generate_subtitles(
    words: list[WordTiming],
    config: PipelineConfig,
    video: VideoInfo,
    ass_path: Path,
) -> tuple[SubtitleStyle, list[SubtitleLine]]:
    style = _resolve_style(config, video)
    max_width_px = video.width - 2 * style.margin_lr
    lines = group_words(
        words,
        max_words=config.words_per_line,
        max_width_px=max_width_px,
        font_size=style.font_size,
        max_line_seconds=8.0,
    )
    if not lines:
        raise SubtitleError("Could not group any words into subtitle lines.")
    ass_text = render_ass(
        lines,
        style,
        play_res_x=video.width,
        play_res_y=video.height,
        hold_seconds=DEFAULT_HOLD_SECONDS,
    )
    _atomic_write_text(ass_path, ass_text)
    logger.info(
        "Subtitles: %d lines, %d word highlights (font %s %dpx, box=%s).",
        len(lines),
        len(words),
        style.font,
        style.font_size,
        "on" if style.box else "off",
    )
    logger.debug(
        "Subtitle style: text=%s highlight=%s outline=%s width=%.1f padding=%.1f position=%s",
        style.text_color,
        style.highlight_color,
        style.outline_color,
        style.outline_width,
        style.box_padding,
        style.position,
    )
    return style, lines


def run_pipeline(config: PipelineConfig) -> Path:
    """Run the full pipeline and return the final MP4 path."""

    started = time.monotonic()
    input_path = config.input_path.expanduser()
    video_path = config.video_path.expanduser()
    output_path = config.output_path.expanduser()
    work_dir = config.work_dir.expanduser()

    if not input_path.is_file():
        raise ConfigError(f"Input text file not found: {input_path}")
    if not video_path.is_file():
        raise ConfigError(f"Background video not found: {video_path}")
    if output_path.exists() and not config.overwrite:
        raise ConfigError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    if output_path.suffix.lower() != ".mp4":
        raise ConfigError("Output must use the .mp4 extension (for example --output result.mp4).")
    if output_path.resolve() in {input_path.resolve(), video_path.resolve()}:
        raise ConfigError("Output must not overwrite the input text or background video.")

    text = _read_input_text(input_path)
    ensure_tools()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Could not create output directory {output_path.parent}: {exc}") from exc
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Could not create work directory {work_dir}: {exc}") from exc
    if not work_dir.is_dir():
        raise ConfigError(f"Work directory is not a directory: {work_dir}")

    video = probe_video(video_path)
    logger.info(
        "Background: %dx%d, %.2fs%s",
        video.width,
        video.height,
        video.duration,
        f", {video.fps:.3f} fps" if video.fps else "",
    )
    offset = _resolve_video_offset(config.video_offset, video.duration, config.seed)

    backend = create_backend(config.backend)
    options = TTSOptions(
        voice=config.voice,
        language=config.language,
        speed=config.speed,
        device=config.device,
        extra=dict(config.extra),
    )
    backend.validate(options)
    cache_key = backend.cache_key(text, options)

    audio_path = work_dir / "audio.wav"
    timing_path = work_dir / "timing.json"
    manifest_path = work_dir / "manifest.json"
    ass_path = work_dir / "subtitles.ass"

    protected = {
        input_path.resolve(),
        video_path.resolve(),
        output_path.resolve(),
    }
    for generated in (audio_path, timing_path, manifest_path, ass_path):
        if generated.resolve() in protected:
            raise ConfigError(
                f"Generated file {generated} would overwrite an input or output file. "
                "Choose a different --work-dir."
            )
    if config.regenerate:
        logger.info("--regenerate: discarding cached narration and subtitles.")
        _remove_cache_files(work_dir, protected)
        _unlink_generated(ass_path, protected)

    cached = (
        None
        if config.regenerate
        else _load_cached_narration(
            work_dir, audio_path, timing_path, cache_key, verbose=config.verbose
        )
    )

    if cached is None:
        result = backend.synthesize(text, options, work_dir, force=config.regenerate)
        if not result.audio_path.is_file():
            raise ConfigError(f"TTS backend did not produce audio at {result.audio_path}.")
        if not result.words:
            raise ConfigError("TTS backend produced no word timings.")
        _atomic_write_json(timing_path, _timing_payload(result))
        _atomic_write_json(
            manifest_path,
            {
                "cache_version": CACHE_VERSION,
                "cache_key": cache_key,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "text_chars": len(text),
                "words": len(result.words),
            },
        )
        words = result.words
        logger.info(
            "Narration ready: %.2fs, %d words -> %s",
            result.duration,
            len(result.words),
            result.audio_path,
        )
    else:
        words, _cached_duration = cached

    audio_duration = probe_duration(audio_path)
    if audio_duration <= 0:
        raise ConfigError(f"Narration audio has no duration: {audio_path}")

    _generate_subtitles(words, config, video, ass_path)

    output_path = output_path.resolve()
    temp_output = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    logger.info(
        "Rendering %s (%.2fs, offset %.2fs, %s/crf %d) ...",
        output_path,
        audio_duration,
        offset,
        config.preset,
        config.crf,
    )
    try:
        compose_video(
            background=video_path,
            audio=audio_path,
            ass_path=ass_path,
            output=temp_output,
            duration=audio_duration,
            offset=offset,
            crf=config.crf,
            preset=config.preset,
            verbose=config.verbose,
        )
        try:
            os.replace(temp_output, output_path)
        except OSError as exc:
            raise ConfigError(
                f"Could not move the rendered video into place at {output_path}: {exc}"
            ) from exc
    except BaseException:
        temp_output.unlink(missing_ok=True)
        raise

    size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.monotonic() - started
    logger.info(
        "Wrote %s (%.1f MiB, %.2fs narration, %.1fs elapsed).",
        output_path,
        size_mb,
        audio_duration,
        elapsed,
    )

    if config.clean:
        _clean_work_dir(work_dir, protected)
        logger.info("Removed intermediate files from %s (--clean).", work_dir)
    else:
        logger.info("Intermediate files kept in %s (use --clean to remove them).", work_dir)
    return output_path
