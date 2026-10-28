"""Command-line interface for brainrot-tts."""

from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

from brainrot_tts import __version__
from brainrot_tts.errors import BrainrotError, ConfigError
from brainrot_tts.pipeline import PipelineConfig, run_pipeline
from brainrot_tts.subtitles.ass import parse_color
from brainrot_tts.tts import available_backends, create_backend
from brainrot_tts.video.ffmpeg import X264_PRESETS

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )
    # Keep Hugging Face's own logger quiet by default; users who need download
    # diagnostics can set HF_HUB_VERBOSITY=info themselves.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    # Third-party libraries are extremely chatty at DEBUG (HTTP headers, cache
    # checks, ...).  Keep only their warnings; our own logger stays at DEBUG.
    for noisy in (
        "httpx",
        "httpcore",
        "httpx2",
        "httpcore2",
        "huggingface_hub",
        "urllib3",
        "filelock",
        "torch",
        "transformers",
        "spacy",
        "numba",
        "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    # Known harmless deprecation warnings from torch when Kokoro is imported.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\..*)?")
    warnings.filterwarnings(
        "ignore",
        message=r"dropout option adds dropout after all but last recurrent layer.*",
        category=UserWarning,
    )


EPILOG = """\
examples:
  brainrot-tts -i inputs/example.txt -v background.mp4 -o result.mp4
  brainrot-tts -i text.txt -v bg.mp4 -o result.mp4 -V bm_george --lang en-gb
  brainrot-tts -i text.txt -v bg.mp4 -o result.mp4 --video-offset -1 --words-per-line 3
  brainrot-tts -i text.txt -v bg.mp4 -o result.mp4 --font-size 90 --position center
  brainrot-tts --list-voices

Default voice: af_heart (American English, female).
Run `brainrot-tts --help` for all options. See README.md for installation and
details about word timing and subtitle styling.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brainrot-tts",
        description=(
            "Turn a UTF-8 text file into a narrated MP4 with word-by-word\n"
            "highlighted ASS subtitles. Runs fully locally (Kokoro TTS + FFmpeg)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument(
        "-i",
        "--input",
        type=Path,
        metavar="TEXT_FILE",
        help="UTF-8 text file to narrate (required unless --list-voices).",
    )
    io_group.add_argument(
        "-v",
        "--video",
        type=Path,
        metavar="VIDEO_FILE",
        help=(
            "background video (required). Looped if shorter than the narration; trimmed if longer."
        ),
    )
    io_group.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="OUTPUT_MP4",
        help="output MP4 path (default: input file name with .mp4).",
    )
    io_group.add_argument(
        "--work-dir",
        type=Path,
        metavar="DIR",
        help="directory for audio.wav, subtitles.ass, timing.json (default: <output stem>.work).",
    )

    tts_group = parser.add_argument_group("speech synthesis")
    tts_group.add_argument(
        "--backend",
        default="kokoro",
        metavar="NAME",
        help=(f"TTS backend (default: kokoro; available: {', '.join(available_backends())})."),
    )
    tts_group.add_argument(
        "-V",
        "--voice",
        default="af_heart",
        metavar="VOICE",
        help="Kokoro voice name or path to a .pt voice (default: af_heart).",
    )
    tts_group.add_argument(
        "--lang",
        metavar="LOCALE",
        help="language/locale (en-us, en-gb, es, fr-fr, hi, it, pt-br, ja, zh; "
        "default: inferred from the voice, else en-us).",
    )
    tts_group.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="SPEED",
        help="speech speed multiplier, e.g. 0.9 or 1.15 (default: 1.0).",
    )
    tts_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="inference device (default: auto).",
    )
    tts_group.add_argument(
        "--list-voices",
        action="store_true",
        help="print available Kokoro voices and exit.",
    )

    subtitle_group = parser.add_argument_group("subtitle style")
    subtitle_group.add_argument(
        "--font",
        default="Arial",
        metavar="FONT",
        help="font family name (default: Arial; must be installed).",
    )
    subtitle_group.add_argument(
        "--font-size",
        type=int,
        metavar="SIZE",
        help="font size in video pixels (default: auto, 6.2%% of video height).",
    )
    subtitle_group.add_argument(
        "--text-color",
        default="#FFFFFF",
        metavar="COLOR",
        help="spoken text colour (default: #FFFFFF).",
    )
    subtitle_group.add_argument(
        "--highlight-color",
        default="#FFD400",
        metavar="COLOR",
        help="active word colour (default: #FFD400).",
    )
    subtitle_group.add_argument(
        "--outline-color",
        default="#000000",
        metavar="COLOR",
        help="text outline colour, used with or without the box (default: #000000).",
    )
    subtitle_group.add_argument(
        "--outline-width",
        type=float,
        metavar="WIDTH",
        help="text outline width in pixels (default: auto, ~4.5%% of font size).",
    )
    subtitle_group.add_argument(
        "--box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw an opaque box behind subtitles (default: on; use --no-box for outline-only).",
    )
    subtitle_group.add_argument(
        "--background-color",
        default="#000000",
        metavar="COLOR",
        help="subtitle box colour (default: #000000).",
    )
    subtitle_group.add_argument(
        "--background-opacity",
        type=float,
        default=1.0,
        metavar="0..1",
        help="subtitle box opacity, 0=transparent, 1=opaque (default: 1.0).",
    )
    subtitle_group.add_argument(
        "--box-padding",
        type=float,
        metavar="PIXELS",
        help="subtitle box padding in pixels (default: auto, ~16%% of font size).",
    )
    subtitle_group.add_argument(
        "--position",
        choices=["bottom", "center", "top"],
        default="bottom",
        help="vertical subtitle position (default: bottom).",
    )
    subtitle_group.add_argument(
        "--margin-bottom",
        type=int,
        metavar="PIXELS",
        help="distance from the bottom/top edge in pixels (default: auto, 8%% of height).",
    )
    subtitle_group.add_argument(
        "--words-per-line",
        type=int,
        default=4,
        metavar="N",
        help="maximum words visible at once, punctuation permitting (default: 4).",
    )

    video_group = parser.add_argument_group("video")
    video_group.add_argument(
        "--video-offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="start position in the background video; -1 picks a random offset (default: 0).",
    )
    video_group.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="seed for --video-offset -1 (useful for reproducible renders).",
    )
    video_group.add_argument(
        "--crf",
        type=int,
        default=20,
        metavar="N",
        help="libx264 quality, lower is better (default: 20).",
    )
    video_group.add_argument(
        "--preset",
        choices=sorted(X264_PRESETS),
        default="medium",
        help="libx264 speed/quality preset (default: medium).",
    )

    cache_group = parser.add_argument_group("caching / debugging")
    cache_group.add_argument(
        "--clean",
        action="store_true",
        help="delete generated intermediate files after a successful run.",
    )
    cache_group.add_argument(
        "--regenerate", action="store_true", help="ignore cached narration and synthesize again."
    )
    cache_group.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it already exists."
    )
    cache_group.add_argument(
        "--verbose", action="store_true", help="show debug logging and the full FFmpeg command."
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.font_size is not None and args.font_size < 1:
        raise ConfigError("--font-size must be at least 1.")
    if args.outline_width is not None and args.outline_width < 0:
        raise ConfigError("--outline-width must be zero or positive.")
    if args.box_padding is not None and args.box_padding < 0:
        raise ConfigError("--box-padding must be zero or positive.")
    if args.margin_bottom is not None and args.margin_bottom < 0:
        raise ConfigError("--margin-bottom must be zero or positive.")
    if args.words_per_line < 1:
        raise ConfigError("--words-per-line must be at least 1.")
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise ConfigError("--speed must be a positive number.")
    if not 0.0 <= args.background_opacity <= 1.0:
        raise ConfigError("--background-opacity must be between 0 and 1.")
    if not math.isfinite(args.video_offset):
        raise ConfigError("--video-offset must be a finite number.")
    if args.video_offset < 0 and args.video_offset != -1:
        raise ConfigError("--video-offset must be -1 (random) or a non-negative number.")
    if not 0 <= args.crf <= 51:
        raise ConfigError("--crf must be between 0 and 51.")
    if not args.font or "," in args.font or "\n" in args.font:
        raise ConfigError("--font must be a single font family name (no commas).")
    if args.lang is not None and not args.lang.strip():
        raise ConfigError("--lang must not be empty.")

    # Validate colours early, before any expensive synthesis work.
    for name in ("text_color", "highlight_color", "outline_color", "background_color"):
        parse_color(getattr(args, name))


def _print_voices(backend_name: str) -> int:
    backend = create_backend(backend_name)
    groups = backend.list_voices()
    if not groups:
        print(f"Backend {backend_name!r} reports no voice list.")
        return 0
    for label, voices in groups.items():
        print(f"{label}:")
        print("  " + ", ".join(voices))
    return 0


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if args.input is None:
        raise ConfigError("--input is required (or use --list-voices).")
    if args.video is None:
        raise ConfigError("--video is required (a background video).")

    input_path = args.input.expanduser()
    output_path = (
        args.output.expanduser() if args.output is not None else input_path.with_suffix(".mp4")
    )
    work_dir = (
        args.work_dir.expanduser()
        if args.work_dir is not None
        else output_path.with_name(output_path.stem + ".work")
    )

    language = args.lang
    if language is None:
        language = create_backend(args.backend).language_for_voice(args.voice) or "en-us"

    return PipelineConfig(
        input_path=input_path,
        video_path=args.video,
        output_path=output_path,
        work_dir=work_dir,
        backend=args.backend,
        voice=args.voice,
        language=language,
        speed=args.speed,
        device=args.device,
        font=args.font,
        font_size=args.font_size,
        text_color=args.text_color,
        highlight_color=args.highlight_color,
        outline_color=args.outline_color,
        outline_width=args.outline_width,
        box=args.box,
        background_color=args.background_color,
        background_opacity=args.background_opacity,
        box_padding=args.box_padding,
        position=args.position,
        margin_bottom=args.margin_bottom,
        words_per_line=args.words_per_line,
        video_offset=args.video_offset,
        seed=args.seed,
        crf=args.crf,
        preset=args.preset,
        clean=args.clean,
        regenerate=args.regenerate,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        if args.list_voices:
            return _print_voices(args.backend)
        _validate_args(args)
        config = _config_from_args(args)
        run_pipeline(config)
    except BrainrotError as exc:
        logger.error("error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        return 130
    return 0
