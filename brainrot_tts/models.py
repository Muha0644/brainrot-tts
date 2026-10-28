"""Shared, backend-independent data structures.

The rest of the application only ever sees :class:`WordTiming` objects.  A TTS
backend is responsible for converting its own timing representation (Kokoro
phoneme tokens, Whisper alignment output, ...) into these values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WordTiming:
    """One spoken word and when it starts/ends, in seconds from audio start."""

    text: str
    start: float
    end: float
    #: True when the source text had a newline immediately before this word.
    #: Used by the subtitle grouper as a hint; not required for correctness.
    break_before: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("WordTiming.text must not be empty")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("WordTiming times must be finite")
        if self.start < 0 or self.end < 0:
            raise ValueError("WordTiming times must be non-negative")
        if self.end < self.start:
            raise ValueError(
                f"WordTiming end ({self.end}) is before start ({self.start}) for {self.text!r}"
            )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
        }
        if self.break_before:
            data["break_before"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordTiming:
        return cls(
            text=str(data["text"]),
            start=float(data["start"]),
            end=float(data["end"]),
            break_before=bool(data.get("break_before", False)),
        )


@dataclass(slots=True)
class TTSOptions:
    """Backend-independent synthesis options.

    ``language`` uses human-readable locale tags (``en-us``, ``en-gb``, ``ja``,
    ...).  Backends are free to map them to whatever their engine expects.
    """

    voice: str = "af_heart"
    language: str = "en-us"
    speed: float = 1.0
    device: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TTSResult:
    """What a backend returns: a local audio file plus word timings."""

    audio_path: Path
    words: list[WordTiming]
    duration: float
    sample_rate: int
    backend: str


@dataclass(slots=True)
class VideoInfo:
    """Basic properties of the background video."""

    path: Path
    width: int
    height: int
    duration: float
    fps: float | None
    rotation: int = 0
