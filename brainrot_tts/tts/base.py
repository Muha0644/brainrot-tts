"""TTS backend interface.

Keep this module free of engine-specific imports.  Adding Piper, F5-TTS or a
forced-alignment backend means implementing :class:`TTSBackend` and registering
it in ``brainrot_tts.tts.BACKENDS``; nothing else in the pipeline changes.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from brainrot_tts.errors import TTSError
from brainrot_tts.models import TTSOptions, TTSResult


class TTSBackend(ABC):
    """Abstract local text-to-speech engine."""

    #: Stable short name used by ``--backend`` and in the cache manifest.
    name: ClassVar[str] = "base"

    @abstractmethod
    def synthesize(
        self,
        text: str,
        options: TTSOptions,
        work_dir: Path,
        *,
        force: bool = False,
    ) -> TTSResult:
        """Synthesize ``text`` into ``work_dir`` and return audio + word timings.

        Implementations must write a mono or stereo PCM WAV file to
        ``work_dir / "audio.wav"``.  ``force`` asks the backend to ignore any
        engine-level cache; the pipeline normally handles caching before calling
        this method.  Timings must be relative to the start of that audio file.
        """

    def cache_key(self, text: str, options: TTSOptions) -> dict[str, Any]:
        """Return a JSON-serialisable key describing this synthesis request.

        The pipeline stores this alongside the generated files and regenerates
        when it changes.  Backends should add their own version/model identity.
        """

        return {
            "backend": self.name,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "voice": options.voice,
            "language": options.language,
            "speed": options.speed,
        }

    def validate(self, options: TTSOptions) -> None:
        """Validate backend-specific options, raising :class:`TTSError`."""

        ensure_voice_not_empty(options.voice)

    @classmethod
    def language_for_voice(cls, voice: str) -> str | None:
        """Return the natural locale for ``voice``, or None if unknown.

        Used by the CLI to infer ``--lang`` when it is omitted.  Backends that
        have language-specific voices can override this.
        """

        return None

    @classmethod
    def list_voices(cls) -> dict[str, list[str]]:
        """Return voices grouped by language, for ``--list-voices``."""

        return {}


def ensure_voice_not_empty(voice: str) -> None:
    if not voice.strip():
        raise TTSError("Voice must not be empty. Use --list-voices to see available voices.")
