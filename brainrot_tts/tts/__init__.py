"""Registry of available TTS backends."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from brainrot_tts.errors import ConfigError

if TYPE_CHECKING:
    from brainrot_tts.tts.base import TTSBackend

#: Backend name -> (module, class). Imports are lazy so that ``--help`` and
#: ``--list-voices`` do not need torch/Kokoro installed.
BACKENDS: dict[str, tuple[str, str]] = {
    "kokoro": ("brainrot_tts.tts.kokoro", "KokoroTTS"),
}


def available_backends() -> list[str]:
    return sorted(BACKENDS)


def create_backend(name: str) -> TTSBackend:
    """Instantiate a backend by name, with a clear error for unknown names."""

    key = name.strip().lower()
    try:
        module_name, class_name = BACKENDS[key]
    except KeyError:
        raise ConfigError(
            f"Unknown TTS backend {name!r}. Available backends: {', '.join(available_backends())}"
        ) from None
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()
