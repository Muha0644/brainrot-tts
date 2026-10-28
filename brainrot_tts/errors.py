"""Exception types used across brainrot-tts.

All expected failures inherit from :class:`BrainrotError` so the CLI can print a
single helpful message instead of a traceback. Unexpected bugs are deliberately
not wrapped.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence


class BrainrotError(Exception):
    """Base class for errors that should be shown to the user."""


class ConfigError(BrainrotError):
    """Invalid command-line arguments, input files, or output paths."""


class TTSError(BrainrotError):
    """The TTS backend failed (missing model, invalid voice, ...)."""


class SubtitleError(BrainrotError):
    """Subtitle grouping or ASS generation failed."""


class FFmpegError(BrainrotError):
    """An FFmpeg/FFprobe invocation failed.

    The message includes a copy-pasteable command and the tail of stderr, so a
    failure can be reproduced without re-running with ``--verbose``.
    """

    def __init__(
        self,
        summary: str,
        command: Sequence[str],
        returncode: int | None = None,
        stderr: str = "",
        *,
        stderr_tail_lines: int = 25,
    ) -> None:
        self.summary = summary
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr
        self.stderr_tail_lines = stderr_tail_lines
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [self.summary]
        if self.returncode is not None:
            lines.append(f"exit code: {self.returncode}")
        if self.command:
            lines.append("command: " + shlex.join(self.command))
        stderr = self.stderr.strip()
        if stderr:
            tail = stderr.splitlines()[-self.stderr_tail_lines :]
            lines.append(f"stderr (last {len(tail)} lines):")
            lines.extend(tail)
        return "\n".join(lines)
