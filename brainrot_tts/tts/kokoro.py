"""Kokoro TTS backend.

Kokoro's ``KPipeline`` (via ``misaki``) returns phoneme-aligned tokens with
``start_ts``/``end_ts`` for English.  Those tokens already carry the original
grapheme text and whitespace, so we can group them into words without running
ASR or forced alignment.  The timings are relative to each synthesized chunk;
this module adds the running audio offset for chunks.

All Kokoro-specific details live here.  The rest of the application only uses
:class:`brainrot_tts.models.TTSResult` / :class:`WordTiming`.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import unicodedata
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from brainrot_tts.errors import TTSError
from brainrot_tts.models import TTSOptions, TTSResult, WordTiming
from brainrot_tts.tts.base import TTSBackend, ensure_voice_not_empty

logger = logging.getLogger(__name__)

MODEL_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000

#: Human-readable locale -> Kokoro language code.
KOKORO_LANG_CODES: dict[str, str] = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "en": "en-us",
    "en-us": "en-us",
    "en_us": "en-us",
    "en-gb": "en-gb",
    "en_gb": "en-gb",
    "es": "es",
    "es-es": "es",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "hi": "hi",
    "it": "it",
    "pt": "pt-br",
    "pt-br": "pt-br",
    "pt_br": "pt-br",
    "ja": "ja",
    "jp": "ja",
    "zh": "zh",
    "zh-cn": "zh",
    "cmn": "zh",
    # Raw Kokoro codes are accepted too.
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "j": "ja",
    "z": "zh",
}

VOICE_GROUPS: dict[str, tuple[str, list[str]]] = {
    "en-us": (
        "American English",
        [
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
        ],
    ),
    "en-gb": (
        "British English",
        [
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
        ],
    ),
    "es": ("Spanish", ["ef_dora", "em_alex", "em_santa"]),
    "fr-fr": ("French", ["ff_siwis"]),
    "hi": ("Hindi", ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]),
    "it": ("Italian", ["if_sara", "im_nicola"]),
    "pt-br": ("Brazilian Portuguese", ["pf_dora", "pm_alex", "pm_santa"]),
    "ja": ("Japanese", ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]),
    "zh": (
        "Mandarin Chinese",
        [
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ],
    ),
}

ALL_VOICES: frozenset[str] = frozenset(
    voice for _label, voices in VOICE_GROUPS.values() for voice in voices
)

# Characters that carry no spoken content; used to attach orphan punctuation
# to the preceding word so a comma/dash/quote is never highlighted on its own.
_PUNCT_CATEGORIES = frozenset({"P", "S"})
_SPACED_ORPHANS = frozenset({"—", "–", "-", "−", "…"})


def normalize_language(language: str) -> str:
    """Return a canonical locale name, raising a helpful error if unknown."""

    key = language.strip().lower()
    try:
        return LANGUAGE_ALIASES[key]
    except KeyError:
        supported = ", ".join(sorted(KOKORO_LANG_CODES))
        raise TTSError(
            f"Kokoro does not support language {language!r}. Supported values: {supported} "
            "(or a Kokoro code: a, b, e, f, h, i, p, j, z)."
        ) from None


def infer_language_from_voice(voice: str) -> str | None:
    """Infer a locale from a Kokoro voice name such as ``bm_george``."""

    if voice.lower().endswith(".pt") or not voice:
        return None
    prefix = voice[0].lower()
    for locale, code in KOKORO_LANG_CODES.items():
        if code == prefix:
            return locale
    return None


@dataclass(slots=True)
class _WordDraft:
    text: str
    start: float | None
    end: float | None
    break_before: bool


def _is_punctuation_only(text: str) -> bool:
    if not text:
        return False
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char)[0] not in _PUNCT_CATEGORIES:
            return False
    return True


def draft_words_from_tokens(tokens: Sequence[Any], offset: float = 0.0) -> list[_WordDraft]:
    """Group Kokoro/misaki tokens into word drafts.

    Tokens look like ``MToken(text='world', whitespace=' ', start_ts=..., end_ts=...)``.
    A token's ``whitespace`` marks the end of a word.  Punctuation tokens are
    attached to their word (``world`` + ``!``), and orphan punctuation such as a
    dash between spaces is attached to the previous word.
    """

    drafts: list[_WordDraft] = []
    buffer: list[str] = []
    word_start: float | None = None
    word_end: float | None = None
    word_break = False
    pending_break = False

    def flush() -> None:
        nonlocal buffer, word_start, word_end, word_break
        if not buffer:
            return
        drafts.append(
            _WordDraft(
                text="".join(buffer),
                start=None if word_start is None else offset + word_start,
                end=None if word_end is None else offset + word_end,
                break_before=word_break,
            )
        )
        buffer = []
        word_start = None
        word_end = None
        word_break = False

    for token in tokens:
        raw_text = str(getattr(token, "text", "") or "")
        whitespace = str(getattr(token, "whitespace", "") or "")
        token_start = getattr(token, "start_ts", None)
        token_end = getattr(token, "end_ts", None)
        token_start = None if token_start is None else float(token_start)
        token_end = None if token_end is None else float(token_end)

        # misaki sometimes puts leading newlines inside token.text (for example
        # token.text == "\n\nNew").  Strip and turn them into break hints
        # instead of letting a newline end up inside a displayed word.
        stripped = raw_text.strip()
        if stripped:
            start = raw_text.find(stripped)
            leading = raw_text[:start]
            trailing = raw_text[start + len(stripped) :]
            text = stripped
        else:
            leading = raw_text
            trailing = ""
            text = ""
        if leading:
            flush()
            if "\n" in leading or "\r" in leading:
                pending_break = True
        if text:
            if any(char.isspace() for char in text):
                text = re.sub(r"\s+", " ", text)
            if not buffer:
                word_start = token_start
                word_end = token_end
                word_break = pending_break
                pending_break = False
            else:
                if word_start is None:
                    word_start = token_start
                if token_end is not None:
                    word_end = token_end if word_end is None else max(word_end, token_end)
            buffer.append(text)
        if trailing:
            flush()
            if "\n" in trailing or "\r" in trailing:
                pending_break = True

        if whitespace:
            flush()
            if "\n" in whitespace or "\r" in whitespace:
                pending_break = True

    flush()

    # Attach standalone punctuation to the preceding word.  Opening quotes and
    # brackets are already part of the following word because misaki gives them
    # whitespace == "".
    merged: list[_WordDraft] = []
    for draft in drafts:
        if (
            merged
            and _is_punctuation_only(draft.text)
            and not draft.break_before
            and "\n" not in draft.text
        ):
            previous = merged[-1]
            # A dash/ellipsis separated by spaces in the source should keep a
            # space ("paragraph — with"), while commas and closing quotes attach
            # directly ("word," / `"quoted"`).
            if draft.text[:1] in _SPACED_ORPHANS:
                previous.text += " " + draft.text
            else:
                previous.text += draft.text
            if draft.end is not None:
                previous.end = draft.end if previous.end is None else max(previous.end, draft.end)
            continue
        merged.append(draft)
    return merged


def _estimate_word_timings(
    drafts: list[_WordDraft], offset: float, duration: float
) -> list[WordTiming]:
    """Proportionally distribute a chunk's duration when no token timing exists.

    Used for non-English Kokoro pipelines and as a safety net if the model does
    not return ``pred_dur``.  English Kokoro normally uses exact token timings.
    """

    weights = [max(len(draft.text), 1) for draft in drafts]
    total_weight = sum(weights) or 1
    words: list[WordTiming] = []
    cursor = offset
    for draft, weight in zip(drafts, weights, strict=True):
        word_duration = duration * weight / total_weight
        start = cursor
        end = cursor + word_duration
        words.append(WordTiming(draft.text, start, end, draft.break_before))
        cursor = end
    return words


def materialize_words(drafts: list[_WordDraft], offset: float, duration: float) -> list[WordTiming]:
    """Convert drafts to word timings, estimating only when timings are absent."""

    all_timed = all(d.start is not None and d.end is not None for d in drafts)
    timed_span = sum(max(0.0, (draft.end or 0.0) - (draft.start or 0.0)) for draft in drafts)
    if not all_timed or timed_span <= 0.0:
        return _estimate_word_timings(drafts, offset, duration)

    words: list[WordTiming] = []
    previous_end = offset
    for draft in drafts:
        assert draft.start is not None and draft.end is not None
        start = max(draft.start, previous_end, 0.0)
        end = max(draft.end, start)
        words.append(WordTiming(draft.text, start, end, draft.break_before))
        previous_end = end
    return words


def split_plain_words(text: str) -> list[tuple[str, bool]]:
    """Fallback word splitter for backends/chunks without token timings.

    Returns ``(word, break_before)`` pairs, preserving punctuation in the word
    and treating newlines as hard line-break hints.
    """

    results: list[tuple[str, bool]] = []
    previous_end = 0
    for match in re.finditer(r"\S+", text):
        gap = text[previous_end : match.start()]
        results.append((match.group(0), bool(results) and ("\n" in gap or "\r" in gap)))
        previous_end = match.end()
    return results


def _enable_huggingface_offline() -> None:
    """Set HF offline mode, including when huggingface_hub is already imported."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True  # type: ignore[attr-defined]


def _huggingface_cache_root() -> Path:
    """Mirror huggingface_hub's default cache location without importing it."""

    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(os.path.expandvars(explicit)).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(os.path.expandvars(hf_home)).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(os.path.expandvars(xdg)).expanduser() if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def _write_wav(path: Path, samples: Any, sample_rate: int) -> None:
    """Write float32 [-1, 1] mono samples as 16-bit PCM, atomically."""

    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with wave.open(str(tmp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class KokoroTTS(TTSBackend):
    """Local Kokoro-82M synthesis using the official ``kokoro`` package."""

    name: ClassVar[str] = "kokoro"
    MODEL_FILENAMES: ClassVar[dict[str, str]] = {
        "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
        "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
    }

    def __init__(self, repo_id: str = MODEL_REPO) -> None:
        self.repo_id = repo_id
        self._pipeline: Any = None
        self._pipeline_key: tuple[str, str] | None = None

    def validate(self, options: TTSOptions) -> None:
        ensure_voice_not_empty(options.voice)
        if options.device not in {"auto", "cpu", "cuda", "mps"}:
            raise TTSError(f"Invalid device {options.device!r}. Choose auto, cpu, cuda or mps.")
        if not math.isfinite(options.speed) or options.speed <= 0:
            raise TTSError(f"Speed must be a positive number, got {options.speed!r}.")
        normalize_language(options.language)
        voice = options.voice
        if voice.lower().endswith(".pt"):
            path = Path(voice).expanduser()
            if not path.is_file():
                raise TTSError(f"Voice file not found: {path}")
        elif not re.fullmatch(r"[a-z]{2}_[A-Za-z0-9_]+", voice):
            raise TTSError(
                f"Invalid Kokoro voice {voice!r}. Expected a name like 'af_heart', "
                "a .pt voice file, or see --list-voices."
            )
        elif self.repo_id == MODEL_REPO and voice not in ALL_VOICES:
            raise TTSError(
                f"Unknown Kokoro voice {voice!r}. Run `brainrot-tts --list-voices` "
                "to see available voices, or pass a local .pt voice file."
            )

    def cache_key(self, text: str, options: TTSOptions) -> dict[str, Any]:
        key = super().cache_key(text, options)
        key["model_repo"] = self.repo_id
        if options.voice.lower().endswith(".pt"):
            voice_path = Path(options.voice).expanduser()
            if voice_path.is_file():
                key["voice_path"] = str(voice_path.resolve())
                key["voice_mtime_ns"] = voice_path.stat().st_mtime_ns
        try:
            from importlib.metadata import version

            key["kokoro_version"] = version("kokoro")
        except Exception:  # pragma: no cover - metadata always present in practice
            key["kokoro_version"] = "unknown"
        return key

    @classmethod
    def list_voices(cls) -> dict[str, list[str]]:
        return {f"{label} ({locale})": voices for locale, (label, voices) in VOICE_GROUPS.items()}

    @classmethod
    def language_for_voice(cls, voice: str) -> str | None:
        return infer_language_from_voice(voice)

    def _enable_offline_if_cached(self, voice: str) -> None:
        """Use Hugging Face's cache without network calls when everything is local.

        ``hf_hub_download`` performs a HEAD request even on cache hits.  If the
        model, config and requested voice are already in the current cache
        snapshot, setting ``HF_HUB_OFFLINE`` before huggingface_hub is imported
        keeps normal operation fully local and slightly faster.  An explicitly
        configured offline setting is left untouched.
        """

        if "HF_HUB_OFFLINE" in os.environ or "TRANSFORMERS_OFFLINE" in os.environ:
            return
        model_filename = self.MODEL_FILENAMES.get(self.repo_id)
        if model_filename is None:
            return

        files = ["config.json", model_filename]
        if not voice.lower().endswith(".pt"):
            files.append(f"voices/{voice}.pt")

        try:
            cache_root = _huggingface_cache_root()
        except (OSError, RuntimeError):  # pragma: no cover - unusual HOME setups
            return
        repo_dir = cache_root / ("models--" + self.repo_id.replace("/", "--"))
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            return

        candidates: list[Path] = []
        ref = repo_dir / "refs" / "main"
        try:
            if ref.is_file():
                commit = ref.read_text(encoding="utf-8").strip()
                if commit:
                    candidates.append(snapshots / commit)
            else:
                candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
            for snapshot in candidates:
                if all((snapshot / filename).is_file() for filename in files):
                    _enable_huggingface_offline()
                    logger.debug(
                        "Kokoro model and voice are cached; enabling Hugging Face offline mode."
                    )
                    return
        except OSError:  # pragma: no cover - never block synthesis on cache inspection
            return

    def _get_pipeline(self, lang_code: str, device: str) -> Any:
        key = (lang_code, device)
        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover - depends on install
            raise TTSError(
                "The Kokoro TTS backend is not installed. Install the project with "
                "`pip install -e .` (or `pip install kokoro>=0.9.4`) and try again."
            ) from exc

        logger.info("Loading Kokoro model %s (downloaded on first use)", self.repo_id)
        try:
            self._pipeline = KPipeline(
                lang_code=lang_code,
                repo_id=self.repo_id,
                device=None if device == "auto" else device,
            )
        except Exception as exc:
            raise TTSError(
                "Could not initialize Kokoro. The model files may be missing or the "
                f"requested device ({device}) may be unavailable. Original error: {exc}"
            ) from exc
        self._pipeline_key = key
        return self._pipeline

    def synthesize(
        self,
        text: str,
        options: TTSOptions,
        work_dir: Path,
        *,
        force: bool = False,
    ) -> TTSResult:
        del force  # The pipeline owns file-level caching.
        self.validate(options)
        if not text.strip():
            raise TTSError("Cannot synthesize empty text.")

        import numpy as np

        language = normalize_language(options.language)
        lang_code = KOKORO_LANG_CODES[language]
        voice = options.voice
        self._enable_offline_if_cached(voice)
        pipeline = self._get_pipeline(lang_code, options.device)
        if voice.lower().endswith(".pt"):
            voice = str(Path(voice).expanduser().resolve())
        else:
            inferred = infer_language_from_voice(voice)
            if inferred is not None and inferred != language:
                logger.warning(
                    "Voice %s is a %s voice but --lang is %s; Kokoro may sound wrong.",
                    voice,
                    inferred,
                    language,
                )

        logger.info(
            "Synthesizing with Kokoro (voice=%s, lang=%s, speed=%s, device=%s)",
            Path(voice).name if voice.lower().endswith(".pt") else voice,
            language,
            options.speed,
            options.device,
        )

        chunks: list[Any] = []
        words: list[WordTiming] = []
        offset = 0.0
        try:
            for result in pipeline(text, voice=voice, speed=options.speed, split_pattern=None):
                audio = result.audio
                if audio is None:
                    continue
                samples = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
                if samples.size == 0:
                    continue
                chunk_duration = float(samples.size) / SAMPLE_RATE
                chunks.append(samples)

                tokens = getattr(result, "tokens", None)
                if tokens:
                    drafts = draft_words_from_tokens(tokens, offset)
                    words.extend(materialize_words(drafts, offset, chunk_duration))
                else:
                    # Non-English pipelines do not expose token timestamps.
                    graphemes = str(getattr(result, "graphemes", "") or "")
                    drafts = [
                        _WordDraft(word, None, None, brk)
                        for word, brk in split_plain_words(graphemes)
                    ]
                    words.extend(_estimate_word_timings(drafts, offset, chunk_duration))
                offset += chunk_duration
        except TTSError:
            raise
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            hint = ""
            if "voice" in message.lower() or "404" in message or "not found" in message.lower():
                hint = " Check the voice name with --list-voices (and your network/HF cache)."
            raise TTSError(f"Kokoro synthesis failed: {message}.{hint}") from exc

        if not chunks:
            raise TTSError("Kokoro produced no audio for the supplied text.")
        if not words:
            raise TTSError("Kokoro produced audio but no word timings.")

        combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        audio_path = work_dir / "audio.wav"
        try:
            _write_wav(audio_path, combined, SAMPLE_RATE)
        except OSError as exc:
            raise TTSError(f"Could not write narration audio to {audio_path}: {exc}") from exc
        duration = float(combined.size) / SAMPLE_RATE

        if words[-1].end > duration + 0.1:
            logger.warning(
                "Last word timing (%.3fs) is longer than the audio (%.3fs); clamping.",
                words[-1].end,
                duration,
            )

        return TTSResult(
            audio_path=audio_path,
            words=words,
            duration=duration,
            sample_rate=SAMPLE_RATE,
            backend=self.name,
        )
