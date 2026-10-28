# brainrot-tts

Turn a text file into a narrated MP4 with **word-by-word highlighted subtitles**, entirely on your machine.

```
text (UTF-8)
   -> Kokoro TTS (local)
   -> audio + word-level timestamps
   -> ASS subtitles (one active word highlighted at a time)
   -> FFmpeg (loop/trim background, burn subtitles, mux audio)
   -> result.mp4
```

No cloud APIs, no OCR, no document extraction. Input is plain text only.

---

## 1. What the program does

Given a text file and a background video, `brainrot-tts`:

1. synthesizes narration locally with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M);
2. obtains word timings directly from Kokoro's phoneme-token timestamps (no Whisper pass);
3. groups words into readable lines and generates an ASS/SSA subtitle file where the
   currently spoken word is recoloured on every frame;
4. uses FFmpeg to loop or trim the background so the video ends exactly when the narration
   ends, burns in the subtitles, and writes a normal `.mp4`.

The subtitle example:

```
**The** system consists of three components.
The **system** consists of three components.
The system **consists** of three components.
...
```

Everything is local after the initial model/voice download. The code is split into a
backend-independent TTS interface, an independent subtitle generator, and FFmpeg
composition, so another engine (Piper, F5-TTS, a Whisper/forced-alignment backend, ...)
can be added without touching the rest of the pipeline.

---

## 2. Requirements

### 2.1 System requirements

- **Python 3.11 or newer** (tested with 3.13).
- **FFmpeg and ffprobe on `PATH`** (FFmpeg 5+; tested with 7.1), built with **libass**
  (standard FFmpeg builds include it; check with `ffmpeg -filters | grep ass`). FFmpeg
  is a *system* dependency, not a Python package. The program fails early with an
  installation hint if it is missing.
- A font installed on the system for subtitle rendering. The default is **Arial**;
  libass substitutes a similar sans font (for example Nimbus Sans/Liberation Sans) when
  Arial is unavailable. Any font family installed through fontconfig can be selected
  with `--font`.
- ~2–4 GB of disk space for Python packages and the Kokoro model (torch is the largest
  dependency).
- Internet access on the **first run** to download the model and voice from Hugging Face.
  After that, inference is fully offline.
- CPU inference works. A CUDA or Apple-MPS GPU is optional and selected with
  `--device cuda` / `--device mps` (or automatically with `--device auto`).

FFmpeg install examples:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (winget)
winget install Gyan.FFmpeg
```

### 2.2 Python dependencies

The Python dependencies are declared in `pyproject.toml`. The important ones are:

| Package | Purpose |
| --- | --- |
| `kokoro` | local Kokoro-82M TTS (`>=0.9.4,<1.0`) |
| `misaki[en]` | grapheme-to-phoneme + token timestamps (pulled in by Kokoro) |
| `torch` | Kokoro inference (pulled in by Kokoro) |
| `huggingface-hub` | model/voice downloads and cache (pulled in by Kokoro) |
| `numpy` | audio assembly and WAV writing |

`ffmpeg` is **not** installable through pip here; it must be installed as a system
program. `espeak-ng` is bundled through `espeakng-loader` for English out-of-dictionary
words, so a separate system `espeak-ng` install is normally not required.

---

## 3. Installation

### 3.1 Create an environment and install the project

```bash
git clone <this-repository>
cd brainrot-tts

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

On Debian/Ubuntu, if `python3 -m venv` complains about `ensurepip`, install
`python3-venv` first (`sudo apt install python3-venv`).

After installation the `brainrot-tts` command is available. You can also run
`python -m brainrot_tts`.

#### CPU-only PyTorch (optional, smaller install)

The default `pip install -e .` may install a CUDA-enabled torch build on Linux. If you
do not have an NVIDIA GPU, install the CPU wheel first and then the project:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### 3.2 Set up Kokoro locally

The Kokoro backend uses the official [`kokoro`](https://github.com/hexgrad/kokoro)
Python package and `misaki` for G2P and timestamps. No separate Kokoro server or
checkpoint download step is needed: the first call to `KPipeline` downloads the model.

- Model repository: `hexgrad/Kokoro-82M` (~330 MB).
- Voices: individual `voices/<name>.pt` files (~1 MB each) from the same repository.
- Cache location: Hugging Face's cache, usually `~/.cache/huggingface/hub`.
- Override the cache location with `HF_HOME=/path/to/cache` (or `HF_HUB_CACHE` /
  the legacy `HUGGINGFACE_HUB_CACHE`).
- Offline use after the first download:

  ```bash
  HF_HUB_OFFLINE=1 brainrot-tts ...
  ```

  In fact, once the model, config and requested voice are present in the cache, the
  backend automatically enables Hugging Face offline mode before importing it, so normal
  runs make no network requests. Set `HF_HUB_OFFLINE=0` explicitly if you want to allow
  online cache/update checks.

If the model cannot be downloaded, the error includes the underlying Hugging Face
message. Pre-downloading with `huggingface-cli download hexgrad/Kokoro-82M` also works.

### 3.3 Voices and language

The default voice is **`af_heart`** (American English, female). List all known voices:

```bash
brainrot-tts --list-voices
```

Voice prefixes select the language:

| Prefix | Language / locale | `--lang` values |
| --- | --- | --- |
| `af_` / `am_` | American English | `en-us` (default), `en`, `a` |
| `bf_` / `bm_` | British English | `en-gb`, `b` |
| `ef_` / `em_` | Spanish | `es`, `e` |
| `ff_` | French | `fr-fr`, `fr`, `f` |
| `hf_` / `hm_` | Hindi | `hi`, `h` |
| `if_` / `im_` | Italian | `it`, `i` |
| `pf_` / `pm_` | Brazilian Portuguese | `pt-br`, `pt`, `p` |
| `jf_` / `jm_` | Japanese | `ja`, `j` |
| `zf_` / `zm_` | Mandarin Chinese | `zh`, `z` |

If `--lang` is omitted, it is inferred from the voice name. A voice can also be a local
`.pt` file path (`--voice ./voices/my_voice.pt`); pass `--lang` explicitly if that voice
is not English (the default is `en-us` for local voice files).

Japanese and Chinese also need the optional misaki extras, which are not installed by
default:

```bash
pip install "misaki[ja]"   # Japanese
pip install "misaki[zh]"   # Mandarin Chinese
```

**Timing note:** exact token timestamps are available for English (misaki). Kokoro's
non-English pipelines do not expose token timings in the same way, so word timing for
those languages is estimated proportionally within each synthesized chunk. See
[section 7](#7-how-word-timing-works).

---

## 4. Basic usage

```bash
brainrot-tts \
  --input inputs/example.txt \
  --video background.mp4 \
  --output result.mp4 \
  --voice af_heart
```

or with short options:

```bash
brainrot-tts -i text.txt -v background.mp4 -o result.mp4 -V am_adam
```

If `--output` is omitted, the output is placed next to the input text with an `.mp4`
extension (`inputs/example.txt` -> `inputs/example.mp4`).

No background video is bundled. You can generate a simple test background with FFmpeg:

```bash
ffmpeg -f lavfi -i "testsrc2=size=1080x1920:rate=30:duration=10" \
  -c:v libx264 -pix_fmt yuv420p background.mp4
```

The background's original resolution and aspect ratio are preserved (no scaling to a
fixed canvas; only a one-pixel even-dimension adjustment is applied if needed for
H.264/yuv420p). If it is shorter than the narration it loops; if it is longer it is
trimmed. The final video ends when the narration ends.

Generated intermediates are kept by default in a sibling directory named after the
output stem (`result.work/` for `result.mp4`), containing `audio.wav`, `subtitles.ass`,
`timing.json` and `manifest.json` for debugging. Re-running with the same text and TTS
options reuses the cached narration. Use `--clean` to delete the intermediates after a
successful render.

---

## 5. Command-line options

Run `brainrot-tts --help` for the authoritative help screen.

### Input / output

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input TEXT_FILE` | required | UTF-8 text file to narrate. |
| `-v`, `--video VIDEO_FILE` | required | Background video. Looped if short, trimmed if long. |
| `-o`, `--output OUTPUT_MP4` | `<input>.mp4` | Output MP4 path. Must end in `.mp4`. |
| `--work-dir DIR` | `<output stem>.work` | Where audio, ASS, timing and cache manifest are written. |

### Speech synthesis

| Option | Default | Description |
| --- | --- | --- |
| `--backend NAME` | `kokoro` | TTS backend (currently only `kokoro`). |
| `-V`, `--voice VOICE` | `af_heart` | Kokoro voice name or a `.pt` voice file. |
| `--lang LOCALE` | inferred, else `en-us` | Language/locale (see the table above). |
| `--speed SPEED` | `1.0` | Speech speed multiplier; must be positive. |
| `--device {auto,cpu,cuda,mps}` | `auto` | Inference device. |
| `--list-voices` | — | Print available voices and exit. |

### Subtitle style

| Option | Default | Description |
| --- | --- | --- |
| `--font FONT` | `Arial` | Installed font family. |
| `--font-size SIZE` | auto (6.2% of height) | Font size in video pixels. |
| `--font-weight {normal,bold,heavy}` | `heavy` | Weight; `heavy` adds a subtle same-colour stroke to the bold face. |
| `--text-color COLOR` | `#FFFFFF` | Base spoken text colour. |
| `--highlight-color COLOR` | `#FFD400` | Colour of the currently spoken word. |
| `--outline-color COLOR` | `#000000` | Text outline colour (applies with or without the box). |
| `--outline-width WIDTH` | auto (~4.5% of font size) | Text outline width in pixels. |
| `--box` / `--no-box` | box on | Rounded box behind the line, or outline-only. |
| `--background-color COLOR` | `#000000` | Subtitle box colour. |
| `--background-opacity 0..1` | `0.7` | Box opacity (0 transparent, 1 opaque). |
| `--box-padding PIXELS` | auto (~16% of font size) | Padding between text and box edge. |
| `--corner-radius PIXELS` | auto (~35% of font size) | Rounded box corner radius; `0` gives square corners. |
| `--position {bottom,center,top}` | `bottom` | Vertical placement. |
| `--margin-bottom PIXELS` | auto (40% of height) | Distance from bottom/top edge. |
| `--words-per-line N` | `4` | Maximum words visible at once. |

Colours accept `#RRGGBB`, `RRGGBB`, `#RGB`, `&HAABBGGRR`, `&HBBGGRR`, or basic names
(`white`, `black`, `yellow`, `red`, ...). Sizes are in pixels because the ASS `PlayRes`
is set to the video's actual dimensions.

### Video

| Option | Default | Description |
| --- | --- | --- |
| `--video-offset SECONDS` | `0` | Start offset in the background. `-1` picks a random offset. |
| `--seed N` | random | Seed used with `--video-offset -1`. |
| `--crf N` | `20` | libx264 constant rate factor (0–51, lower = better). |
| `--preset NAME` | `medium` | libx264 preset (`ultrafast` ... `veryslow`). |

### Caching / debugging

| Option | Default | Description |
| --- | --- | --- |
| `--clean` | off | Delete generated intermediates after a successful render. |
| `--regenerate` | off | Ignore the cached narration and synthesize again. |
| `--overwrite` | off | Required to replace an existing output file. |
| `--verbose` | off | Debug logs, including the full FFmpeg command. |
| `--version` | — | Print the version and exit. |

Note: `-v` is the short form of `--video`; verbose logging is `--verbose` only.

---

## 6. Examples

```bash
# Basic
brainrot-tts -i inputs/example.txt -v background.mp4 -o result.mp4

# British voice, slower speech
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 -V bm_george --lang en-gb --speed 0.92

# Random starting point in the background, 3 words per line
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 --video-offset -1 --seed 42 --words-per-line 3

# Bigger centred captions without a box
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 \
  --position center --font-size 96 --no-box --outline-width 6

# Semi-transparent box, custom colours
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 \
  --background-opacity 0.65 --text-color '#F5F5F5' --highlight-color '#00E5FF'

# Square corners and the plain bold face
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 --corner-radius 0 --font-weight bold

# Force a fresh narration, keep the intermediates
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 --regenerate

# Verify the cache hit on a second run, then clean up
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 --clean --overwrite

# Custom working directory
brainrot-tts -i text.txt -v bg.mp4 -o out.mp4 --work-dir .cache/run-01

# Same as the installed command
python -m brainrot_tts -i text.txt -v bg.mp4 -o out.mp4
```

---

## 7. How word timing works

### Kokoro token timestamps (default)

For English, Kokoro's `KPipeline` tokenizes the text with `misaki` into tokens that
carry the original grapheme text, whitespace, phonemes, and `start_ts`/`end_ts`
(phoneme-level timestamps derived from the model's predicted durations). We convert
those tokens into words:

- a word ends when a token carries trailing whitespace;
- punctuation tokens such as `,`, `!` and closing quotes are attached to the preceding
  word, so `world!` is highlighted as one unit;
- opening quotes/brackets stay attached to the following word;
- standalone punctuation (for example an em dash between spaces) is attached to the
  previous word and never highlighted alone;
- newlines are recorded as `break_before` hints for the subtitle grouper;
- long text is synthesized in chunks by Kokoro; each chunk's timings are offset by the
  duration of all preceding chunks.

The result is normalized to:

```python
@dataclass
class WordTiming:
    text: str
    start: float  # seconds from the start of the narration
    end: float
```

and written to `timing.json`:

```json
{
  "version": 1,
  "backend": "kokoro",
  "duration": 5.65,
  "sample_rate": 24000,
  "audio": "audio.wav",
  "words": [
    {"text": "The", "start": 0.275, "end": 0.675},
    {"text": "system", "start": 0.675, "end": 1.175},
    {"text": "consists.", "start": 1.475, "end": 2.6}
  ]
}
```

No ASR or forced alignment is run for the default backend. The original text is known,
so recovering timestamps with Whisper would only add error and latency.

### Fallbacks and caveats

- **Non-English Kokoro pipelines** do not expose token timestamps through `KPipeline`.
  In that case word timings are estimated proportionally by word length within each
  chunk. They are usable, but less precise than English timings.
- If the model unexpectedly returns no predicted durations, the same proportional
  fallback is used rather than failing the render.
- Timestamps are derived from Kokoro's duration predictor, so they track the synthesized
  audio rather than the source text's ideal timing. The highlight can be off by a few
  tens of milliseconds on unusual pronunciations.
- The `TTSBackend.synthesize` contract allows any engine to return `WordTiming` objects.
  A future backend can run a forced aligner (WhisperX, whisper-timestamped, ...) if an
  engine cannot provide timing directly.

### Cache and reuse

`<output stem>.work/manifest.json` stores the text hash and TTS options (backend, voice,
language, speed, model identity, Kokoro version). If `audio.wav` and `timing.json` exist
and the manifest matches, narration is reused. Changing the text or TTS options, or
passing `--regenerate`, triggers a fresh synthesis. Subtitle styling is cheap and is
always regenerated, so you can tweak colours/positions without re-running TTS.

---

## 8. How subtitle styling works

The program writes ASS/SSA (not SRT) and lets FFmpeg's `ass` filter burn it in.

- **PlayRes** is the background video's display resolution (after rotation), so sizes
  and margins are real video pixels. There is no accidental scaling between a 720p and
  a 4K background.
- **One text Dialogue event per word (plus one box event and, in heavy mode, one
  outline pass per line).** Every text event contains the whole line, with the active
  word wrapped in ASS colour overrides. At any moment exactly one word is highlighted;
  earlier words return to the base colour. This avoids the “progressive fill” behaviour
  of `\k` karaoke tags, which leaves all previously spoken words highlighted.
- **Rounded box layer.** A lower `BrainrotBox` layer draws a rounded rectangle as an
  ASS vector drawing (`\p1`), filled with `--background-color` at
  `--background-opacity`. Rounded corners use cubic bezier arcs with radius
  `--corner-radius` (auto ≈ 35% of the font size; `0` for square corners). Geometry is
  derived from the same estimated text width and the style margins, so the box does not
  jitter as the highlight moves.
- **Heavy weight.** `--font-weight heavy` (the default) keeps the bold face and adds a
  subtle same-colour stroke of ≈2% of the font size, faking a weight heavier than most
  installed families provide. A static lower `BrainrotOutline` layer supplies the dark
  outer outline, while the per-word text layer switches both fill and stroke colour for
  the active word. Use `--font-weight bold` for the plain bold face, or `normal` for
  regular weight.
- **Grouping** uses `--words-per-line` as a maximum, then:
  1. never crosses a newline from the input text;
  2. respects an estimated width budget based on the video width and font size;
  3. prefers to break after sentence-ending punctuation (`. ! ? …` and CJK equivalents),
     ignoring common abbreviations (`Dr.`, `etc.`) and initials;
  4. otherwise prefers clause punctuation (`, ; : —`);
  5. falls back to the word-count/width limit.
  A line is held for 0.4 s after its last word; if the next line starts within 0.25 s of
  that, it is held until the next line starts, so short pauses do not cause flicker.
- **Timing** is monotonic and rounded to ASS centiseconds, with a minimum event length
  to avoid zero-duration highlights. The first word appears when it is spoken; the last
  word holds briefly.
- **Text fidelity and escaping:** braces are escaped (`\{`, `\}`), and a zero-width
  space is inserted after every literal backslash so sequences such as `\N`, `\n`, `\h`
  or `{\an8}` are shown literally instead of being interpreted as ASS control codes.
  Quotes, apostrophes, commas, Unicode, emoji, and punctuation remain in the subtitle
  text. Control characters that ASS cannot render are dropped.
- **Fonts:** libass resolves the family through fontconfig. If Arial is not installed it
  substitutes a similar font. Use `fc-list` / `fc-match "Font Name"` to check.

Auto defaults are calculated from the background height: font size ≈ 6.2% of height,
outline ≈ 4.5% of font size, box padding ≈ 16% of font size, box corner radius ≈ 35% of
font size, heavy stroke ≈ 2% of font size, and bottom margin ≈ 40% of height. At
1080×1920 this is roughly 119 px text, 5.4 px outline, 19 px padding, 42 px corners,
2.4 px heavy stroke, and 768 px bottom margin — large, readable social-video captions
sitting just below the middle of the frame. Override any of them with the corresponding
flag; explicit values are always pixels.

---

## 9. Adding another TTS backend

1. Create a module under `brainrot_tts/tts/`, for example `piper.py`.
2. Implement `TTSBackend.synthesize`, which must write a WAV file and return a
   `TTSResult` containing `audio_path` and backend-independent `WordTiming` objects:

   ```python
   from pathlib import Path

   from brainrot_tts.models import TTSOptions, TTSResult, WordTiming
   from brainrot_tts.tts.base import TTSBackend


   class PiperTTS(TTSBackend):
       name = "piper"

       def synthesize(self, text, options: TTSOptions, work_dir: Path, *, force=False) -> TTSResult:
           audio_path = work_dir / "audio.wav"
           # ... run Piper locally, write audio_path ...
           words = [
               WordTiming(text="Hello", start=0.0, end=0.3),
               # ...
           ]
           return TTSResult(
               audio_path=audio_path,
               words=words,
               duration=1.2,
               sample_rate=22050,
               backend=self.name,
           )
   ```

   The audio must be written to `work_dir / "audio.wav"`; word timings are measured from
   the start of that file. If the engine only returns audio and phoneme/character
   alignments, map them to `WordTiming` inside the backend. If it returns no timing at all, run your aligner
   there and convert its output; the rest of the pipeline never needs to know.
3. Register it in `brainrot_tts/tts/__init__.py`:

   ```python
   BACKENDS = {
       "kokoro": ("brainrot_tts.tts.kokoro", "KokoroTTS"),
       "piper": ("brainrot_tts.tts.piper", "PiperTTS"),
   }
   ```

4. Use `--backend piper`. `--list-voices` can expose the backend's voices by overriding
   `list_voices()`. Override `cache_key()` if the engine has model/revision identities
   that affect the audio.
5. Keep all engine imports inside methods (or a guarded `try/except ImportError`) so
   `--help` and `--list-voices` keep working when optional engines are not installed.

The pipeline only requires that:
- `audio_path` is a local WAV file;
- `words` are chronological, non-overlapping-or-clamped, and measured from the start of
  that audio;
- word text is suitable for display (the subtitle generator handles ASS escaping).

---

## 10. Troubleshooting

**`Required command(s) not found: ffmpeg`** — install FFmpeg and make sure `ffmpeg` and
`ffprobe` are on `PATH` (`ffmpeg -version`, `ffprobe -version`).

**Model download fails** — check network/proxy access to `huggingface.co`, or pre-download
`hexgrad/Kokoro-82M` and set `HF_HOME`/`HF_HUB_OFFLINE`. The original Hugging Face error is
included in the message.

**Invalid voice** — run `brainrot-tts --list-voices`; use the exact name. A language
mismatch (for example `bm_george` with `--lang en-us`) is warned about but still rendered.

**Subtitles look wrong / missing characters** — install the selected font, or use a font
known to cover the language (`fc-list :lang=ja`, etc.). Try `--no-box --outline-width 4`
to check whether the text itself renders.

**CUDA/MPS errors** — use `--device cpu`; `--device auto` already falls back to CPU except
when CUDA is explicitly requested and unavailable.

**Output already exists** — pass `--overwrite`. (Intermediate narration is still reused
unless you also pass `--regenerate`.)

**FFmpeg failed** — the error includes the exact argv and the tail of FFmpeg's stderr.
Generated files are kept in the work directory. Re-run with `--verbose` to also see the
command in the debug log.

**Stale narration after editing the text** — the text hash changes, so narration is
regenerated automatically. If you want to keep an old run, use separate `--output` /
`--work-dir` paths. Use `--regenerate` to force a fresh take with the same text.

**Clean up** — `--clean` deletes `audio.wav`, `subtitles.ass`, `timing.json`,
`manifest.json` (and our temp files) after a successful run, then removes the work
directory if it is empty. It only removes the generated filenames listed above; it never
empties a custom work directory of other files.

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy brainrot_tts
```

Project layout:

```
brainrot_tts/
  __init__.py
  __main__.py        # python -m brainrot_tts
  cli.py             # argparse CLI, validation, logging
  pipeline.py        # caching + orchestration
  models.py          # WordTiming, TTSOptions, TTSResult, VideoInfo
  errors.py
  tts/
    base.py          # TTSBackend ABC
    kokoro.py        # Kokoro backend (all Kokoro/misaki specifics)
    __init__.py      # backend registry
  subtitles/
    grouping.py      # words -> lines (punctuation/width aware)
    ass.py           # ASS escaping, styling, one-event-per-word rendering
  video/
    ffmpeg.py        # ffprobe + FFmpeg command/composition
inputs/example.txt
outputs/             # suggested output location
tests/
```

FFmpeg is invoked through `subprocess.run` with an argument list (never `shell=True`);
all paths use `pathlib`. The final MP4 is written to a temporary `.part.mp4` and atomically
renamed, so an interrupted render does not leave a half-written `result.mp4`.

## License

MIT
