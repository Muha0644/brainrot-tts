Build a local CLI application called `brainrot-tts`.

The purpose of the application is to turn supplied text into a narrated video with word-by-word highlighted subtitles.

## IMPORTANT CONSTRAINTS
- Everything must run locally, no cloud APIs.
- Input is TEXT ONLY. Do not implement PDF, DOCX, OCR, web scraping, or document extraction yet.
- Options should be command-line arguments.
- Use FFmpeg for final video composition/encoding.
- The code should be cleanly structured so the TTS backend can be replaced later.
- Treat the supplied text as already suitable for TTS. It is acceptable to have basic handling for long TTS inputs if the selected engine requires it, but don't build a document-processing subsystem.

## PRIMARY PIPELINE
The application should perform:

    text
      ↓
    local TTS
      ↓
    audio + word-level timing information
      ↓
    .ASS subtitle generation
      ↓
    FFmpeg
      ↓
    final MP4

The initial/default TTS backend should be Kokoro.
Investigate the current Kokoro Python ecosystem and choose a well-maintained local implementation/library that provides reliable word-level timing information if possible.
Do NOT automatically run Whisper over the generated audio just to recover timestamps IF Kokoro can provide timing information directly. The original text is already known.
However, design the TTS abstraction so that a forced-alignment/Whisper backend could be used if necessary.

## FUNCTIONAL REQUIREMENTS
Input: --input text.txt
Require a user-provided video: --video background.mp4
The video should be used as the visual background.
The final video should end when the narration ends. If the background video is shorter than the narration, loop it automatically. If the background video is longer than the narration, trim it. Preserve the video's original resolution/aspect ratio by default. Do not unnecessarily re-encode the background multiple times. Add an option to start the video from an offset, if value specified is -1 make the offset random.

Generate narration locally. Expose at least: --voice VOICE
The default voice should be documented. Allow the user to specify the language/locale where supported. Allow useful TTS controls such as speech speed if supported by the selected backend: --speed SPEED
The exact set of options should reflect what the chosen tts implementation actually supports. Generated intermediate audio should be written to a temporary directory.

The central requirement is accurate word-level timing. Normalize TTS timing output into an internal representation similar to:
```
    @dataclass
    class WordTiming:
        text: str
        start: float
        end: float
```
The rest of the application must not depend directly on Kokoro's timing representation.

For example:
```
    [
        WordTiming("The", 0.00, 0.18),
        WordTiming("system", 0.18, 0.53),
        WordTiming("consists", 0.53, 0.91),
        ...
    ]
```
Handle punctuation correctly. The subtitle text should remain faithful to the supplied text.
Generate ASS/SSA subtitles rather than SRT. The subtitles should display several words at once and highlight the word currently being spoken.

Example concept:
```
    **The** system consists of three components.
    The **system** consists of three components.
    The system **consists** of three components.
    ...
```
while the spoken word changes appearance as narration progresses. Use ASS karaoke tags or another robust ASS mechanism where appropriate. The subtitle generator should be independent from the TTS implementation.
Expose command-line options for at least:
    --font FONT
    --font-size SIZE
    --text-color COLOR
    --highlight-color COLOR
    --outline-color COLOR
    --outline-width WIDTH
    --position POSITION
    --margin-bottom PIXELS
    --words-per-line N

Choose sensible defaults. The default style should be suitable for modern narrated-video/social-media content: large, readable text with a clearly visible active-word highlight and an opaque background.

Implement sensible grouping of words into subtitle lines. Support:
    --words-per-line N
but don't blindly break every N words if that creates an obviously bad result. Prefer breaking at punctuation/sentence boundaries where possible. The timing of each displayed word must remain accurate.

Use FFmpeg through Python's subprocess facilities. Do not use a Python video rendering library for the actual final composition.
Final composition should:
- use the supplied background video
- use generated narration audio
- burn the ASS subtitles into the video
- trim/loop the background appropriately
- produce a normal MP4 suitable for playback/upload

Support:
    --output output.mp4

By default preserve generated files such as audio.wav, subtitles.ass, timing.json for debugging, unless --clean is specified. If a file exists, reuse it instead of starting from scratch.

Provide useful logging. Add:
    --verbose
A failed FFmpeg invocation should print the actual command or enough information to reproduce/debug it. Do not hide errors behind generic exceptions.

## ARCHITECTURE
Use a modular architecture approximately like:

    brainrot-tts/
        __init__.py
        __main__.py
        cli.py
        tts/
            __init__.py
            base.py
            kokoro.py
        subtitles/
            __init__.py
            ass.py
        video/
            __init__.py
            ffmpeg.py
        models.py
        pipeline.py
        inputs/
            video.mp4
            file.txt
        outputs/
            final.mp4

The exact structure can differ if there is a better design.

Define a TTS abstraction similar to:

    class TTSBackend(ABC):
        def synthesize(self, text: str, options: ...) -> TTSResult:
            ...

where TTSResult contains at minimum:

    audio_path
    word_timings

For example:

    @dataclass
    class TTSResult:
        audio_path: Path
        words: list[WordTiming]

This abstraction is important because I may add F5-TTS or Piper later.
Keep Kokoro-specific code confined to the Kokoro backend.

## DEPENDENCIES
Use modern Python packaging. Clearly distinguish: Python dependencies from system dependencies. FFmpeg should be a system dependency and the application should check that it exists. Kokoro/model dependencies should be installed through the project's documented Python dependency mechanism. Everything required for inference must be local after installation/model download.
Do not introduce unnecessary frameworks. Use argparse or another lightweight CLI library. argparse is preferred unless there is a compelling reason otherwise.

## QUALITY REQUIREMENTS
- Type hints throughout.
- Python 3.11+.
- Clear error messages.
- No hard-coded absolute paths.
- Use pathlib.
- Use subprocess safely rather than shell=True.
- Proper temporary-file cleanup.

## IMPORTANT EDGE CASES
Handle:
- punctuation
- quotation marks
- apostrophes
- multiple spaces
- newlines
- very short text
- long text
- Unicode text
- text containing characters that ASS treats specially
- words containing ASS override characters
- background videos with different frame rates
- background videos shorter than narration
- background videos longer than narration
- missing FFmpeg
- missing/unavailable TTS model
- invalid voice
- output file already existing
- empty input

ASS escaping must be handled correctly. Do not silently corrupt text because of subtitle syntax.

## CLI DESIGN
Implement a useful help screen. The basic command should look approximately like:
`brainrot-tts --input text.txt --video background.mp4 --output result.mp4 --voice am_adam`
It should also support:
`brainrot-tts -i text.txt -v background.mp4 -o result.mp4 -V am_adam`

Provide `brainrot-tts --help`

## DOCUMENTATION
Create a README covering:
1. What the program does.
3. Installation.
3.1. Requirements.
3.2. Installing/configuring Kokoro locally.
3.3. Downloading/using voices/models.
4. Basic usage.
5. All important CLI arguments.
6. Example commands.
7. How word timing works
8. How subtitle styling works.
9. How to add another TTS backend.

## IMPLEMENTATION PROCESS
Before coding:
1. Inspect the current Kokoro Python API and determine the most reliable way to obtain audio and word-level timing locally.
2. Verify the exact API rather than inventing one.
3. Check current FFmpeg behavior for looping/shortening background video and ASS subtitle rendering.
4. Choose dependencies that are actively maintained and compatible with modern Python.
