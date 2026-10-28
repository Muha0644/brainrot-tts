"""ASS/SSA subtitle generation with word-by-word highlighting.

We generate one Dialogue event per *word*, each containing the full line with
the active word recoloured.  This is more robust than progressive ``\\k``
karaoke fill (which leaves every already-spoken word highlighted): at any
moment exactly one word is highlighted, and the text/timing remain independent
of the TTS engine.

An optional second, lower-layer event draws an opaque box behind the line.  The
box and text layers use identical fonts/metrics, so the box does not move when
the highlighted word changes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from brainrot_tts.errors import ConfigError, SubtitleError
from brainrot_tts.subtitles.grouping import SubtitleLine

#: ASS uses &HAABBGGRR (note the reversed RGB order).
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "pink": (255, 192, 203),
    "gold": (255, 215, 0),
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")
_ASS_RE = re.compile(r"^&[hH]([0-9a-fA-F]{6}|[0-9a-fA-F]{8})&?$")

MIN_EVENT_SECONDS = 0.02
DEFAULT_HOLD_SECONDS = 0.40


def _to_ass_color(red: int, green: int, blue: int, alpha: int = 0) -> str:
    return f"&H{alpha & 0xFF:02X}{blue & 0xFF:02X}{green & 0xFF:02X}{red & 0xFF:02X}"


def parse_color(value: str, *, default_alpha: int = 0) -> str:
    """Parse ``#RRGGBB``, ``RRGGBB``, ``&HAABBGGRR`` or a basic colour name.

    Returns an ASS colour string.  For 6-digit hex input, ``default_alpha`` is
    used; 8-digit ``&H`` input keeps its own alpha channel.
    """

    text = value.strip()
    if not text:
        raise ConfigError("Colour must not be empty (try '#FFFFFF' or 'white').")

    named = _NAMED_COLORS.get(text.lower())
    if named is not None:
        return _to_ass_color(*named, default_alpha)

    ass_match = _ASS_RE.match(text)
    if ass_match:
        digits = ass_match.group(1)
        if len(digits) == 8:
            return "&H" + digits.upper()
        # 6 digits in ASS order are BBGGRR.
        return f"&H{default_alpha & 0xFF:02X}{digits.upper()}"

    hex_match = _HEX_RE.match(text)
    if hex_match:
        digits = hex_match.group(1)
        if len(digits) == 3:
            digits = "".join(char * 2 for char in digits)
        red = int(digits[0:2], 16)
        green = int(digits[2:4], 16)
        blue = int(digits[4:6], 16)
        return _to_ass_color(red, green, blue, default_alpha)

    raise ConfigError(
        f"Invalid colour {value!r}. Use #RRGGBB, RRGGBB, &HAABBGGRR, or a name "
        f"such as white, black, yellow."
    )


def _inline_override_color(ass_color: str) -> str:
    """Return only the BGR part for an inline ``\\c`` override."""

    if len(ass_color) == 10:
        return "&H" + ass_color[4:] + "&"
    return ass_color if ass_color.endswith("&") else ass_color + "&"


def escape_ass_text(text: str) -> str:
    """Escape text for an ASS Dialogue field without corrupting it.

    Braces would start an override block, and ``\\N``/``\\n``/``\\h`` are
    escapes.  We therefore backslash-escape braces and insert a zero-width
    space after every literal backslash so no escape sequence can form.  The
    zero-width space is invisible but keeps the displayed text faithful.
    """

    output: list[str] = []
    for char in text:
        if char == "{":
            output.append("\\{")
        elif char == "}":
            output.append("\\}")
        elif char == "\\":
            output.append("\\\u200b")
        elif char in "\r\n\t":
            output.append(" ")
        elif unicodedata.category(char) == "Cc":
            # Other C0/C1 control characters cannot be rendered safely.
            continue
        else:
            output.append(char)
    return "".join(output)


def _to_centiseconds(seconds: float) -> int:
    return max(0, int(round(seconds * 100)))


def format_ass_time(seconds: float) -> str:
    """Format seconds as ``H:MM:SS.cc`` (ASS timestamps)."""

    return format_ass_centiseconds(_to_centiseconds(seconds))


def format_ass_centiseconds(centiseconds: int) -> str:
    centiseconds = max(0, int(centiseconds))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


_ALIGNMENT = {"bottom": 2, "center": 5, "middle": 5, "top": 8}


@dataclass(slots=True)
class SubtitleStyle:
    font: str = "Arial"
    font_size: int = 72
    text_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000D4FF"
    outline_color: str = "&H00000000"
    outline_width: float = 3.0
    box: bool = True
    box_color: str = "&H00000000"
    box_padding: float = 12.0
    position: str = "bottom"
    margin_bottom: int = 90
    margin_lr: int = 40
    bold: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.font_size < 1:
            raise ConfigError("Font size must be at least 1 pixel.")
        if self.outline_width < 0:
            raise ConfigError("Outline width must be zero or positive.")
        if self.box_padding < 0:
            raise ConfigError("Box padding must be zero or positive.")
        if self.position not in _ALIGNMENT:
            raise ConfigError(f"Invalid position {self.position!r}. Choose bottom, center or top.")
        if not self.font or "\n" in self.font or "," in self.font:
            raise ConfigError("Font name must be a single font family without commas.")
        if self.margin_bottom < 0 or self.margin_lr < 0:
            raise ConfigError("Margins must be zero or positive.")

    @property
    def alignment(self) -> int:
        return _ALIGNMENT[self.position]


def _style_line(
    name: str,
    *,
    font: str,
    size: int,
    primary: str,
    secondary: str,
    outline: str,
    back: str,
    bold: bool,
    border_style: int,
    outline_width: float,
    alignment: int,
    margin_lr: int,
    margin_v: int,
) -> str:
    bold_value = -1 if bold else 0
    return (
        f"Style: {name},{font},{size},{primary},{secondary},{outline},{back},"
        f"{bold_value},0,0,0,100,100,0,0,{border_style},{outline_width:g},0,"
        f"{alignment},{margin_lr},{margin_lr},{margin_v},1"
    )


def _dialogue(
    layer: int,
    start_cs: int,
    end_cs: int,
    style_name: str,
    text: str,
) -> str:
    return (
        f"Dialogue: {layer},{format_ass_centiseconds(start_cs)},"
        f"{format_ass_centiseconds(end_cs)},{style_name},,0,0,0,,{text}"
    )


def _active_line(line: SubtitleLine, active_index: int, style: SubtitleStyle) -> str:
    highlight = _inline_override_color(style.highlight_color)
    base = _inline_override_color(style.text_color)
    parts: list[str] = []
    for index, word in enumerate(line.words):
        escaped = escape_ass_text(word.text)
        if index == active_index:
            parts.append(f"{{\\c{highlight}}}{escaped}{{\\c{base}}}")
        else:
            parts.append(escaped)
    return " ".join(parts)


def _plain_line(line: SubtitleLine) -> str:
    return " ".join(escape_ass_text(word.text) for word in line.words)


def _line_end_time(lines: Sequence[SubtitleLine], index: int, hold: float) -> float:
    line = lines[index]
    last = line.words[-1]
    natural_end = last.end + hold
    if index + 1 < len(lines):
        next_start = lines[index + 1].words[0].start
        # Hold the final highlight across short pauses; release the line during
        # long silences so it does not linger for seconds.
        if next_start <= natural_end + 0.25:
            return max(next_start, last.start + MIN_EVENT_SECONDS)
    return max(natural_end, last.start + MIN_EVENT_SECONDS)


def render_ass(
    lines: Sequence[SubtitleLine],
    style: SubtitleStyle,
    *,
    play_res_x: int,
    play_res_y: int,
    hold_seconds: float = DEFAULT_HOLD_SECONDS,
    title: str = "brainrot-tts",
) -> str:
    """Render grouped words as a complete ASS file."""

    if not lines:
        raise SubtitleError("No subtitle lines to render.")
    if play_res_x <= 0 or play_res_y <= 0:
        raise SubtitleError("Video resolution must be positive to render subtitles.")

    alignment = style.alignment
    margin_v = style.margin_bottom

    header = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "Collisions: Normal",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
    ]

    if style.box:
        header.append(
            _style_line(
                "BrainrotBox",
                font=style.font,
                size=style.font_size,
                # Text in the box layer is fully transparent; only its box is
                # visible.  The text layer above draws the actual glyphs.
                primary="&HFF000000",
                secondary="&HFF000000",
                outline=style.box_color,
                back=style.box_color,
                bold=style.bold,
                border_style=3,
                outline_width=style.box_padding,
                alignment=alignment,
                margin_lr=style.margin_lr,
                margin_v=margin_v,
            )
        )

    header.append(
        _style_line(
            "BrainrotText",
            font=style.font,
            size=style.font_size,
            primary=style.text_color,
            secondary=style.text_color,
            outline=style.outline_color,
            back="&H00000000",
            bold=style.bold,
            border_style=1,
            outline_width=style.outline_width,
            alignment=alignment,
            margin_lr=style.margin_lr,
            margin_v=margin_v,
        )
    )
    header.extend(
        [
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    events: list[tuple[int, int, int, str, str]] = []
    for line_index, line in enumerate(lines):
        line_start = max(0.0, line.words[0].start)
        line_end = _line_end_time(lines, line_index, hold_seconds)
        cursor = line_start

        # One box event per line is enough; the per-word events above it only
        # change colour, not geometry.
        if style.box:
            box_start_cs = _to_centiseconds(line_start)
            box_end_cs = _to_centiseconds(line_end)
            if box_end_cs <= box_start_cs:
                box_end_cs = box_start_cs + 1
            events.append((box_start_cs, box_end_cs, 0, "BrainrotBox", _plain_line(line)))

        for word_index, word in enumerate(line.words):
            start = max(cursor, word.start, line_start)
            if word_index + 1 < len(line.words):
                end = max(line.words[word_index + 1].start, start + MIN_EVENT_SECONDS)
            else:
                end = max(line_end, start + MIN_EVENT_SECONDS)
            cursor = start

            start_cs = _to_centiseconds(start)
            end_cs = _to_centiseconds(end)
            if end_cs <= start_cs:
                end_cs = start_cs + 1

            events.append(
                (
                    start_cs,
                    end_cs,
                    1,
                    "BrainrotText",
                    _active_line(line, word_index, style),
                )
            )

    events.sort(key=lambda event: (event[0], event[2], event[1]))
    body = [
        _dialogue(layer, start, end, style_name, text)
        for start, end, layer, style_name, text in events
    ]
    return "\n".join(header + body) + "\n"
