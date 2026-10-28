from __future__ import annotations

from pathlib import Path

from brainrot_tts.cli import _config_from_args, _validate_args, build_parser, main


def parse(*args: str):
    parser = build_parser()
    return parser.parse_args(list(args))


def test_defaults_and_language_inference() -> None:
    args = parse("-i", "text.txt", "-v", "bg.mp4", "-V", "bm_george")
    _validate_args(args)
    config = _config_from_args(args)
    assert config.language == "en-gb"
    assert config.output_path == Path("text.mp4")
    assert config.work_dir == Path("text.work")
    assert config.voice == "bm_george"
    assert config.words_per_line == 4
    assert config.box is True


def test_explicit_language_wins() -> None:
    args = parse("-i", "text.txt", "-v", "bg.mp4", "-V", "af_heart", "--lang", "en-gb")
    config = _config_from_args(args)
    assert config.language == "en-gb"


def test_list_voices_does_not_require_input(capsys) -> None:
    assert main(["--list-voices"]) == 0
    output = capsys.readouterr().out
    assert "af_heart" in output
    assert "American English" in output
