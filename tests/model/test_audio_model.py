from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import value_of


def _text(output: Any) -> str:
    return output if isinstance(output, str) else str(value_of(output, "text", "transcript"))


@pytest.mark.model
def test_whisper_accepts_a_generated_pcm_wav(registry, generated_silence_wav: Path) -> None:
    assert isinstance(_text(registry.transcribe(generated_silence_wav)), str)


@pytest.mark.model
def test_whisper_handles_generated_non_speech_audio(registry, generated_tone_wav: Path) -> None:
    assert isinstance(_text(registry.transcribe(generated_tone_wav)), str)


@pytest.mark.model
def test_whisper_silence_output_is_bounded(registry, generated_silence_wav: Path) -> None:
    transcript = _text(registry.transcribe(generated_silence_wav))
    assert len(transcript) < 120


@pytest.mark.model
def test_whisper_tone_output_is_bounded(registry, generated_tone_wav: Path) -> None:
    transcript = _text(registry.transcribe(generated_tone_wav))
    assert len(transcript) < 120


@pytest.mark.model
def test_whisper_generation_is_deterministic(registry, generated_silence_wav: Path) -> None:
    first = _text(registry.transcribe(generated_silence_wav))
    second = _text(registry.transcribe(generated_silence_wav))
    assert first == second
