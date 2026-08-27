from __future__ import annotations

import wave
from pathlib import Path

import pytest


@pytest.mark.parametrize("fixture_name", ["generated_image_path", "alternate_image_path"])
def test_generated_image_fixture_is_rgb(request: pytest.FixtureRequest, fixture_name: str) -> None:
    Image = pytest.importorskip("PIL.Image")
    path: Path = request.getfixturevalue(fixture_name)
    with Image.open(path) as image:
        assert image.mode == "RGB"


@pytest.mark.parametrize("fixture_name", ["generated_image_path", "alternate_image_path"])
def test_generated_image_fixture_has_model_native_dimensions(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    Image = pytest.importorskip("PIL.Image")
    path: Path = request.getfixturevalue(fixture_name)
    with Image.open(path) as image:
        assert image.size == (224, 224)


def test_generated_images_are_visually_distinct(
    generated_image_path: Path, alternate_image_path: Path
) -> None:
    Image = pytest.importorskip("PIL.Image")
    with Image.open(generated_image_path) as first, Image.open(alternate_image_path) as second:
        assert list(first.get_flattened_data()) != list(second.get_flattened_data())


@pytest.mark.parametrize("fixture_name", ["generated_silence_wav", "generated_tone_wav"])
def test_generated_wav_is_mono(request: pytest.FixtureRequest, fixture_name: str) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1


@pytest.mark.parametrize("fixture_name", ["generated_silence_wav", "generated_tone_wav"])
def test_generated_wav_uses_whisper_sample_rate(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16_000


@pytest.mark.parametrize("fixture_name", ["generated_silence_wav", "generated_tone_wav"])
def test_generated_wav_uses_pcm16(request: pytest.FixtureRequest, fixture_name: str) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    with wave.open(str(path), "rb") as wav:
        assert wav.getsampwidth() == 2


@pytest.mark.parametrize("fixture_name", ["generated_silence_wav", "generated_tone_wav"])
def test_generated_wav_duration_is_bounded(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    with wave.open(str(path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    assert 0.9 <= duration <= 1.5
