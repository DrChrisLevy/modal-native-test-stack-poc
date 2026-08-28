from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError
from modal_native_test_stack_poc.remote.processes import (
    capture_process,
    read_process,
    stream_process,
)


def _process(source: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_read_process_drains_stdout_and_stderr() -> None:
    result = read_process(_process("import sys; print('out'); print('err', file=sys.stderr)"))
    assert result.returncode == 0
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_read_process_preserves_failure_code() -> None:
    result = read_process(_process("raise SystemExit(7)"))
    assert result.returncode == 7


def test_capture_process_returns_stdout() -> None:
    assert capture_process(_process("print('captured')"), label="example") == "captured\n"


def test_capture_process_raises_actionable_error() -> None:
    with pytest.raises(ModalNativeTestStackError, match="example failed with exit code 3"):
        capture_process(
            _process("import sys; print('details', file=sys.stderr); raise SystemExit(3)"),
            label="example",
        )


def test_capture_process_includes_child_error() -> None:
    with pytest.raises(ModalNativeTestStackError, match="details"):
        capture_process(
            _process("import sys; print('details', file=sys.stderr); raise SystemExit(2)"),
            label="example",
        )


def test_stream_process_mirrors_and_captures_both_streams(capsys) -> None:
    result = stream_process(
        _process("import sys; print('out'); print('err', file=sys.stderr)"), prefix="[remote] "
    )
    captured = capsys.readouterr()
    assert "[remote] out" in captured.out
    assert "[remote] err" in captured.err
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_stream_process_without_prefix_preserves_output(capsys) -> None:
    result = stream_process(_process("print('plain')"))
    assert capsys.readouterr().out == "plain\n"
    assert result.returncode == 0


def test_stream_process_transforms_complete_stdout_lines_and_preserves_raw_output(
    capsys,
) -> None:
    process = SimpleNamespace(
        stdout=iter(['{"one":', '1}\n{"two":2}\n', '{"three":3}']),
        stderr=iter(["codex warning\n"]),
        returncode=0,
        wait=lambda: None,
    )

    result = stream_process(
        process,
        stdout_line_transform=lambda line: f"parsed:{line}",
    )

    captured = capsys.readouterr()
    assert captured.out == ('parsed:{"one":1}\nparsed:{"two":2}\nparsed:{"three":3}')
    assert captured.err == "codex warning\n"
    assert result.stdout == '{"one":1}\n{"two":2}\n{"three":3}'
    assert result.stderr == "codex warning\n"
