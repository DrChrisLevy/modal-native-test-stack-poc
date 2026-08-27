"""Small process helpers for Modal Sandbox commands."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, TextIO

from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError


@dataclass(frozen=True, slots=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


def read_process(process: Any) -> ProcessResult:
    """Drain both pipes concurrently so a chatty child cannot deadlock."""

    stdout: list[str] = []
    stderr: list[str] = []

    def drain(stream: Any, destination: list[str]) -> None:
        destination.append(stream.read())

    threads = (
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
    )
    for thread in threads:
        thread.start()
    process.wait()
    for thread in threads:
        thread.join()
    return ProcessResult((), process.returncode, "".join(stdout), "".join(stderr))


def capture_process(process: Any, *, label: str) -> str:
    result = read_process(process)
    if result.returncode != 0:
        raise ModalNativeTestStackError(
            f"{label} failed with exit code {result.returncode}\n{result.output[-16_000:]}"
        )
    return result.stdout


def stream_process(process: Any, *, prefix: str = "") -> ProcessResult:
    """Mirror remote output live while retaining it for summaries and diagnostics."""

    output_lock = threading.Lock()
    stdout: list[str] = []
    stderr: list[str] = []

    def pump(stream: Any, destination: TextIO, captured: list[str]) -> None:
        at_line_start = True
        try:
            for chunk in stream:
                captured.append(chunk)
                with output_lock:
                    if not prefix:
                        print(chunk, end="", file=destination, flush=True)
                        continue
                    pieces = chunk.splitlines(keepends=True)
                    for piece in pieces:
                        if at_line_start:
                            print(prefix, end="", file=destination)
                        print(piece, end="", file=destination, flush=True)
                        at_line_start = piece.endswith(("\n", "\r"))
        except Exception:
            # The process return code remains authoritative. Modal may close a
            # stream before the local iterator receives its final sentinel.
            return

    threads = (
        threading.Thread(
            target=pump,
            args=(process.stdout, sys.stdout, stdout),
            daemon=True,
        ),
        threading.Thread(
            target=pump,
            args=(process.stderr, sys.stderr, stderr),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    process.wait()
    for thread in threads:
        thread.join()
    return ProcessResult((), process.returncode, "".join(stdout), "".join(stderr))
