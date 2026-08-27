from __future__ import annotations

from modal_native_test_stack_poc.cli import build_parser


def test_test_command_defaults_to_four_grouped_xdist_workers() -> None:
    arguments = build_parser().parse_args(["test"])

    assert arguments.workers == 4


def test_test_command_can_select_worker_count() -> None:
    arguments = build_parser().parse_args(["test", "--workers", "3"])

    assert arguments.workers == 3


def test_focused_pytest_arguments_remain_after_worker_option() -> None:
    arguments = build_parser().parse_args(["test", "--workers", "3", "--", "tests/model", "-x"])

    assert arguments.pytest_args == ["--", "tests/model", "-x"]
