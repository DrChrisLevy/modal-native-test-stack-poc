from __future__ import annotations

from modal_native_test_stack_poc.cli import build_parser


def test_test_command_defaults_to_learned_duration_scheduler() -> None:
    arguments = build_parser().parse_args(["test"])

    assert arguments.scheduler == "duration"
    assert arguments.learn is True


def test_test_command_can_select_count_control_scheduler() -> None:
    arguments = build_parser().parse_args(["test", "--scheduler", "count"])

    assert arguments.scheduler == "count"


def test_test_command_can_freeze_duration_history() -> None:
    arguments = build_parser().parse_args(["test", "--no-learn"])

    assert arguments.learn is False


def test_focused_pytest_arguments_remain_after_scheduler_options() -> None:
    arguments = build_parser().parse_args(
        ["test", "--scheduler", "duration", "--no-learn", "--", "tests/model", "-x"]
    )

    assert arguments.pytest_args == ["--", "tests/model", "-x"]
