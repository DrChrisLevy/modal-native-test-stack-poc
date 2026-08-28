from __future__ import annotations

from modal_native_test_stack_poc.remote.runner import CodexEventRenderer


def test_codex_json_events_render_a_compact_transcript_without_command_output() -> None:
    renderer = CodexEventRenderer(color=False, width=80)
    assert (
        renderer.format_event(
            {
                "type": "item.started",
                "item": {
                    "type": "command_execution",
                    "command": '/bin/bash -lc "python -m pytest -q"',
                },
            }
        )
        == "\n• Running python -m pytest -q\n"
    )
    assert (
        renderer.format_event(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "status": "completed",
                    "aggregated_output": "large test output that should not be rendered",
                },
            }
        )
        == "  └ ✓ completed\n"
    )
    assert (
        renderer.format_event(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Everything passed."},
            }
        )
        == "\n• Everything passed.\n"
    )
    assert renderer.format_event(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 120,
                "cached_input_tokens": 80,
                "output_tokens": 30,
                "reasoning_output_tokens": 5,
            },
        }
    ) == (
        f"\n{renderer.rule('Completed')}\n"
        "Token usage: total=70 input=40 (+80 cached) output=30 (reasoning 5)\n"
    )


def test_codex_json_renderer_ignores_unknown_events_and_preserves_invalid_lines() -> None:
    renderer = CodexEventRenderer(color=False, width=80)
    assert renderer('{"type":"thread.started","thread_id":"abc"}\n') is None
    assert renderer('{"type":"future.event"}\n') is None
    assert renderer("codex startup warning\n") == "codex startup warning\n"
    assert renderer("\n") is None


def test_codex_json_renderer_formats_failures() -> None:
    renderer = CodexEventRenderer(color=False, width=80)
    assert renderer('{"type":"turn.failed","error":{"message":"model unavailable"}}\n') == (
        f"\n{renderer.rule('Failed')}\nmodel unavailable\n"
    )
    assert renderer('{"type":"error","message":"authentication failed"}\n') == (
        f"\n{renderer.rule('Error')}\nauthentication failed\n"
    )
