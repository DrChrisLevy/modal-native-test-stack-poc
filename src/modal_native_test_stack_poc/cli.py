"""Command-line interface for the Modal execution harness."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from modal_native_test_stack_poc.remote.config import RuntimeConfig
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal-native-test-stack-poc",
        description=(
            "Run the real multimodal application, services, tests, shells, and agents on Modal"
        ),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build", help="prewarm runtime and Sidecar Images")
    build.add_argument("--force", action="store_true", help="bypass Modal Image caches")
    build.add_argument("--agent", action="store_true", help="also build the Codex layer")

    seed = subparsers.add_parser("seed", help="download real model snapshots into the Volume")
    seed.add_argument("--force", action="store_true", help="replace matching snapshots")

    subparsers.add_parser("check-models", help="show the committed model-Volume manifest")

    delete_models = subparsers.add_parser(
        "delete-models",
        help="irreversibly delete only the named model Volume",
    )
    delete_models.add_argument("--yes", action="store_true", help="confirm irreversible deletion")

    test = subparsers.add_parser(
        "test",
        help="run pytest in one fresh full-stack Sandbox",
    )
    test.add_argument("--workers", type=int, default=4)
    test.add_argument("--no-lint", action="store_true")
    test.add_argument("--force-build", action="store_true")
    test.add_argument("--keep-on-failure", action="store_true")
    test.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="focused pytest arguments after --",
    )

    shell = subparsers.add_parser("shell", help="open a fresh remote development shell")
    shell.add_argument("--command", help="run a command instead of attaching a terminal")
    shell.add_argument("--no-services", action="store_true")
    shell.add_argument("--network", action="store_true", help="allow public network egress")
    shell.add_argument("--secret", action="append", default=[], help="named Modal Secret")

    api = subparsers.add_parser("api", help="serve FastAPI and attach a remote shell")
    api.add_argument("--network", action="store_true", help="allow public network egress")

    agent = subparsers.add_parser("agent", help="run Codex or another agent in the full stack")
    agent.add_argument("--prompt", help="one-shot Codex prompt; omit for the interactive TUI")
    agent.add_argument("--command", help="custom agent command instead of bundled Codex")
    agent.add_argument("--secret", action="append", default=[], help="named Modal Secret")
    agent.add_argument("--offline", action="store_true", help="block public network egress")

    subparsers.add_parser("status", help="list live resources owned by this project")
    cleanup = subparsers.add_parser(
        "cleanup", help="terminate live resources owned by this project"
    )
    cleanup.add_argument("--run-id", help="limit cleanup to one tagged run")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = RuntimeConfig.from_environment()

    # Imports stay lazy so MODAL_PROFILE and MODAL_ENVIRONMENT supplied by the
    # caller are resolved before the Modal SDK initializes its client.
    if arguments.action == "build":
        from modal_native_test_stack_poc.remote.runner import prewarm

        prewarm(config, force=arguments.force, include_agent=arguments.agent)
        return 0
    if arguments.action == "seed":
        import modal

        from modal_native_test_stack_poc.remote.images import runtime_image_definition
        from modal_native_test_stack_poc.remote.models import seed_model_volume
        from modal_native_test_stack_poc.remote.runner import lookup_app

        config.validate()
        app = lookup_app(config)
        with modal.enable_output():
            image = runtime_image_definition(config.root).build(app)
        return seed_model_volume(app, config, image, force=arguments.force)
    if arguments.action == "check-models":
        from modal_native_test_stack_poc.remote.models import check_model_volume

        return check_model_volume(config)
    if arguments.action == "delete-models":
        from modal_native_test_stack_poc.remote.models import delete_model_volume

        if not arguments.yes:
            raise ModalNativeTestStackError("delete-models is irreversible; pass --yes to confirm")
        delete_model_volume(config)
        return 0
    if arguments.action == "test":
        from modal_native_test_stack_poc.remote.runner import run_tests

        pytest_args = list(arguments.pytest_args)
        if pytest_args[:1] == ["--"]:
            pytest_args.pop(0)
        return run_tests(
            config,
            worker_count=arguments.workers,
            pytest_args=pytest_args,
            include_lint=not arguments.no_lint,
            force_build=arguments.force_build,
            keep_on_failure=arguments.keep_on_failure,
        )
    if arguments.action == "shell":
        from modal_native_test_stack_poc.remote.runner import run_shell

        return run_shell(
            config,
            command=arguments.command,
            with_services=not arguments.no_services,
            allow_network=arguments.network,
            secret_names=arguments.secret,
        )
    if arguments.action == "api":
        from modal_native_test_stack_poc.remote.runner import run_api

        return run_api(config, allow_network=arguments.network)
    if arguments.action == "agent":
        from modal_native_test_stack_poc.remote.runner import run_agent

        secrets = list(arguments.secret)
        default_secret = os.getenv("MODAL_NATIVE_TEST_STACK_POC_AGENT_SECRET")
        if default_secret and default_secret not in secrets:
            secrets.append(default_secret)
        return run_agent(
            config,
            prompt=arguments.prompt,
            command=arguments.command,
            secret_names=secrets,
            allow_network=not arguments.offline,
        )
    if arguments.action == "status":
        from modal_native_test_stack_poc.remote.runner import show_status

        show_status(config)
        return 0
    if arguments.action == "cleanup":
        from modal_native_test_stack_poc.remote.runner import cleanup

        cleanup(config, run_id=arguments.run_id)
        return 0
    raise AssertionError(arguments.action)


def main() -> None:
    try:
        raise SystemExit(run_cli())
    except KeyboardInterrupt:
        print("Interrupted; owned session teardown was requested.", file=sys.stderr)
        raise SystemExit(130) from None
    except ModalNativeTestStackError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
