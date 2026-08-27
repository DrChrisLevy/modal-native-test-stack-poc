"""Direct local-to-Modal development, test, API, shell, and agent workflows."""

from __future__ import annotations

import json
import shlex
import sys
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

import modal

from modal_native_test_stack_poc.remote.config import (
    OWNER_TAG_KEY,
    OWNER_TAG_VALUE,
    REMOTE_ARTIFACTS,
    RuntimeConfig,
)
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError
from modal_native_test_stack_poc.remote.images import (
    BuiltImages,
    build_images,
    ensure_supported_modal,
)
from modal_native_test_stack_poc.remote.models import read_model_state
from modal_native_test_stack_poc.remote.processes import ProcessResult
from modal_native_test_stack_poc.remote.session import SandboxSession

AGENT_PROMPT_PATH = "/tmp/modal-native-test-stack-poc-prompt.txt"


def lookup_app(config: RuntimeConfig, *, create: bool = True) -> modal.App:
    ensure_supported_modal()
    return modal.App.lookup(
        config.app_name,
        environment_name=config.environment_name,
        create_if_missing=create,
    )


def prewarm(config: RuntimeConfig, *, force: bool, include_agent: bool) -> BuiltImages:
    app = lookup_app(config)
    return build_images(app, config, force=force, include_agent=include_agent)


def _terminate_session(session: SandboxSession) -> None:
    warnings = session.terminate()
    if warnings:
        print("Teardown warnings: " + "; ".join(warnings), file=sys.stderr)


def _run_pytest(
    session: SandboxSession,
    *,
    worker_count: int,
    native_threads: int,
    pytest_args: Sequence[str],
    selection_args: Sequence[str],
    enforce_coverage: bool,
) -> ProcessResult:
    if session.sandbox is None:
        raise ModalNativeTestStackError("test Sandbox was not created")
    junit = f"{REMOTE_ARTIFACTS}/junit.xml"
    coverage_xml = f"{REMOTE_ARTIFACTS}/coverage.xml"
    arguments = [
        "pytest",
        "-ra",
        "--durations=15",
        f"--junitxml={junit}",
        "--cov=modal_native_test_stack_poc",
        "--cov-report=term-missing",
        f"--cov-report=xml:{coverage_xml}",
        f"--cov-fail-under={80 if enforce_coverage else 0}",
    ]
    if worker_count > 1:
        arguments.extend(("-n", str(worker_count), "--dist", "loadgroup"))
    arguments.extend(pytest_args or selection_args)
    return session.run(
        *arguments,
        prefix="[pytest] ",
        environment={
            "COVERAGE_FILE": f"{REMOTE_ARTIFACTS}/.coverage",
            "OMP_NUM_THREADS": str(native_threads),
            "MKL_NUM_THREADS": str(native_threads),
        },
    )


def _run_timed_pytest(
    session: SandboxSession,
    *,
    worker_count: int,
    native_threads: int,
    pytest_args: Sequence[str],
    selection_args: Sequence[str],
    enforce_coverage: bool,
) -> tuple[ProcessResult, float]:
    started = time.monotonic()
    result = _run_pytest(
        session,
        worker_count=worker_count,
        native_threads=native_threads,
        pytest_args=pytest_args,
        selection_args=selection_args,
        enforce_coverage=enforce_coverage,
    )
    return result, time.monotonic() - started


def _run_lint(session: SandboxSession) -> ProcessResult:
    return session.run(
        "bash",
        "-c",
        "ruff check src tests && ruff format --check src tests",
        prefix="[lint] ",
    )


def _run_timed_lint(session: SandboxSession) -> tuple[ProcessResult, float]:
    started = time.monotonic()
    return _run_lint(session), time.monotonic() - started


def _collect_artifacts(
    session: SandboxSession,
    output: Path,
) -> tuple[bool, int, float]:
    output.mkdir(parents=True, exist_ok=True)
    if session.sandbox is None:
        raise ModalNativeTestStackError("test Sandbox is missing")
    junit = output / "junit.xml"
    coverage = output / "coverage.xml"
    try:
        session.sandbox.filesystem.copy_to_local(f"{REMOTE_ARTIFACTS}/junit.xml", junit)
        session.sandbox.filesystem.copy_to_local(
            f"{REMOTE_ARTIFACTS}/coverage.xml",
            coverage,
        )
    except Exception as error:
        print(f"Artifact collection warning: {type(error).__name__}: {error}", file=sys.stderr)
        return False, 0, 0.0

    try:
        root = ET.parse(junit).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        test_count = int(root.get("tests", 0)) or sum(
            int(suite.get("tests", 0)) for suite in suites
        )
        testcase_seconds = sum(float(case.get("time", 0.0)) for case in root.iter("testcase"))
    except (ET.ParseError, OSError, TypeError, ValueError) as error:
        print(f"JUnit parse warning: {type(error).__name__}: {error}", file=sys.stderr)
        return False, 0, 0.0
    return True, test_count, testcase_seconds


def run_tests(
    config: RuntimeConfig,
    *,
    worker_count: int,
    pytest_args: Sequence[str] = (),
    include_lint: bool = True,
    force_build: bool = False,
    keep_on_failure: bool = False,
    selection_args: Sequence[str] = (),
    enforce_coverage: bool = True,
) -> int:
    if worker_count < 1:
        raise ModalNativeTestStackError("--workers must be at least one")

    overall_started = time.monotonic()
    phase_seconds: dict[str, float] = {}
    phase_started = time.monotonic()
    config.validate()
    read_model_state(config)
    app = lookup_app(config)
    images = build_images(app, config, force=force_build)
    phase_seconds["image_resolution"] = time.monotonic() - phase_started

    run_id = uuid.uuid4().hex
    artifact_dir = config.artifacts_root / run_id
    native_threads = max(1, int(config.cpu) // worker_count)
    test_session = SandboxSession(
        app,
        config,
        images.runtime,
        images.services,
        role="test",
        run_id=run_id,
    )
    failed = True
    return_code = 1
    summary: dict[str, object] | None = None
    try:
        phase_started = time.monotonic()
        test_session.create()
        assert test_session.sandbox is not None
        print(f"Main Sandbox ready: {test_session.sandbox.object_id}")
        phase_seconds["sandbox_creation"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        test_session.attach_services()
        phase_seconds["sidecar_attachment"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        test_session.wait_until_usable()
        print("Service stack ready")
        phase_seconds["service_readiness"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        bootstrap = test_session.run_captured(
            "python",
            "-c",
            "import asyncio, os; "
            "from modal_native_test_stack_poc.application import PostgresAssetRepository; "
            "repository=PostgresAssetRepository(os.environ['MULTIMODAL_POSTGRES_URL']); "
            "asyncio.run(repository.initialize())",
            timeout=120,
        )
        if bootstrap.returncode != 0:
            raise ModalNativeTestStackError(
                f"PostgreSQL schema bootstrap failed\n{bootstrap.output[-16_000:]}"
            )
        phase_seconds["schema_bootstrap"] = time.monotonic() - phase_started

        pytest_result: ProcessResult
        pytest_seconds: float
        lint_result: ProcessResult | None = None
        lint_seconds: float | None = None
        phase_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2 if include_lint else 1) as executor:
            pytest_future = executor.submit(
                _run_timed_pytest,
                test_session,
                worker_count=worker_count,
                native_threads=native_threads,
                pytest_args=pytest_args,
                selection_args=selection_args,
                enforce_coverage=enforce_coverage,
            )
            lint_future = executor.submit(_run_timed_lint, test_session) if include_lint else None
            pytest_result, pytest_seconds = pytest_future.result()
            print(f"pytest exited {pytest_result.returncode} in {pytest_seconds:.1f}s")
            if lint_future is not None:
                lint_result, lint_seconds = lint_future.result()
                print(f"lint exited {lint_result.returncode} in {lint_seconds:.1f}s")
        phase_seconds["tests_and_lint"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        artifacts_ok, test_count, testcase_seconds = _collect_artifacts(
            test_session,
            artifact_dir,
        )
        phase_seconds["coverage_and_artifacts"] = time.monotonic() - phase_started
        failed = (
            pytest_result.returncode != 0
            or bool(lint_result and lint_result.returncode != 0)
            or not artifacts_ok
        )
        return_code = 1 if failed else 0
        summary = {
            "run_id": run_id,
            "result": "failed" if failed else "passed",
            "topology": "single-stack-xdist",
            "sandboxes": 1,
            "sidecars": 3,
            "xdist_workers": worker_count,
            "xdist_distribution": "loadgroup" if worker_count > 1 else "none",
            "xdist_groups": ["audio", "core", "image", "text"],
            "native_threads_per_worker": native_threads,
            "test_count": test_count,
            "testcase_seconds": round(testcase_seconds, 3),
            "pytest_returncode": pytest_result.returncode,
            "pytest_seconds": round(pytest_seconds, 3),
            "lint_returncode": lint_result.returncode if lint_result else None,
            "lint_seconds": round(lint_seconds, 3) if lint_seconds is not None else None,
        }
    finally:
        phase_started = time.monotonic()
        if keep_on_failure and failed:
            print(f"Keeping failed run {run_id}; clean it with cleanup --run-id {run_id}")
            if test_session.sandbox is not None:
                test_session.sandbox.detach()
        else:
            _terminate_session(test_session)
        phase_seconds["teardown_submit"] = time.monotonic() - phase_started

    assert summary is not None
    summary["seconds"] = round(time.monotonic() - overall_started, 3)
    summary["phases"] = {name: round(seconds, 3) for name, seconds in phase_seconds.items()}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Phase timings:")
    for name, seconds in phase_seconds.items():
        print(f"  {name.replace('_', ' '):30} {seconds:6.2f}s")
    print(f"Artifacts: {artifact_dir}")
    print(f"Run {run_id}: {'FAIL' if failed else 'PASS'} in {summary['seconds']:.1f}s")
    return return_code


def run_smoke(config: RuntimeConfig, *, workers: int = 3) -> int:
    """Exercise all real model and service tests, omitting only pure unit tests."""

    return run_tests(
        config,
        worker_count=workers,
        include_lint=False,
        selection_args=("-m", "model or services or e2e"),
        enforce_coverage=False,
    )


def run_shell(
    config: RuntimeConfig,
    *,
    command: str | None,
    with_services: bool,
    allow_network: bool,
    secret_names: Sequence[str],
) -> int:
    config.validate()
    if with_services:
        read_model_state(config)
    app = lookup_app(config)
    images = build_images(app, config)
    session = SandboxSession(
        app,
        config,
        images.runtime,
        images.services,
        role="shell",
        run_id=uuid.uuid4().hex,
        with_services=with_services,
        allow_network=allow_network,
        secret_names=tuple(secret_names),
    )
    try:
        session.create()
        session.attach_services()
        session.wait_until_usable()
        assert session.sandbox is not None
        print(f"Remote development environment ready: {session.sandbox.object_id}")
        if command:
            return session.run("bash", "-c", command).returncode
        return session.open_terminal()
    finally:
        _terminate_session(session)


def run_api(config: RuntimeConfig, *, allow_network: bool = False) -> int:
    config.validate()
    read_model_state(config)
    app = lookup_app(config)
    images = build_images(app, config)
    session = SandboxSession(
        app,
        config,
        images.runtime,
        images.services,
        role="api",
        run_id=uuid.uuid4().hex,
        allow_network=allow_network,
        encrypted_ports=(8000,),
    )
    try:
        session.create()
        session.attach_services()
        session.wait_until_usable()
        session.capture(
            "bash",
            "-c",
            "nohup uvicorn modal_native_test_stack_poc.application.api:create_app --factory "
            "--host 0.0.0.0 --port 8000 "
            f">{REMOTE_ARTIFACTS}/api.log 2>&1 </dev/null & "
            f"echo $! >{REMOTE_ARTIFACTS}/api.pid",
            label="API launch",
        )
        session.capture(
            "bash",
            "-c",
            "for attempt in $(seq 1 180); do "
            "curl -fsS http://127.0.0.1:8000/health/ready >/dev/null && exit 0; "
            f"sleep 1; done; cat {REMOTE_ARTIFACTS}/api.log; exit 1",
            label="API readiness",
            timeout=240,
        )
        assert session.sandbox is not None
        tunnel = session.sandbox.tunnels(timeout=60)[8000]
        print(f"API ready: {tunnel.url}")
        print("A remote shell is attached; exit it to tear down the API and all Sidecars.")
        return session.open_terminal()
    finally:
        _terminate_session(session)


def run_agent(
    config: RuntimeConfig,
    *,
    prompt: str | None,
    command: str | None,
    secret_names: Sequence[str],
    allow_network: bool,
) -> int:
    config.validate()
    read_model_state(config)
    app = lookup_app(config)
    images = build_images(app, config, include_agent=command is None)
    image = images.agent if command is None else images.runtime
    if image is None:
        raise ModalNativeTestStackError("agent Image was not built")
    run_id = uuid.uuid4().hex
    session = SandboxSession(
        app,
        config,
        image,
        images.services,
        role="agent",
        run_id=run_id,
        allow_network=allow_network,
        secret_names=tuple(secret_names),
    )
    try:
        session.create()
        session.attach_services()
        session.wait_until_usable()
        guidance = (
            "You are in an ephemeral Modal Sandbox at /workspace. PostgreSQL, Redis, and "
            "OpenSearch are ready at their service DNS names. Real Hugging Face snapshots "
            "are mounted read-only at /models and network model downloads are disabled. "
            "Do not use Docker or Compose. Run tests directly with pytest."
        )
        assert session.sandbox is not None
        session.sandbox.filesystem.write_text(guidance + "\n", "/workspace/AGENTS.override.md")
        session.capture(
            "bash",
            "-c",
            "git init -q && git config user.email modal-native-test-stack-poc@example.invalid && "
            "git config user.name 'Modal-Native Test Stack POC' && "
            "git add -A && git commit -qm baseline",
            label="ephemeral agent Git baseline",
        )
        if command:
            result = session.run("bash", "-c", command, prefix="[agent] ")
        elif prompt is None:
            process = session.sandbox.exec(
                "codex",
                "--sandbox",
                "danger-full-access",
                "--ask-for-approval",
                "never",
                workdir="/workspace",
                timeout=config.timeout_seconds,
                pty=True,
            )
            process.attach()
            process.wait()
            result = ProcessResult((), process.returncode, "", "")
        else:
            session.sandbox.filesystem.write_text(prompt, AGENT_PROMPT_PATH)
            result = session.run(
                "bash",
                "-c",
                'test -n "${OPENAI_API_KEY:-}" && '
                "printenv OPENAI_API_KEY | codex login --with-api-key >/dev/null && "
                "codex --sandbox danger-full-access --ask-for-approval never exec - "
                f"<{AGENT_PROMPT_PATH}",
                prefix="[agent] ",
            )
        session.capture(
            "bash",
            "-c",
            f"git add -N . && git diff --binary HEAD >{REMOTE_ARTIFACTS}/agent.patch",
            label="agent patch export",
        )
        output = config.artifacts_root / run_id
        output.mkdir(parents=True, exist_ok=True)
        session.sandbox.filesystem.copy_to_local(
            f"{REMOTE_ARTIFACTS}/agent.patch",
            output / "agent.patch",
        )
        print(f"Agent patch: {output / 'agent.patch'}")
        return result.returncode
    finally:
        _terminate_session(session)


def owned_sandboxes(config: RuntimeConfig, *, run_id: str | None = None) -> list[modal.Sandbox]:
    try:
        app = lookup_app(config, create=False)
    except modal.exception.NotFoundError:
        return []
    tags = {OWNER_TAG_KEY: OWNER_TAG_VALUE}
    if run_id:
        tags["modal-native-test-stack-poc-run"] = run_id
    return list(modal.Sandbox.list(app_id=app.app_id, tags=tags))


def show_status(config: RuntimeConfig) -> int:
    sandboxes = [sandbox for sandbox in owned_sandboxes(config) if sandbox.poll() is None]
    if not sandboxes:
        print("No live Modal-Native Test Stack POC Sandboxes.")
        return 0
    for sandbox in sandboxes:
        tags = sandbox.get_tags()
        print(
            f"{sandbox.object_id} role={tags.get('modal-native-test-stack-poc-role', '?')} "
            f"run={tags.get('modal-native-test-stack-poc-run', '?')}"
        )
    return len(sandboxes)


def cleanup(config: RuntimeConfig, *, run_id: str | None = None) -> int:
    sandboxes = owned_sandboxes(config, run_id=run_id)
    for sandbox in sandboxes:
        try:
            sandbox.terminate(wait=True)
        finally:
            sandbox.detach()
    scope = f" for run {shlex.quote(run_id)}" if run_id else ""
    print(f"Terminated {len(sandboxes)} owned Sandbox(es){scope}.")
    return len(sandboxes)
