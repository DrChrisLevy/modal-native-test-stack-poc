"""Direct local-to-Modal development, test, API, shell, and agent workflows."""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

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
from modal_native_test_stack_poc.remote.sharding import (
    TestShard,
    load_durations,
    merge_durations,
    parse_collected_nodeids,
    parse_junit_durations,
    plan_shards,
    save_durations,
)

MAX_PARALLEL_SETUP = 12
AGENT_PROMPT_PATH = "/tmp/modal-native-test-stack-poc-prompt.txt"


@dataclass(frozen=True, slots=True)
class ShardExecution:
    shard: TestShard
    result: ProcessResult
    seconds: float


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


def _create_sessions(sessions: Sequence[SandboxSession]) -> None:
    created: list[SandboxSession] = []
    try:
        with ThreadPoolExecutor(max_workers=min(len(sessions), MAX_PARALLEL_SETUP)) as executor:
            futures = {executor.submit(session.create): session for session in sessions}
            for future in as_completed(futures):
                session = futures[future]
                future.result()
                created.append(session)
                assert session.sandbox is not None
                print(f"Main Sandbox ready: {session.role} ({session.sandbox.object_id})")
    except BaseException:
        for session in sessions:
            session.terminate()
        raise


def _attach_sessions(sessions: Sequence[SandboxSession]) -> None:
    # Sidecar creation is an alpha control-plane API. Serialize attachments
    # across all Sandboxes; concurrent calls can stall indefinitely.
    for session in sessions:
        session.attach_services()


def _ready_sessions(sessions: Sequence[SandboxSession]) -> None:
    # Once attached, independently wait for every stack in parallel.
    with ThreadPoolExecutor(max_workers=min(len(sessions), MAX_PARALLEL_SETUP)) as executor:
        futures = {executor.submit(session.wait_until_usable): session for session in sessions}
        for future in as_completed(futures):
            future.result()
            print(f"Service stack ready: {futures[future].role}")


def _attach_and_ready(sessions: Sequence[SandboxSession]) -> None:
    _attach_sessions(sessions)
    _ready_sessions(sessions)


def _terminate_sessions(sessions: Sequence[SandboxSession]) -> None:
    warnings: list[str] = []
    if sessions:
        with ThreadPoolExecutor(max_workers=min(len(sessions), MAX_PARALLEL_SETUP)) as executor:
            for result in executor.map(lambda item: item.terminate(), sessions):
                warnings.extend(result)
    if warnings:
        print("Teardown warnings: " + "; ".join(warnings), file=sys.stderr)


def _run_shard(
    session: SandboxSession,
    shard: TestShard,
    *,
    focused_args: Sequence[str],
    selection_args: Sequence[str],
) -> ProcessResult:
    if session.sandbox is None:
        raise ModalNativeTestStackError("test Sandbox was not created")
    junit = f"{REMOTE_ARTIFACTS}/junit-{shard.index + 1}.xml"
    coverage_file = f"{REMOTE_ARTIFACTS}/.coverage-{shard.index + 1}"
    arguments = [
        "pytest",
        "-ra",
        "--durations=15",
        f"--junitxml={junit}",
        "--cov=modal_native_test_stack_poc",
        "--cov-report=",
        "--cov-fail-under=0",
    ]
    if focused_args:
        arguments.extend(focused_args)
    else:
        arguments.extend(selection_args)
        nodeid_file = f"{REMOTE_ARTIFACTS}/shard-{shard.index + 1}.txt"
        session.sandbox.filesystem.write_text("\n".join(shard.nodeids), nodeid_file)
        runner = (
            "import pathlib, subprocess, sys; "
            f"nodeids=pathlib.Path({nodeid_file!r}).read_text().splitlines(); "
            f"raise SystemExit(subprocess.call({arguments!r} + nodeids))"
        )
        arguments = ["python", "-c", runner]
    return session.run(
        *arguments,
        prefix=f"[shard {shard.index + 1}] ",
        environment={"COVERAGE_FILE": coverage_file},
    )


def _run_timed_shard(
    session: SandboxSession,
    shard: TestShard,
    *,
    focused_args: Sequence[str],
    selection_args: Sequence[str],
) -> ShardExecution:
    started = time.monotonic()
    result = _run_shard(
        session,
        shard,
        focused_args=focused_args,
        selection_args=selection_args,
    )
    return ShardExecution(shard, result, time.monotonic() - started)


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
    sessions: Sequence[SandboxSession],
    shards: Sequence[TestShard],
    output: Path,
    *,
    enforce_coverage: bool,
) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    primary = sessions[0]
    if primary.sandbox is None:
        raise ModalNativeTestStackError("primary test Sandbox is missing")
    with tempfile.TemporaryDirectory(prefix="modal-native-test-stack-poc-coverage-") as temporary:
        temporary_root = Path(temporary)
        for session, shard in zip(sessions, shards, strict=True):
            if session.sandbox is None:
                continue
            junit_remote = f"{REMOTE_ARTIFACTS}/junit-{shard.index + 1}.xml"
            coverage_remote = f"{REMOTE_ARTIFACTS}/.coverage-{shard.index + 1}"
            with suppress(Exception):
                session.sandbox.filesystem.copy_to_local(
                    junit_remote,
                    output / f"junit-{shard.index + 1}.xml",
                )
            local_coverage = temporary_root / f".coverage-{shard.index + 1}"
            try:
                session.sandbox.filesystem.copy_to_local(coverage_remote, local_coverage)
            except Exception:
                continue
            if session is not primary:
                primary.sandbox.filesystem.copy_from_local(local_coverage, coverage_remote)

        report_command = "coverage report" if enforce_coverage else "coverage report --fail-under=0"
        coverage = primary.run_captured(
            "bash",
            "-c",
            f"coverage combine {REMOTE_ARTIFACTS}/.coverage-* && "
            f"coverage xml --fail-under=0 -o {REMOTE_ARTIFACTS}/coverage.xml && " + report_command,
            timeout=300,
        )
        if coverage.output:
            print(coverage.output)
        if coverage.returncode == 0:
            primary.sandbox.filesystem.copy_to_local(
                f"{REMOTE_ARTIFACTS}/coverage.xml",
                output / "coverage.xml",
            )
            return True
        return False


def run_tests(
    config: RuntimeConfig,
    *,
    shard_count: int,
    pytest_args: Sequence[str] = (),
    include_lint: bool = True,
    force_build: bool = False,
    keep_on_failure: bool = False,
    selection_args: Sequence[str] = (),
    enforce_coverage: bool = True,
    scheduler: str = "duration",
    learn: bool = True,
) -> int:
    if shard_count < 1:
        raise ModalNativeTestStackError("--shards must be at least one")
    if scheduler not in {"count", "duration"}:
        raise ModalNativeTestStackError("scheduler must be 'count' or 'duration'")

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
    history_path = config.artifacts_root / "test-durations.json"
    effective_count = 1 if pytest_args else shard_count
    test_sessions = [
        SandboxSession(
            app,
            config,
            images.runtime,
            images.services,
            role=f"test-{index + 1}",
            run_id=run_id,
            ordinal=index,
        )
        for index in range(effective_count)
    ]
    lint_session = (
        SandboxSession(
            app,
            config,
            images.runtime,
            {},
            role="lint",
            run_id=run_id,
            ordinal=effective_count,
            with_services=False,
            mount_models=False,
        )
        if include_lint
        else None
    )
    all_sessions = [*test_sessions, *([lint_session] if lint_session else [])]
    failed = True
    return_code = 1
    summary: dict[str, object] | None = None
    try:
        phase_started = time.monotonic()
        _create_sessions(all_sessions)
        phase_seconds["sandbox_creation"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        _attach_sessions(test_sessions)
        phase_seconds["sidecar_attachment"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        _ready_sessions(test_sessions)
        phase_seconds["service_readiness"] = time.monotonic() - phase_started

        duration_history: dict[str, float] = {}
        nodeids: list[str] = []
        planner_weight_unit: str | None = None
        phase_started = time.monotonic()
        if pytest_args:
            shards = [TestShard(0, (), 0, 0, 0.0)]
        else:
            collection = test_sessions[0].run_captured(
                "pytest",
                "--collect-only",
                "-q",
                *selection_args,
                timeout=600,
            )
            if collection.returncode != 0:
                raise ModalNativeTestStackError(
                    f"pytest collection failed\n{collection.output[-16_000:]}"
                )
            nodeids = parse_collected_nodeids(collection.stdout)
            if not nodeids:
                raise ModalNativeTestStackError("pytest collected no node IDs")
            duration_history = load_durations(history_path)
            planner_history = duration_history if scheduler == "duration" else {}
            planner_weight_unit = "seconds" if planner_history else "test-count"
            shards = plan_shards(nodeids, effective_count, planner_history)
            print(f"Collected {len(nodeids)} tests into {len(shards)} isolated shard(s):")
            entry_label = "entry" if len(planner_history) == 1 else "entries"
            print(f"Scheduler: {scheduler} ({len(planner_history)} learned duration {entry_label})")
            for shard in shards:
                weight_suffix = "s" if planner_weight_unit == "seconds" else " test units"
                print(
                    f"  shard {shard.index + 1}: {shard.test_count} tests, "
                    f"{shard.file_count} files, planned weight "
                    f"{shard.estimated_seconds:.1f}{weight_suffix}"
                )
            extras = test_sessions[len(shards) :]
            _terminate_sessions(extras)
            test_sessions = test_sessions[: len(shards)]
            all_sessions = [*test_sessions, *([lint_session] if lint_session else [])]
        phase_seconds["test_collection_and_planning"] = time.monotonic() - phase_started

        executions: list[ShardExecution] = []
        lint_result: ProcessResult | None = None
        lint_seconds: float | None = None
        worker_count = len(shards) + int(lint_session is not None)
        phase_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            test_futures = {
                executor.submit(
                    _run_timed_shard,
                    session,
                    shard,
                    focused_args=pytest_args,
                    selection_args=selection_args,
                ): shard
                for session, shard in zip(test_sessions, shards, strict=True)
            }
            lint_future = executor.submit(_run_timed_lint, lint_session) if lint_session else None
            for future in as_completed(test_futures):
                execution = future.result()
                executions.append(execution)
                print(
                    f"shard {execution.shard.index + 1} exited "
                    f"{execution.result.returncode} in {execution.seconds:.1f}s"
                )
            if lint_future is not None:
                lint_result, lint_seconds = lint_future.result()
                print(f"lint exited {lint_result.returncode} in {lint_seconds:.1f}s")
        phase_seconds["tests_and_lint"] = time.monotonic() - phase_started

        executions.sort(key=lambda execution: execution.shard.index)
        results = [execution.result for execution in executions]
        phase_started = time.monotonic()
        coverage_ok = _collect_artifacts(
            test_sessions,
            shards,
            artifact_dir,
            enforce_coverage=enforce_coverage,
        )
        phase_seconds["coverage_and_artifacts"] = time.monotonic() - phase_started
        failed = (
            any(result.returncode != 0 for result in results)
            or bool(lint_result and lint_result.returncode != 0)
            or not coverage_ok
        )
        observed_durations: dict[str, float] = {}
        updated_history = duration_history
        history_error: str | None = None
        if nodeids:
            try:
                junit_paths = sorted(artifact_dir.glob("junit-*.xml"))
                observed_durations = parse_junit_durations(
                    junit_paths,
                    known_nodeids=nodeids,
                )
                save_durations(artifact_dir / "history-input.json", duration_history)
                save_durations(artifact_dir / "observed-durations.json", observed_durations)
                if learn and not selection_args and not failed:
                    active_files = {nodeid.partition("::")[0] for nodeid in nodeids}
                    updated_history = merge_durations(
                        duration_history,
                        observed_durations,
                        active_keys=active_files,
                    )
                    save_durations(history_path, updated_history)
            except Exception as error:
                history_error = f"{type(error).__name__}: {error}"
                print(f"Duration history warning: {history_error}", file=sys.stderr)
        return_code = 1 if failed else 0
        summary = {
            "run_id": run_id,
            "result": "failed" if failed else "passed",
            "scheduler": scheduler,
            "learn_requested": learn,
            "learn_applied": bool(
                nodeids and learn and not selection_args and not failed and not history_error
            ),
            "duration_history_entries": len(duration_history),
            "duration_history_entries_after": len(updated_history),
            "observed_duration_entries": len(observed_durations),
            "duration_history_error": history_error,
            "planner_weight_unit": planner_weight_unit,
            "shards": len(shards),
            "test_returncodes": [result.returncode for result in results],
            "shard_seconds": [round(execution.seconds, 3) for execution in executions],
            "shard_planned_weights": [round(shard.estimated_seconds, 3) for shard in shards],
            "lint_returncode": lint_result.returncode if lint_result else None,
            "lint_seconds": round(lint_seconds, 3) if lint_seconds is not None else None,
        }
    finally:
        phase_started = time.monotonic()
        if keep_on_failure and failed:
            print(f"Keeping failed run {run_id}; clean it with cleanup --run-id {run_id}")
            for session in all_sessions:
                if session.sandbox is not None:
                    session.sandbox.detach()
        else:
            _terminate_sessions(all_sessions)
        phase_seconds["teardown_submit"] = time.monotonic() - phase_started

    assert summary is not None
    shard_seconds = [execution.seconds for execution in executions]
    slowest_shard = max(shard_seconds, default=0.0)
    summary["observed_test_work_seconds"] = round(sum(observed_durations.values()), 3)
    summary["slowest_shard_seconds"] = round(slowest_shard, 3)
    summary["parallel_test_utilization"] = (
        round(sum(shard_seconds) / (len(shard_seconds) * slowest_shard), 4)
        if slowest_shard
        else None
    )
    summary["seconds"] = round(time.monotonic() - overall_started, 3)
    summary["phases"] = {name: round(seconds, 3) for name, seconds in phase_seconds.items()}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": 2,
        "scheduler": scheduler,
        "weight_unit": planner_weight_unit,
        "shards": [
            {
                "index": shard.index + 1,
                "planned_weight": round(shard.estimated_seconds, 3),
                "test_count": shard.test_count,
                "files": list(dict.fromkeys(nodeid.partition("::")[0] for nodeid in shard.nodeids)),
                "nodeids": list(shard.nodeids),
            }
            for shard in shards
        ],
    }
    (artifact_dir / "shard-plan.json").write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def run_smoke(config: RuntimeConfig, *, shards: int = 3) -> int:
    """Exercise all real model and service tests, omitting only pure unit tests."""

    return run_tests(
        config,
        shard_count=shards,
        include_lint=False,
        selection_args=("-m", "model or services or e2e"),
        enforce_coverage=False,
        learn=False,
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
        mount_models=with_services,
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
        _terminate_sessions([session])


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
        _terminate_sessions([session])


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
        session.capture(
            "bash",
            "-c",
            "git init -q && git config user.email modal-native-test-stack-poc@example.invalid && "
            "git config user.name 'Modal-Native Test Stack POC' && "
            "git add -A && git commit -qm baseline",
            label="ephemeral agent Git baseline",
        )
        guidance = (
            "You are in an ephemeral Modal Sandbox at /workspace. PostgreSQL, Redis, and "
            "OpenSearch are ready at their service DNS names. Real Hugging Face snapshots "
            "are mounted read-only at /models and network model downloads are disabled. "
            "Do not use Docker or Compose. Run tests directly with pytest."
        )
        assert session.sandbox is not None
        session.sandbox.filesystem.write_text(guidance + "\n", "/workspace/AGENTS.override.md")
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
        _terminate_sessions([session])


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
