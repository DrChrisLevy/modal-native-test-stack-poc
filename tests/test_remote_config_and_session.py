from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from modal_native_test_stack_poc.remote.config import RuntimeConfig, project_root, source_is_ignored
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError
from modal_native_test_stack_poc.remote.processes import ProcessResult
from modal_native_test_stack_poc.remote.services import OPENSEARCH, POSTGRES, REDIS, SERVICES
from modal_native_test_stack_poc.remote.session import SandboxSession


@pytest.fixture
def runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        root=tmp_path,
        app_name="modal-native-test-stack-poc-test",
        model_volume_name="modal-native-test-stack-poc-test-models",
        environment_name="main",
        cpu=4.0,
        memory_mb=16_384,
        timeout_seconds=3_600,
        idle_timeout_seconds=600,
        service_timeout_seconds=300,
    )


@pytest.fixture
def clean_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (
        "MODAL_NATIVE_TEST_STACK_POC_APP",
        "MODAL_NATIVE_TEST_STACK_POC_MODEL_VOLUME",
        "MODAL_ENVIRONMENT",
        "MODAL_NATIVE_TEST_STACK_POC_CPU",
        "MODAL_NATIVE_TEST_STACK_POC_MEMORY_MB",
        "MODAL_NATIVE_TEST_STACK_POC_TIMEOUT_SECONDS",
        "MODAL_NATIVE_TEST_STACK_POC_IDLE_TIMEOUT_SECONDS",
        "MODAL_NATIVE_TEST_STACK_POC_SERVICE_TIMEOUT_SECONDS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_project_root_contains_public_lockfiles() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "models.lock.json").is_file()


def test_runtime_config_has_documented_defaults(clean_runtime_environment: None) -> None:
    config = RuntimeConfig.from_environment()
    assert config.app_name == "modal-native-test-stack-poc"
    assert config.model_volume_name == "modal-native-test-stack-poc-models"
    assert config.cpu == 4.0
    assert config.memory_mb == 16_384
    assert config.timeout_seconds == 3_600


@pytest.mark.parametrize(
    ("variable", "attribute", "raw", "expected"),
    [
        ("MODAL_NATIVE_TEST_STACK_POC_APP", "app_name", "custom-app", "custom-app"),
        ("MODAL_NATIVE_TEST_STACK_POC_MODEL_VOLUME", "model_volume_name", "weights", "weights"),
        ("MODAL_ENVIRONMENT", "environment_name", "dev", "dev"),
        ("MODAL_NATIVE_TEST_STACK_POC_CPU", "cpu", "7.5", 7.5),
        ("MODAL_NATIVE_TEST_STACK_POC_MEMORY_MB", "memory_mb", "2048", 2048),
        ("MODAL_NATIVE_TEST_STACK_POC_TIMEOUT_SECONDS", "timeout_seconds", "99", 99),
        ("MODAL_NATIVE_TEST_STACK_POC_IDLE_TIMEOUT_SECONDS", "idle_timeout_seconds", "45", 45),
        (
            "MODAL_NATIVE_TEST_STACK_POC_SERVICE_TIMEOUT_SECONDS",
            "service_timeout_seconds",
            "30",
            30,
        ),
    ],
)
def test_runtime_config_reads_environment(
    clean_runtime_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    attribute: str,
    raw: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, raw)
    assert getattr(RuntimeConfig.from_environment(), attribute) == expected


@pytest.mark.parametrize(
    ("variable", "raw"),
    [
        ("MODAL_NATIVE_TEST_STACK_POC_CPU", "zero"),
        ("MODAL_NATIVE_TEST_STACK_POC_MEMORY_MB", "1.5"),
        ("MODAL_NATIVE_TEST_STACK_POC_TIMEOUT_SECONDS", "infinite"),
    ],
)
def test_runtime_config_rejects_non_numeric_values(
    clean_runtime_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    raw: str,
) -> None:
    monkeypatch.setenv(variable, raw)
    with pytest.raises(ModalNativeTestStackError, match="must be"):
        RuntimeConfig.from_environment()


@pytest.mark.parametrize(
    ("variable", "raw"),
    [
        ("MODAL_NATIVE_TEST_STACK_POC_CPU", "0"),
        ("MODAL_NATIVE_TEST_STACK_POC_MEMORY_MB", "0"),
        ("MODAL_NATIVE_TEST_STACK_POC_TIMEOUT_SECONDS", "-1"),
    ],
)
def test_runtime_config_rejects_nonpositive_values(
    clean_runtime_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    raw: str,
) -> None:
    monkeypatch.setenv(variable, raw)
    with pytest.raises(ModalNativeTestStackError, match="must be positive"):
        RuntimeConfig.from_environment()


def test_runtime_config_validates_required_project_files(
    runtime_config: RuntimeConfig,
) -> None:
    for name in ("pyproject.toml", "uv.lock", "models.lock.json"):
        (runtime_config.root / name).touch()
    runtime_config.validate()


def test_runtime_config_names_missing_project_files(runtime_config: RuntimeConfig) -> None:
    with pytest.raises(ModalNativeTestStackError, match=r"pyproject\.toml"):
        runtime_config.validate()


def test_artifacts_root_lives_under_checkout(runtime_config: RuntimeConfig) -> None:
    assert runtime_config.artifacts_root == runtime_config.root / "artifacts" / "modal"


@pytest.mark.parametrize(
    "path",
    [
        Path(".git/config"),
        Path(".venv/bin/python"),
        Path("src/__pycache__/module.pyc"),
        Path(".pytest_cache/state"),
        Path("artifacts/result.xml"),
        Path("htmlcov/index.html"),
        Path("frontend/node_modules/pkg"),
        Path(".env"),
        Path(".env.local"),
        Path(".netrc"),
    ],
)
def test_sensitive_or_generated_source_is_ignored(path: Path) -> None:
    assert source_is_ignored(path) is True


@pytest.mark.parametrize(
    "path",
    [Path("src/modal_native_test_stack_poc/app.py"), Path("tests/test_api.py"), Path("uv.lock")],
)
def test_project_source_is_included(path: Path) -> None:
    assert source_is_ignored(path) is False


def test_all_three_sidecars_are_declared() -> None:
    assert {service.key for service in SERVICES} == {"postgres", "redis", "opensearch"}


def test_sidecar_names_match_private_dns_names() -> None:
    assert POSTGRES.name == "postgres"
    assert REDIS.name == "redis"
    assert OPENSEARCH.name == "opensearch"


def test_opensearch_sidecar_disables_demo_security() -> None:
    environment = dict(OPENSEARCH.environment)
    assert environment["DISABLE_SECURITY_PLUGIN"] == "true"
    assert environment["discovery.type"] == "single-node"


def test_process_result_combines_stdout_and_stderr() -> None:
    result = ProcessResult(("pytest",), 1, "out", "error")
    assert result.output == "out\nerror"


def test_process_result_omits_empty_stream_separator() -> None:
    assert ProcessResult(("true",), 0, "done", "").output == "done"


def test_sandbox_environment_is_offline_and_volume_backed(runtime_config: RuntimeConfig) -> None:
    session = SandboxSession(
        app=None,  # type: ignore[arg-type]
        config=runtime_config,
        image=None,  # type: ignore[arg-type]
        service_images={},
        role="test",
        run_id="run-123",
    )
    environment = session.environment()
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["MULTIMODAL_MODELS_ROOT"] == "/models"


def test_sandbox_environment_points_at_sidecar_dns(runtime_config: RuntimeConfig) -> None:
    session = SandboxSession(
        app=None,  # type: ignore[arg-type]
        config=runtime_config,
        image=None,  # type: ignore[arg-type]
        service_images={},
        role="test",
        run_id="run-123",
    )
    environment = session.environment()
    assert "@postgres:5432/" in environment["MULTIMODAL_POSTGRES_URL"]
    assert environment["MULTIMODAL_REDIS_URL"] == "redis://redis:6379/0"
    assert environment["MULTIMODAL_OPENSEARCH_URL"] == "http://opensearch:9200"


def test_sandbox_environment_url_encodes_postgres_password(
    runtime_config: RuntimeConfig,
) -> None:
    session = SandboxSession(
        app=None,  # type: ignore[arg-type]
        config=runtime_config,
        image=None,  # type: ignore[arg-type]
        service_images={},
        role="test",
        run_id="run-123",
        postgres_password="slash/ space",
    )
    assert "slash%2F%20space" in session.environment()["MULTIMODAL_POSTGRES_URL"]


def test_session_operations_require_a_running_sandbox(runtime_config: RuntimeConfig) -> None:
    session = SandboxSession(
        app=None,  # type: ignore[arg-type]
        config=runtime_config,
        image=None,  # type: ignore[arg-type]
        service_images={},
        role="test",
        run_id="run-123",
    )
    with pytest.raises(ModalNativeTestStackError, match="before attaching"):
        session.attach_services()
    with pytest.raises(ModalNativeTestStackError, match="before waiting"):
        session.wait_until_usable()
    with pytest.raises(ModalNativeTestStackError, match="not running"):
        session.capture("true", label="test")
    with pytest.raises(ModalNativeTestStackError, match="not running"):
        session.run("true")
    with pytest.raises(ModalNativeTestStackError, match="not running"):
        session.run_captured("true")


def test_terminal_preserves_the_image_environment(runtime_config: RuntimeConfig) -> None:
    sandbox = Mock()
    sandbox.exec.return_value.wait.return_value = 0
    session = SandboxSession(
        app=None,  # type: ignore[arg-type]
        config=runtime_config,
        image=None,  # type: ignore[arg-type]
        service_images={},
        role="shell",
        run_id="run-123",
        sandbox=sandbox,
    )

    assert session.open_terminal() == 0
    sandbox.exec.assert_called_once_with(
        "bash",
        workdir="/workspace",
        timeout=runtime_config.timeout_seconds,
        pty=True,
    )
    sandbox.exec.return_value.attach.assert_called_once_with()


def test_service_free_session_skips_sidecar_lifecycle(runtime_config: RuntimeConfig) -> None:
    session = SandboxSession(
        app=None,  # type: ignore[arg-type]
        config=runtime_config,
        image=None,  # type: ignore[arg-type]
        service_images={},
        role="shell",
        run_id="run-123",
        with_services=False,
    )
    session.attach_services()
    session.wait_until_usable()
