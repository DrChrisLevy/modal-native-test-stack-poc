"""Configuration for the Modal execution harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError

APP_NAME = "modal-native-test-stack-poc"
MODEL_VOLUME_NAME = "modal-native-test-stack-poc-models"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"
OWNER_TAG_KEY = "modal-native-test-stack-poc-owner"
OWNER_TAG_VALUE = "modal-native-test-stack-poc"

WORKSPACE = "/workspace"
MODEL_MOUNT = "/models"
REMOTE_ARTIFACTS = "/workspace/.modal-native-test-stack-poc-artifacts"


def project_root() -> Path:
    """Return the checkout containing ``pyproject.toml`` and ``models.lock.json``."""

    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file():
        raise ModalNativeTestStackError(
            f"could not find pyproject.toml at expected project root: {root}"
        )
    return root


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ModalNativeTestStackError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ModalNativeTestStackError(f"{name} must be positive, got {value}")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ModalNativeTestStackError(f"{name} must be numeric, got {raw!r}") from exc
    if value <= 0:
        raise ModalNativeTestStackError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime settings, all overrideable without changing public source."""

    root: Path
    app_name: str
    model_volume_name: str
    environment_name: str | None
    cpu: float
    memory_mb: int
    timeout_seconds: int
    idle_timeout_seconds: int
    service_timeout_seconds: int

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        # Must be selected before the lazy Modal imports in ``cli.py``. The
        # modern builder can prebuild non-Python registry Images for Sidecars;
        # the legacy builder tries to install Python into PostgreSQL/Redis.
        os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", MODAL_IMAGE_BUILDER_VERSION)
        root = project_root()
        return cls(
            root=root,
            app_name=os.getenv("MODAL_NATIVE_TEST_STACK_POC_APP", APP_NAME),
            model_volume_name=os.getenv(
                "MODAL_NATIVE_TEST_STACK_POC_MODEL_VOLUME", MODEL_VOLUME_NAME
            ),
            environment_name=os.getenv("MODAL_ENVIRONMENT") or None,
            cpu=_positive_float("MODAL_NATIVE_TEST_STACK_POC_CPU", 4.0),
            memory_mb=_positive_int("MODAL_NATIVE_TEST_STACK_POC_MEMORY_MB", 16_384),
            timeout_seconds=_positive_int("MODAL_NATIVE_TEST_STACK_POC_TIMEOUT_SECONDS", 3_600),
            idle_timeout_seconds=_positive_int(
                "MODAL_NATIVE_TEST_STACK_POC_IDLE_TIMEOUT_SECONDS", 600
            ),
            service_timeout_seconds=_positive_int(
                "MODAL_NATIVE_TEST_STACK_POC_SERVICE_TIMEOUT_SECONDS", 300
            ),
        )

    @property
    def artifacts_root(self) -> Path:
        return self.root / "artifacts" / "modal"

    def validate(self) -> None:
        required = (
            self.root / "pyproject.toml",
            self.root / "uv.lock",
            self.root / "models.lock.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ModalNativeTestStackError(
                "required project files are missing: " + ", ".join(missing)
            )


def source_is_ignored(path: Path) -> bool:
    """Keep local caches, credentials, generated output, and VCS data out of Images."""

    ignored_names = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "artifacts",
        "htmlcov",
        "node_modules",
    }
    return bool(set(path.parts).intersection(ignored_names)) or path.name in {
        ".env",
        ".env.local",
        ".netrc",
    }
