"""Fresh, tagged Modal Sandbox sessions."""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import modal

from modal_native_test_stack_poc.remote.config import (
    MODEL_MOUNT,
    OWNER_TAG_KEY,
    OWNER_TAG_VALUE,
    REMOTE_ARTIFACTS,
    RuntimeConfig,
)
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError
from modal_native_test_stack_poc.remote.models import model_volume
from modal_native_test_stack_poc.remote.processes import (
    ProcessResult,
    capture_process,
    read_process,
    stream_process,
)
from modal_native_test_stack_poc.remote.services import attach_services, wait_for_services


@dataclass(slots=True)
class SandboxSession:
    app: modal.App
    config: RuntimeConfig
    image: modal.Image
    service_images: dict[str, modal.Image]
    role: str
    run_id: str
    with_services: bool = True
    allow_network: bool = False
    secret_names: tuple[str, ...] = ()
    encrypted_ports: tuple[int, ...] = ()
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    postgres_password: str = field(
        default_factory=lambda: secrets.token_urlsafe(24),
        repr=False,
    )
    sandbox: modal.Sandbox | None = None
    sidecars: list[Any] = field(default_factory=list)

    def environment(self) -> dict[str, str]:
        password = quote(self.postgres_password, safe="")
        postgres_url = f"postgresql://postgres:{password}@postgres:5432/modal_native_test_stack_poc"
        return {
            "DATABASE_URL": postgres_url,
            "TEST_DATABASE_URL": postgres_url,
            "POSTGRES_HOST": "postgres",
            "POSTGRES_DB": "modal_native_test_stack_poc",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": self.postgres_password,
            "REDIS_URL": "redis://redis:6379/0",
            "OPENSEARCH_URL": "http://opensearch:9200",
            "MODELS_ROOT": MODEL_MOUNT,
            "MODEL_ROOT": MODEL_MOUNT,
            "MULTIMODAL_MODELS_ROOT": MODEL_MOUNT,
            "MULTIMODAL_MODELS_LOCK_PATH": "/workspace/models.lock.json",
            "MULTIMODAL_POSTGRES_URL": postgres_url,
            "MULTIMODAL_REDIS_URL": "redis://redis:6379/0",
            "MULTIMODAL_OPENSEARCH_URL": "http://opensearch:9200",
            "HF_HOME": f"{MODEL_MOUNT}/.cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MODAL_NATIVE_TEST_STACK_POC_REMOTE": "1",
            "MODAL_NATIVE_TEST_STACK_POC_RUN_ID": self.run_id,
        }

    def create(self) -> None:
        if self.sandbox is not None:
            raise ModalNativeTestStackError("Sandbox session was already created")
        volume_mounts: dict[str, Any] = {}
        if self.with_services:
            volume_mounts[MODEL_MOUNT] = model_volume(
                self.config,
                create=False,
            ).with_mount_options(read_only=True)
        secrets_handles = tuple(
            modal.Secret.from_name(
                name,
                environment_name=self.config.environment_name,
            )
            for name in self.secret_names
        )
        name = f"modal-native-test-stack-poc-{self.role}-{self.run_id[:8]}"
        arguments: dict[str, Any] = {
            "app": self.app,
            "name": name,
            "tags": {
                OWNER_TAG_KEY: OWNER_TAG_VALUE,
                "modal-native-test-stack-poc-role": self.role,
                "modal-native-test-stack-poc-run": self.run_id,
                "modal-native-test-stack-poc-session": self.session_id,
            },
            "image": self.image,
            "cpu": self.config.cpu,
            "memory": self.config.memory_mb,
            "timeout": self.config.timeout_seconds,
            "idle_timeout": self.config.idle_timeout_seconds,
            "workdir": "/workspace",
            "env": self.environment(),
            "secrets": secrets_handles,
            "volumes": volume_mounts,
            "encrypted_ports": self.encrypted_ports,
            "readiness_probe": modal.Probe.with_exec(
                "sh",
                "-c",
                "test -f /tmp/modal-native-test-stack-poc-main-ready",
                interval_ms=500,
            ),
        }
        if not self.allow_network:
            arguments["outbound_cidr_allowlist"] = []
        self.sandbox = modal.Sandbox.create(
            "bash",
            "-lc",
            f"mkdir -p {REMOTE_ARTIFACTS} && "
            "touch /tmp/modal-native-test-stack-poc-main-ready && exec sleep infinity",
            **arguments,
        )
        self.sandbox.wait_until_ready(timeout=300)
        schema = capture_process(
            self.sandbox.exec("cat", "/opt/modal-native-test-stack-poc-image-schema", timeout=30),
            label="runtime Image schema check",
        ).strip()
        if schema != "modal-native-test-stack-poc-runtime-v1":
            raise ModalNativeTestStackError(f"unexpected runtime Image schema: {schema!r}")

    def attach_services(self) -> None:
        if not self.with_services:
            return
        if self.sandbox is None:
            raise ModalNativeTestStackError("create the main Sandbox before attaching Sidecars")
        self.sidecars.extend(
            attach_services(
                self.sandbox,
                self.service_images,
                postgres_password=self.postgres_password,
            )
        )

    def wait_until_usable(self) -> None:
        if not self.with_services:
            return
        if self.sandbox is None:
            raise ModalNativeTestStackError("create the main Sandbox before waiting for Sidecars")
        wait_for_services(
            self.sandbox,
            timeout_seconds=self.config.service_timeout_seconds,
        )

    def capture(self, *command: str, label: str, timeout: int | None = None) -> str:
        if self.sandbox is None:
            raise ModalNativeTestStackError("Sandbox is not running")
        return capture_process(
            self.sandbox.exec(
                *command,
                workdir="/workspace",
                timeout=timeout or self.config.timeout_seconds,
            ),
            label=label,
        )

    def run(
        self,
        *command: str,
        prefix: str = "",
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
        stdout_line_transform: Callable[[str], str | None] | None = None,
    ) -> ProcessResult:
        if self.sandbox is None:
            raise ModalNativeTestStackError("Sandbox is not running")
        process = self.sandbox.exec(
            *command,
            workdir="/workspace",
            timeout=timeout or self.config.timeout_seconds,
            env=environment,
        )
        return stream_process(
            process,
            prefix=prefix,
            stdout_line_transform=stdout_line_transform,
        )

    def run_captured(
        self,
        *command: str,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
    ) -> ProcessResult:
        if self.sandbox is None:
            raise ModalNativeTestStackError("Sandbox is not running")
        process = self.sandbox.exec(
            *command,
            workdir="/workspace",
            timeout=timeout or self.config.timeout_seconds,
            env=environment,
        )
        return read_process(process)

    def open_terminal(self) -> int:
        if self.sandbox is None:
            raise ModalNativeTestStackError("Sandbox is not running")
        process = self.sandbox.exec(
            "bash",
            workdir="/workspace",
            timeout=self.config.timeout_seconds,
            pty=True,
        )
        process.attach()
        return process.wait()

    def terminate(self) -> list[str]:
        warnings: list[str] = []
        for sidecar in reversed(self.sidecars):
            try:
                sidecar.terminate(wait=False)
            except Exception as error:
                warnings.append(f"Sidecar: {type(error).__name__}")
        self.sidecars.clear()
        if self.sandbox is not None:
            sandbox, self.sandbox = self.sandbox, None
            try:
                sandbox.terminate(wait=False)
            except Exception as error:
                warnings.append(f"Sandbox {sandbox.object_id}: {type(error).__name__}")
            finally:
                try:
                    sandbox.detach()
                except Exception as error:
                    warnings.append(f"detach {sandbox.object_id}: {type(error).__name__}")
        return warnings

    def __enter__(self) -> SandboxSession:
        self.create()
        self.attach_services()
        self.wait_until_usable()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.terminate()
