"""Ephemeral PostgreSQL, Redis, and OpenSearch Sandbox Sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import modal

from modal_native_test_stack_poc.remote.config import REMOTE_ARTIFACTS
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError
from modal_native_test_stack_poc.remote.processes import capture_process


@dataclass(frozen=True, slots=True)
class ServiceDefinition:
    key: str
    name: str
    display_name: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    workdir: str | None = None


POSTGRES = ServiceDefinition(
    key="postgres",
    name="postgres",
    display_name="PostgreSQL",
    command=("/usr/local/bin/docker-entrypoint.sh", "postgres"),
)

REDIS = ServiceDefinition(
    key="redis",
    name="redis",
    display_name="Redis",
    command=(
        "/usr/local/bin/docker-entrypoint.sh",
        "redis-server",
        "--save",
        "",
        "--appendonly",
        "no",
    ),
)

OPENSEARCH = ServiceDefinition(
    key="opensearch",
    name="opensearch",
    display_name="OpenSearch",
    command=(
        "/usr/bin/setpriv",
        "--reuid=1000",
        "--regid=1000",
        "--init-groups",
        "/usr/share/opensearch/opensearch-docker-entrypoint.sh",
        "opensearch",
    ),
    environment=(
        ("discovery.type", "single-node"),
        ("DISABLE_INSTALL_DEMO_CONFIG", "true"),
        ("DISABLE_SECURITY_PLUGIN", "true"),
        ("OPENSEARCH_JAVA_OPTS", "-Xms512m -Xmx512m"),
    ),
    workdir="/usr/share/opensearch",
)

SERVICES = (POSTGRES, REDIS, OPENSEARCH)

READINESS_SCRIPT = r"""from __future__ import annotations

import asyncio
import os
import sys
import time

import asyncpg
import redis
from opensearchpy import OpenSearch


async def postgres_ready() -> None:
    connection = await asyncpg.connect(
        host="postgres",
        user="postgres",
        password=os.environ["POSTGRES_PASSWORD"],
        database="modal_native_test_stack_poc",
        timeout=3,
    )
    try:
        value = await connection.fetchval("SELECT 40 + 2")
        if value != 42:
            raise RuntimeError(f"unexpected PostgreSQL result: {value!r}")
    finally:
        await connection.close()


def redis_ready() -> None:
    client = redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=3)
    key = "modal-native-test-stack-poc:readiness"
    client.set(key, b"ready", ex=30)
    if client.get(key) != b"ready":
        raise RuntimeError("Redis write/read contract failed")
    client.delete(key)


def opensearch_ready() -> None:
    client = OpenSearch(
        hosts=[{"host": "opensearch", "port": 9200}],
        use_ssl=False,
        verify_certs=False,
        timeout=3,
    )
    if not client.ping():
        raise RuntimeError("OpenSearch ping failed")
    index = ".modal-native-test-stack-poc-readiness"
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    client.indices.create(
        index=index,
        body={"mappings": {"properties": {"value": {"type": "keyword"}}}},
    )
    client.index(index=index, id="ready", body={"value": "ready"}, refresh=True)
    document = client.get(index=index, id="ready")
    if document["_source"]["value"] != "ready":
        raise RuntimeError("OpenSearch index/get contract failed")
    client.indices.delete(index=index)


deadline = time.monotonic() + float(os.environ.get("SERVICE_READY_TIMEOUT", "300"))
attempt = 0
last_error: BaseException | None = None
while time.monotonic() < deadline:
    attempt += 1
    try:
        asyncio.run(postgres_ready())
        redis_ready()
        opensearch_ready()
    except BaseException as error:
        last_error = error
        time.sleep(min(0.25 * (2 ** min(attempt, 4)), 3.0))
    else:
        print(f"all services passed semantic readiness after {attempt} attempt(s)")
        raise SystemExit(0)

print(f"service readiness timed out: {type(last_error).__name__}: {last_error}", file=sys.stderr)
raise SystemExit(1)
"""


def attach_services(
    sandbox: modal.Sandbox,
    images: dict[str, modal.Image],
    *,
    postgres_password: str,
) -> list[Any]:
    """Attach one service at a time; concurrent calls can stall the alpha API."""

    sidecars: list[Any] = []
    for service in SERVICES:
        print(f"Attaching {service.display_name} to {sandbox.object_id}...")
        environment = dict(service.environment)
        if service.key == "postgres":
            environment.update(
                {
                    "POSTGRES_DB": "modal_native_test_stack_poc",
                    "POSTGRES_PASSWORD": postgres_password,
                    "POSTGRES_USER": "postgres",
                }
            )
        sidecars.append(
            sandbox._experimental_sidecars.create(
                *service.command,
                name=service.name,
                image=images[service.key],
                env=environment,
                workdir=service.workdir,
                outbound_cidr_allowlist=[],
            )
        )
    return sidecars


def wait_for_services(sandbox: modal.Sandbox, *, timeout_seconds: int) -> None:
    """Probe real read/write behavior rather than merely checking open ports."""

    script_path = f"{REMOTE_ARTIFACTS}/service_readiness.py"
    sandbox.filesystem.write_text(READINESS_SCRIPT, script_path)
    try:
        capture_process(
            sandbox.exec(
                "python",
                script_path,
                workdir="/workspace",
                timeout=timeout_seconds + 30,
                env={"SERVICE_READY_TIMEOUT": str(timeout_seconds)},
            ),
            label="service readiness",
        )
    except Exception as error:
        states: dict[str, object] = {}
        for service in SERVICES:
            try:
                sidecar = sandbox._experimental_sidecars.get(
                    name=service.name,
                    include_terminated=True,
                )
                states[service.name] = sidecar.poll()
            except Exception as state_error:
                states[service.name] = type(state_error).__name__
        raise ModalNativeTestStackError(
            f"service readiness failed; Sidecar states: {states}"
        ) from error
