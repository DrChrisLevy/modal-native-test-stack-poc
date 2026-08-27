"""Modal Image definitions with dependencies isolated from changing source."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

from modal_native_test_stack_poc.remote.config import (
    REMOTE_ARTIFACTS,
    RuntimeConfig,
    source_is_ignored,
)
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError

PYTHON_VERSION = "3.12"
UV_VERSION = "0.8.15"
IMAGE_SCHEMA = "modal-native-test-stack-poc-runtime-v1"
CODEX_RELEASE_TAG = "rust-v0.149.1"

POSTGRES_IMAGE = "public.ecr.aws/docker/library/postgres:17.5-bookworm"
REDIS_IMAGE = "public.ecr.aws/docker/library/redis:8.0.3-bookworm"
OPENSEARCH_IMAGE = "public.ecr.aws/opensearchproject/opensearch:2.19.1"


@dataclass(frozen=True, slots=True)
class BuiltImages:
    runtime: modal.Image
    services: dict[str, modal.Image]
    agent: modal.Image | None = None


def ensure_supported_modal() -> None:
    """Fail clearly instead of producing opaque errors from alpha Sidecar APIs."""

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", getattr(modal, "__version__", ""))
    if match is None:
        raise ModalNativeTestStackError("could not determine the installed Modal SDK version")
    version = tuple(int(value) for value in match.groups())
    if not ((1, 5, 4) <= version < (1, 6, 0)):
        raise ModalNativeTestStackError(
            "modal-native-test-stack-poc requires Modal >=1.5.4,<1.6 because it uses the alpha "
            f"Sandbox Sidecar API; found {modal.__version__}"
        )


def dependency_image_definition(root: Path, *, force: bool = False) -> modal.Image:
    """Build the expensive locked layer without installing local project source."""

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION, force_build=force)
        .apt_install(
            "ca-certificates",
            "curl",
            "ffmpeg",
            "git",
            "libgl1",
            "libglib2.0-0",
            "libgomp1",
            "postgresql-client",
            "redis-tools",
            "ripgrep",
            force_build=force,
        )
        .env(
            {
                "DEBIAN_FRONTEND": "noninteractive",
                "PYTHONPATH": "/workspace/src",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .uv_sync(
            str(root),
            extras=["remote", "test"],
            frozen=True,
            force_build=force,
            uv_version=UV_VERSION,
        )
        .run_commands(
            f"mkdir -p /workspace {REMOTE_ARTIFACTS} /models",
            f"printf '%s\\n' {IMAGE_SCHEMA} >/opt/modal-native-test-stack-poc-image-schema",
            force_build=force,
        )
        .workdir("/workspace")
    )


def runtime_image_definition(root: Path, *, force: bool = False) -> modal.Image:
    """Add the current checkout only after the expensive dependency layer."""

    return dependency_image_definition(root, force=force).add_local_dir(
        root,
        "/workspace",
        copy=True,
        ignore=source_is_ignored,
    )


def _codex_install_command() -> str:
    # The release API provides the SHA-256 digest for the selected Linux asset.
    # The Image layer is cached; --force-build is the explicit refresh mechanism.
    return (
        "set -eu; "
        "release=/tmp/codex-release.json; archive=/tmp/codex.tar.gz; "
        "tag=" + CODEX_RELEASE_TAG + "; "
        'curl -fsSL "https://api.github.com/repos/openai/codex/releases/tags/$tag" '
        "-o $release; "
        "asset=codex-x86_64-unknown-linux-musl.tar.gz; "
        'digest=$(jq -r --arg asset "$asset" '
        "'.assets[] | select(.name == $asset) | .digest // empty' $release | sed 's/^sha256://'); "
        'test -n "$tag"; test -n "$digest"; '
        'curl -fsSL "https://github.com/openai/codex/releases/download/$tag/$asset" -o $archive; '
        "printf '%s  %s\\n' \"$digest\" $archive | sha256sum -c -; "
        "tar -xzf $archive -C /usr/local/bin; "
        "mv /usr/local/bin/codex-x86_64-unknown-linux-musl /usr/local/bin/codex; "
        "chmod 755 /usr/local/bin/codex; codex --version"
    )


def agent_image_definition(root: Path, *, force: bool = False) -> modal.Image:
    """Optional developer layer; normal test and shell builds do not pay for it."""

    return (
        dependency_image_definition(root, force=force)
        .apt_install("jq", force_build=force)
        .run_commands(_codex_install_command(), force_build=force)
        .add_local_dir(
            root,
            "/workspace",
            copy=True,
            ignore=source_is_ignored,
        )
    )


def service_image_definitions(*, force: bool = False) -> dict[str, modal.Image]:
    """Return immutable definitions that are explicitly built before attachment."""

    postgres = modal.Image.from_registry(POSTGRES_IMAGE, force_build=force)
    redis = modal.Image.from_registry(REDIS_IMAGE, force_build=force)
    opensearch = modal.Image.from_registry(OPENSEARCH_IMAGE, force_build=force).run_commands(
        "dnf install -y util-linux && dnf clean all",
        force_build=force,
    )
    return {"postgres": postgres, "redis": redis, "opensearch": opensearch}


def build_images(
    app: modal.App,
    config: RuntimeConfig,
    *,
    force: bool = False,
    include_agent: bool = False,
) -> BuiltImages:
    """Resolve all lazy Images before any Sidecar calls are made."""

    ensure_supported_modal()
    config.validate()
    built_services: dict[str, modal.Image] = {}
    with modal.enable_output():
        for name, definition in service_image_definitions(force=force).items():
            print(f"Resolving {name} Sidecar Image...")
            built_services[name] = definition.build(app)
        print("Resolving locked Python/ML runtime and current source...")
        runtime = runtime_image_definition(config.root, force=force).build(app)
        agent: Any = None
        if include_agent:
            print("Resolving optional Codex developer Image...")
            agent = agent_image_definition(config.root, force=force).build(app)
    return BuiltImages(runtime=runtime, services=built_services, agent=agent)
