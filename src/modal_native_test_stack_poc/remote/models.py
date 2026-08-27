"""Persistent model Volume lifecycle."""

from __future__ import annotations

import json
import secrets
from typing import Any

import modal

from modal_native_test_stack_poc.remote.config import (
    MODEL_MOUNT,
    OWNER_TAG_KEY,
    OWNER_TAG_VALUE,
    RuntimeConfig,
)
from modal_native_test_stack_poc.remote.errors import ModalNativeTestStackError
from modal_native_test_stack_poc.remote.processes import stream_process


def model_volume(config: RuntimeConfig, *, create: bool) -> modal.Volume:
    return modal.Volume.from_name(
        config.model_volume_name,
        environment_name=config.environment_name,
        create_if_missing=create,
    )


def seed_model_volume(
    app: modal.App,
    config: RuntimeConfig,
    image: modal.Image,
    *,
    force: bool = False,
) -> int:
    """Download lockfile snapshots once, then commit them to durable storage."""

    volume = model_volume(config, create=True)
    run_id = secrets.token_hex(6)
    sandbox: modal.Sandbox | None = None
    try:
        sandbox = modal.Sandbox.create(
            "bash",
            "-lc",
            "touch /tmp/modal-native-test-stack-poc-ready && exec sleep infinity",
            app=app,
            name=f"modal-native-test-stack-poc-seed-{run_id}",
            tags={
                OWNER_TAG_KEY: OWNER_TAG_VALUE,
                "modal-native-test-stack-poc-role": "seed",
                "modal-native-test-stack-poc-run": run_id,
            },
            image=image,
            cpu=max(config.cpu, 4.0),
            memory=max(config.memory_mb, 16_384),
            timeout=min(24 * 60 * 60, max(config.timeout_seconds, 6 * 60 * 60)),
            idle_timeout=config.idle_timeout_seconds,
            workdir="/workspace",
            volumes={MODEL_MOUNT: volume},
            env={
                "HF_HOME": f"{MODEL_MOUNT}/.cache",
                "HF_HUB_OFFLINE": "0",
                "TRANSFORMERS_OFFLINE": "0",
            },
            readiness_probe=modal.Probe.with_exec(
                "sh", "-c", "test -f /tmp/modal-native-test-stack-poc-ready", interval_ms=500
            ),
        )
        sandbox.wait_until_ready(timeout=300)
        command = [
            "python",
            "-m",
            "modal_native_test_stack_poc.remote.seed_worker",
            "--lockfile",
            "/workspace/models.lock.json",
            "--models-root",
            MODEL_MOUNT,
        ]
        if force:
            command.append("--force")
        result = stream_process(
            sandbox.exec(*command, workdir="/workspace", timeout=12 * 60 * 60),
            prefix="[models] ",
        )
        if result.returncode != 0:
            raise ModalNativeTestStackError(
                f"model seeding failed with exit code {result.returncode}"
            )
        # Sandboxes receive no reusable Modal credentials, so commit() cannot
        # hydrate a new SDK handle from an exec child. Modal durably commits a
        # mounted Volume when the Sandbox exits; wait for that boundary before
        # verifying from the authenticated local client.
        sandbox.terminate(wait=True)
        sandbox.detach()
        sandbox = None
        committed = b"".join(volume.read_file("models.state.json"))
        committed_state = json.loads(committed)
        if not isinstance(committed_state.get("models"), list):
            raise ModalNativeTestStackError(
                "committed model Volume failed its manifest verification"
            )
        print(f"Committed model snapshots to Volume {config.model_volume_name!r}.")
        return 0
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate(wait=True)
            finally:
                sandbox.detach()


def read_model_state(config: RuntimeConfig) -> dict[str, Any]:
    volume = model_volume(config, create=False)
    try:
        payload = b"".join(volume.read_file("models.state.json"))
    except Exception as error:
        raise ModalNativeTestStackError(
            f"model Volume {config.model_volume_name!r} is missing or unseeded; run models-seed"
        ) from error
    try:
        state = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ModalNativeTestStackError(
            "models.state.json in the model Volume is invalid"
        ) from error
    if not isinstance(state, dict):
        raise ModalNativeTestStackError("models.state.json in the model Volume is not an object")
    return state


def check_model_volume(config: RuntimeConfig) -> int:
    state = read_model_state(config)
    models = state.get("models", [])
    print(f"Volume: {config.model_volume_name}")
    print(f"Seeded models: {len(models) if isinstance(models, list) else 0}")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            revision = str(model.get("resolved_revision", "unknown"))
            print(f"  {model.get('key', '?'):22} {model.get('repo_id', '?')}@{revision[:12]}")
    return 0


def delete_model_volume(config: RuntimeConfig) -> None:
    modal.Volume.objects.delete(
        config.model_volume_name,
        environment_name=config.environment_name,
        allow_missing=True,
    )
    print(f"Deleted model Volume {config.model_volume_name!r}.")
