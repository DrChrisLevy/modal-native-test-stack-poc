"""Executed inside a writable model-Volume Sandbox."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from modal_native_test_stack_poc.inference.lockfile import ModelManifest

MARKER_NAME = ".modal-native-test-stack-poc-model.json"
STATE_NAME = "models.state.json"
ARTIFACT_POLICY_VERSION = 1

COMMON_SNAPSHOT_IGNORE_PATTERNS = (
    ".gitattributes",
    "README.md",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "onnx/*",
    "openvino/*",
    "map.jpeg",
    "train_script.py",
)


def _snapshot_ignore_patterns(model_key: str) -> list[str]:
    patterns = list(COMMON_SNAPSHOT_IGNORE_PATTERNS)
    # The pinned OpenAI CLIP snapshot has no safetensors weights.
    if model_key != "image_embedding":
        patterns.append("pytorch_model.bin")
    return patterns


def _read_marker(path: Path) -> dict[str, object] | None:
    marker = path / MARKER_NAME
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def seed(lockfile: Path, models_root: Path, *, force: bool) -> dict[str, object]:
    manifest = ModelManifest.load(lockfile)
    models_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    seeded: list[dict[str, object]] = []

    for spec in manifest.models:
        print(f"Resolving {spec.repo_id}@{spec.revision}...", flush=True)
        resolved_revision = api.model_info(spec.repo_id, revision=spec.revision).sha
        if not resolved_revision:
            raise RuntimeError(f"Hugging Face did not resolve a commit for {spec.repo_id}")

        destination = models_root / spec.key
        marker = _read_marker(destination)
        unchanged = bool(
            marker
            and marker.get("repo_id") == spec.repo_id
            and marker.get("resolved_revision") == resolved_revision
            and marker.get("artifact_policy_version") == ARTIFACT_POLICY_VERSION
        )
        if unchanged and not force:
            print(f"  {spec.key}: already seeded at {resolved_revision[:12]}", flush=True)
        else:
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            print(f"  {spec.key}: downloading {resolved_revision[:12]}...", flush=True)
            snapshot_download(
                repo_id=spec.repo_id,
                revision=resolved_revision,
                local_dir=destination,
                token=os.getenv("HF_TOKEN"),
                max_workers=8,
                ignore_patterns=_snapshot_ignore_patterns(spec.key),
            )
            marker = {
                **asdict(spec),
                "artifact_policy_version": ARTIFACT_POLICY_VERSION,
                "requested_revision": spec.revision,
                "resolved_revision": resolved_revision,
                "seeded_at": datetime.now(UTC).isoformat(),
            }
            (destination / MARKER_NAME).write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        seeded.append(dict(marker or {}))

    state: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": seeded,
    }
    (models_root / STATE_NAME).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lockfile", type=Path, default=Path("/workspace/models.lock.json"))
    parser.add_argument("--models-root", type=Path, default=Path("/models"))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    state = seed(arguments.lockfile, arguments.models_root, force=arguments.force)
    print(f"Seeded {len(state['models'])} real model snapshots.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
