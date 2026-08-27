"""Deterministic, duration-aware, whole-file pytest sharding."""

from __future__ import annotations

import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

DURATION_HISTORY_SCHEMA_VERSION = 1
DEFAULT_DURATION_SECONDS = 1.0
DEFAULT_SMOOTHING_ALPHA = 0.35

DurationGranularity = Literal["file", "nodeid"]


@dataclass(frozen=True, slots=True)
class TestShard:
    index: int
    nodeids: tuple[str, ...]
    file_count: int
    test_count: int
    estimated_seconds: float


def parse_collected_nodeids(output: str) -> list[str]:
    """Extract pytest node IDs from ``pytest --collect-only -q`` output."""

    return [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("tests/") and ".py::" in line
    ]


def _coerce_duration(value: object, *, strings: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        seconds = float(value)
    elif strings and isinstance(value, str):
        try:
            seconds = float(value)
        except ValueError:
            return None
    else:
        return None
    return seconds if math.isfinite(seconds) else None


def _clean_durations(
    durations: Mapping[object, object],
    *,
    clamp_negative: bool,
) -> dict[str, float]:
    clean: dict[str, float] = {}
    for raw_key, raw_seconds in durations.items():
        key = str(raw_key).strip()
        seconds = _coerce_duration(raw_seconds)
        if not key or seconds is None:
            continue
        if seconds < 0:
            if not clamp_negative:
                continue
            seconds = 0.0
        clean[key] = seconds
    return clean


def load_durations(path: Path) -> dict[str, float]:
    """Load duration history without letting a damaged cache break a test run.

    Version 1 uses ``{"schema_version": 1, "durations": {...}}``. The original
    flat ``{nodeid: seconds}`` representation remains accepted so existing
    workspaces learn forward without a migration step.
    """

    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    if "schema_version" in raw or "durations" in raw:
        if raw.get("schema_version") != DURATION_HISTORY_SCHEMA_VERSION:
            return {}
        raw = raw.get("durations")
        if not isinstance(raw, dict):
            return {}

    return _clean_durations(raw, clamp_negative=True)


def save_durations(path: Path, durations: Mapping[str, float]) -> None:
    """Atomically persist a deterministic, versioned duration history file."""

    clean = _clean_durations(durations, clamp_negative=True)
    payload = {
        "schema_version": DURATION_HISTORY_SCHEMA_VERSION,
        "durations": dict(sorted(clean.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _normalise_test_path(value: str) -> str:
    candidate = value.strip().replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    marker = "/tests/"
    if marker in f"/{candidate}":
        candidate = "tests/" + f"/{candidate}".rsplit(marker, 1)[1]
    return candidate.lstrip("/")


def _file_module(file_path: str) -> str:
    path = file_path.removesuffix(".py").strip("/")
    return path.replace("/", ".")


def _known_test_maps(
    nodeids: Sequence[str],
) -> tuple[dict[str, str], dict[tuple[str, str], str | None]]:
    modules: dict[str, str] = {}
    cases: dict[tuple[str, str], str | None] = {}
    for nodeid in nodeids:
        file_path, separator, qualified_name = nodeid.partition("::")
        if not separator:
            continue
        file_path = _normalise_test_path(file_path)
        module = _file_module(file_path)
        modules[module] = file_path
        parts = qualified_name.split("::")
        classname = ".".join((module, *parts[:-1]))
        case_key = (classname, parts[-1])
        # Duplicate keys are deliberately marked ambiguous instead of silently
        # assigning their combined runtime to whichever test was seen last.
        cases[case_key] = nodeid if case_key not in cases else None
    return modules, cases


def _infer_file_from_classname(classname: str) -> str | None:
    parts = [part for part in classname.split(".") if part]
    candidates = [
        index
        for index, part in enumerate(parts)
        if part.startswith("test_") or part.endswith("_test")
    ]
    if not candidates:
        return None
    return "/".join(parts[: candidates[-1] + 1]) + ".py"


def _testcase_file(element: ET.Element, modules: Mapping[str, str]) -> str | None:
    file_attribute = element.get("file")
    if file_attribute:
        return _normalise_test_path(file_attribute)

    classname = element.get("classname", "")
    matches = [
        (module, file_path)
        for module, file_path in modules.items()
        if classname == module or classname.startswith(module + ".")
    ]
    if matches:
        return max(matches, key=lambda item: len(item[0]))[1]
    return _infer_file_from_classname(classname)


def _testcase_nodeid(
    element: ET.Element,
    file_path: str,
    known_cases: Mapping[tuple[str, str], str | None],
) -> str | None:
    name = element.get("name", "").strip()
    if not name:
        return None
    classname = element.get("classname", "").strip()
    known = known_cases.get((classname, name))
    if known:
        return known

    module = _file_module(file_path)
    qualifier = ""
    if classname.startswith(module + "."):
        qualifier = classname[len(module) + 1 :].replace(".", "::")
    parts = [file_path, qualifier, name]
    return "::".join(part for part in parts if part)


def _junit_paths(paths: Path | Sequence[Path]) -> Iterable[Path]:
    if isinstance(paths, Path):
        return (paths,)
    return paths


def parse_junit_durations(
    paths: Path | Sequence[Path],
    *,
    granularity: DurationGranularity = "file",
    known_nodeids: Sequence[str] = (),
) -> dict[str, float]:
    """Read pytest JUnit XML and aggregate observed testcase wall times.

    File-level observations are the default because files are the indivisible
    scheduling unit. Passing collected node IDs lets xunit2 ``classname`` and
    ``name`` attributes be mapped exactly, including classes and parameters.
    Missing, malformed, and partial XML files are ignored; duration history is
    an optimization and must never turn a passing suite into a failed run.
    """

    if granularity not in ("file", "nodeid"):
        raise ValueError("granularity must be 'file' or 'nodeid'")
    modules, known_cases = _known_test_maps(known_nodeids)
    observed: dict[str, float] = {}
    for path in _junit_paths(paths):
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for element in root.iter():
            if element.tag.rpartition("}")[2] != "testcase":
                continue
            seconds = _coerce_duration(element.get("time"), strings=True)
            if seconds is None or seconds < 0:
                continue
            file_path = _testcase_file(element, modules)
            if file_path is None:
                continue
            key = (
                file_path
                if granularity == "file"
                else _testcase_nodeid(element, file_path, known_cases)
            )
            if key is not None:
                observed[key] = observed.get(key, 0.0) + seconds
    return observed


def merge_durations(
    previous: Mapping[str, float],
    observations: Mapping[str, float],
    *,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
    active_keys: Iterable[str] | None = None,
) -> dict[str, float]:
    """Merge observations with an exponentially weighted moving average.

    A fixed alpha dampens cold-model and noisy Sidecar outliers while adapting
    to persistent changes. Unobserved history is retained unless ``active_keys``
    is provided, which allows callers to prune tests removed from the suite.
    """

    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in the interval (0, 1]")
    old = _clean_durations(previous, clamp_negative=False)
    new = _clean_durations(observations, clamp_negative=False)
    allowed = {str(key) for key in active_keys} if active_keys is not None else None
    merged = {key: seconds for key, seconds in old.items() if allowed is None or key in allowed}
    for key, seconds in new.items():
        if allowed is not None and key not in allowed:
            continue
        if key in merged:
            merged[key] = (1.0 - alpha) * merged[key] + alpha * seconds
        else:
            merged[key] = seconds
    return dict(sorted(merged.items()))


def plan_shards(
    nodeids: Sequence[str],
    shard_count: int,
    durations: Mapping[str, float] | None = None,
) -> list[TestShard]:
    """Balance whole files with deterministic longest-processing-time planning.

    Duration mappings may contain file paths (preferred) or individual node IDs.
    An exact file observation wins; node-level histories remain supported for
    backward compatibility.
    """

    if shard_count < 1:
        raise ValueError("shard_count must be at least one")
    if not nodeids:
        return []

    duration_map = _clean_durations(durations or {}, clamp_negative=True)
    file_values = [value for key, value in duration_map.items() if "::" not in key]
    node_values = [value for key, value in duration_map.items() if "::" in key]
    fallback_values = node_values or list(duration_map.values())
    default_node_seconds = median(fallback_values) if fallback_values else DEFAULT_DURATION_SECONDS
    default_file_seconds = median(file_values) if file_values else None

    by_file: OrderedDict[str, list[str]] = OrderedDict()
    for nodeid in nodeids:
        by_file.setdefault(nodeid.partition("::")[0], []).append(nodeid)

    weighted: list[tuple[str, tuple[str, ...], float]] = []
    for file_path, file_nodeids in by_file.items():
        if file_path in duration_map:
            seconds = duration_map[file_path]
        elif any(nodeid in duration_map for nodeid in file_nodeids):
            seconds = sum(duration_map.get(nodeid, default_node_seconds) for nodeid in file_nodeids)
        elif default_file_seconds is not None:
            seconds = default_file_seconds
        else:
            seconds = len(file_nodeids) * default_node_seconds
        weighted.append((file_path, tuple(file_nodeids), seconds))
    weighted.sort(key=lambda item: (-item[2], item[0]))

    count = min(shard_count, len(weighted))
    bins: list[list[tuple[str, tuple[str, ...], float]]] = [[] for _ in range(count)]
    totals = [0.0] * count
    for unit in weighted:
        target = min(range(count), key=lambda index: (totals[index], index))
        bins[target].append(unit)
        totals[target] += unit[2]

    file_order = {file_path: index for index, file_path in enumerate(by_file)}
    result: list[TestShard] = []
    for index, units in enumerate(bins):
        units.sort(key=lambda unit: file_order[unit[0]])
        nodes = tuple(nodeid for _, file_nodeids, _ in units for nodeid in file_nodeids)
        result.append(
            TestShard(
                index=index,
                nodeids=nodes,
                file_count=len(units),
                test_count=len(nodes),
                estimated_seconds=totals[index],
            )
        )
    return result
