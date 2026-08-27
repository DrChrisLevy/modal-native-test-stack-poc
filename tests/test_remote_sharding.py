from __future__ import annotations

import json
from pathlib import Path

import pytest

from modal_native_test_stack_poc.remote.sharding import (
    load_durations,
    merge_durations,
    parse_collected_nodeids,
    parse_junit_durations,
    plan_shards,
    save_durations,
)


def _write_junit(path: Path, testcases: str) -> Path:
    path.write_text(f'<testsuites><testsuite name="pytest">{testcases}</testsuite></testsuites>')
    return path


def test_parse_collected_nodeids_extracts_only_tests() -> None:
    output = """tests/test_a.py::test_one
tests/test_a.py::test_two[param]

2 tests collected in 0.01s
"""
    assert parse_collected_nodeids(output) == [
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two[param]",
    ]


def test_parse_collected_nodeids_ignores_warnings() -> None:
    output = "warning: tests/test_a.py did something\ntests/test_a.py::test_real"
    assert parse_collected_nodeids(output) == ["tests/test_a.py::test_real"]


def test_parse_collected_nodeids_accepts_nested_test_paths() -> None:
    output = "tests/model/test_text.py::TestEmbeddings::test_shape"
    assert parse_collected_nodeids(output) == [output]


def test_missing_duration_file_is_empty(tmp_path: Path) -> None:
    assert load_durations(tmp_path / "missing.json") == {}


@pytest.mark.parametrize("payload", [[], None, "durations", [1, 2]])
def test_non_mapping_duration_payload_is_empty(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "durations.json"
    path.write_text(json.dumps(payload))
    assert load_durations(path) == {}


def test_duration_loader_accepts_integer_and_float_values(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    path.write_text(json.dumps({"a": 1, "b": 2.5, "ignored": "3"}))
    assert load_durations(path) == {"a": 1.0, "b": 2.5}


def test_duration_loader_clamps_negative_values(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    path.write_text(json.dumps({"test": -4.5}))
    assert load_durations(path) == {"test": 0.0}


def test_duration_loader_ignores_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    path.write_text("{not json")
    assert load_durations(path) == {}


def test_duration_loader_ignores_nonfinite_and_boolean_values(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    path.write_text('{"good": 1.25, "nan": NaN, "infinity": Infinity, "flag": true}')
    assert load_durations(path) == {"good": 1.25}


def test_duration_history_round_trips_versioned_schema(tmp_path: Path) -> None:
    path = tmp_path / "history" / "test-durations.json"
    save_durations(path, {"tests/z.py": 2.5, "tests/a.py": 1.25})

    assert load_durations(path) == {"tests/a.py": 1.25, "tests/z.py": 2.5}
    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "durations": {"tests/a.py": 1.25, "tests/z.py": 2.5},
    }


def test_duration_history_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    path.write_text('{"schema_version": 99, "durations": {"tests/a.py": 3.0}}')
    assert load_durations(path) == {}


def test_duration_history_save_sanitizes_values(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    save_durations(
        path,
        {"tests/good.py": 4.0, "tests/negative.py": -2.0, "tests/nan.py": float("nan")},
    )
    assert load_durations(path) == {"tests/good.py": 4.0, "tests/negative.py": 0.0}


def test_junit_parser_aggregates_testcase_times_by_file(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "junit.xml",
        """
        <testcase classname="tests.api.test_assets" name="test_one" time="1.25" />
        <testcase classname="tests.api.test_assets" name="test_two[param]" time="2.5" />
        <testcase classname="tests.model.test_text" name="test_model" time="3" />
        """,
    )
    assert parse_junit_durations(junit) == {
        "tests/api/test_assets.py": 3.75,
        "tests/model/test_text.py": 3.0,
    }


def test_junit_parser_aggregates_across_shards(tmp_path: Path) -> None:
    first = _write_junit(
        tmp_path / "junit-1.xml",
        '<testcase classname="tests.test_a" name="test_one" time="1" />',
    )
    second = _write_junit(
        tmp_path / "junit-2.xml",
        '<testcase classname="tests.test_a" name="test_two" time="2" />'
        '<testcase classname="tests.test_b" name="test_three" time="4" />',
    )
    assert parse_junit_durations([first, second]) == {
        "tests/test_a.py": 3.0,
        "tests/test_b.py": 4.0,
    }


def test_junit_parser_maps_classes_and_parameters_to_exact_known_nodeids(tmp_path: Path) -> None:
    nodeids = [
        "tests/custom_spec.py::TestFeature::test_case[value.with.dots]",
        "tests/custom_spec.py::test_function",
    ]
    junit = _write_junit(
        tmp_path / "junit.xml",
        """
        <testcase classname="tests.custom_spec.TestFeature"
                  name="test_case[value.with.dots]" time="0.75" />
        <testcase classname="tests.custom_spec" name="test_function" time="0.25" />
        """,
    )
    assert parse_junit_durations(
        junit,
        granularity="nodeid",
        known_nodeids=nodeids,
    ) == {nodeids[0]: 0.75, nodeids[1]: 0.25}


def test_junit_parser_uses_known_nodeids_for_nonstandard_module_names(tmp_path: Path) -> None:
    nodeid = "tests/behavior_spec.py::test_contract"
    junit = _write_junit(
        tmp_path / "junit.xml",
        '<testcase classname="tests.behavior_spec" name="test_contract" time="1.5" />',
    )
    assert parse_junit_durations(junit, known_nodeids=[nodeid]) == {"tests/behavior_spec.py": 1.5}


def test_junit_parser_normalizes_absolute_and_windows_file_attributes(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "junit.xml",
        """
        <testcase file="/workspace/tests/api/test_a.py" name="test_one" time="1" />
        <testcase file="C:\\workspace\\tests\\api\\test_a.py" name="test_two" time="2" />
        """,
    )
    assert parse_junit_durations(junit) == {"tests/api/test_a.py": 3.0}


def test_junit_parser_understands_namespaced_xml(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite xmlns="urn:junit"><testcase classname="tests.test_a" '
        'name="test_one" time="1.25" /></testsuite>'
    )
    assert parse_junit_durations(junit) == {"tests/test_a.py": 1.25}


def test_junit_parser_ignores_missing_malformed_and_invalid_observations(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.xml"
    malformed.write_text("<testsuite>")
    valid = _write_junit(
        tmp_path / "valid.xml",
        """
        <testcase classname="tests.test_a" name="missing_time" />
        <testcase classname="tests.test_a" name="bad_time" time="slow" />
        <testcase classname="tests.test_a" name="negative" time="-2" />
        <testcase classname="application.module" name="not_a_test" time="9" />
        <testcase classname="tests.test_a" name="good" time="0.5" />
        """,
    )
    assert parse_junit_durations([tmp_path / "missing.xml", malformed, valid]) == {
        "tests/test_a.py": 0.5
    }


def test_junit_parser_rejects_unknown_granularity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="granularity"):
        parse_junit_durations(tmp_path / "unused.xml", granularity="case")  # type: ignore[arg-type]


def test_duration_merge_uses_stable_exponential_smoothing() -> None:
    assert merge_durations(
        {"tests/a.py": 10.0},
        {"tests/a.py": 20.0, "tests/b.py": 5.0},
        alpha=0.25,
    ) == {"tests/a.py": 12.5, "tests/b.py": 5.0}


def test_duration_merge_retains_unobserved_history() -> None:
    assert merge_durations(
        {"tests/a.py": 1.0, "tests/b.py": 2.0},
        {"tests/a.py": 3.0},
        alpha=0.5,
    ) == {"tests/a.py": 2.0, "tests/b.py": 2.0}


def test_duration_merge_can_prune_removed_files() -> None:
    assert merge_durations(
        {"tests/removed.py": 9.0, "tests/kept.py": 3.0},
        {"tests/kept.py": 5.0, "tests/not_collected.py": 100.0},
        alpha=0.5,
        active_keys={"tests/kept.py"},
    ) == {"tests/kept.py": 4.0}


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_duration_merge_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        merge_durations({}, {}, alpha=alpha)


@pytest.mark.parametrize("shard_count", [0, -1, -100])
def test_shard_count_must_be_positive(shard_count: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plan_shards(["tests/a.py::test_a"], shard_count)


def test_empty_collection_produces_no_shards() -> None:
    assert plan_shards([], 4) == []


def test_shard_count_is_capped_at_number_of_files() -> None:
    shards = plan_shards(
        ["tests/a.py::test_a", "tests/b.py::test_b", "tests/c.py::test_c"],
        20,
    )
    assert len(shards) == 3


def test_tests_from_one_file_are_never_split() -> None:
    nodeids = [
        "tests/a.py::test_a1",
        "tests/a.py::test_a2",
        "tests/b.py::test_b1",
        "tests/b.py::test_b2",
    ]
    shards = plan_shards(nodeids, 2)
    locations = {nodeid: shard.index for shard in shards for nodeid in shard.nodeids}
    assert locations[nodeids[0]] == locations[nodeids[1]]
    assert locations[nodeids[2]] == locations[nodeids[3]]


def test_every_collected_test_is_assigned_exactly_once() -> None:
    nodeids = [f"tests/test_{index}.py::test_case" for index in range(10)]
    assigned = [nodeid for shard in plan_shards(nodeids, 4) for nodeid in shard.nodeids]
    assert sorted(assigned) == sorted(nodeids)


def test_shard_metadata_matches_assigned_nodes() -> None:
    nodeids = [
        "tests/a.py::test_a1",
        "tests/a.py::test_a2",
        "tests/b.py::test_b",
    ]
    for shard in plan_shards(nodeids, 2):
        assert shard.test_count == len(shard.nodeids)
        assert shard.file_count == len({nodeid.partition("::")[0] for nodeid in shard.nodeids})


def test_unknown_durations_use_known_median() -> None:
    nodeids = ["tests/a.py::test_a", "tests/b.py::test_b", "tests/c.py::test_c"]
    shards = plan_shards(nodeids, 3, {nodeids[0]: 3.0, nodeids[1]: 5.0})
    estimated = {shard.nodeids[0]: shard.estimated_seconds for shard in shards}
    assert estimated[nodeids[2]] == 4.0


def test_duration_aware_planner_balances_long_files() -> None:
    nodeids = [
        "tests/slow.py::test_one",
        "tests/medium.py::test_one",
        "tests/fast_a.py::test_one",
        "tests/fast_b.py::test_one",
    ]
    durations = {
        nodeids[0]: 10.0,
        nodeids[1]: 6.0,
        nodeids[2]: 2.0,
        nodeids[3]: 2.0,
    }
    totals = sorted(shard.estimated_seconds for shard in plan_shards(nodeids, 2, durations))
    assert totals == [10.0, 10.0]


def test_planner_prefers_file_level_duration_history() -> None:
    nodeids = [
        "tests/slow.py::test_one",
        "tests/slow.py::test_two",
        "tests/fast.py::test_one",
    ]
    shards = plan_shards(
        nodeids,
        2,
        {
            "tests/slow.py": 20.0,
            "tests/slow.py::test_one": 100.0,
            "tests/fast.py": 1.0,
        },
    )
    estimated = {shard.nodeids[0].partition("::")[0]: shard.estimated_seconds for shard in shards}
    assert estimated == {"tests/slow.py": 20.0, "tests/fast.py": 1.0}


def test_unknown_files_use_median_file_duration() -> None:
    nodeids = [
        "tests/known_a.py::test_one",
        "tests/known_b.py::test_one",
        "tests/new.py::test_one",
    ]
    shards = plan_shards(
        nodeids,
        3,
        {"tests/known_a.py": 2.0, "tests/known_b.py": 6.0},
    )
    estimated = {shard.nodeids[0]: shard.estimated_seconds for shard in shards}
    assert estimated["tests/new.py::test_one"] == 4.0


def test_file_duration_lpt_is_deterministic_on_uneven_weights() -> None:
    nodeids = [f"tests/{name}.py::test_one" for name in ("a", "b", "c", "d", "e")]
    durations = {
        "tests/a.py": 9.0,
        "tests/b.py": 8.0,
        "tests/c.py": 7.0,
        "tests/d.py": 6.0,
        "tests/e.py": 5.0,
    }
    totals = sorted(shard.estimated_seconds for shard in plan_shards(nodeids, 2, durations))
    assert totals == [15.0, 20.0]


def test_planner_is_deterministic_for_equal_weights() -> None:
    nodeids = [f"tests/{name}.py::test_one" for name in ("c", "a", "b", "d")]
    first = plan_shards(nodeids, 2)
    second = plan_shards(nodeids, 2)
    assert first == second


def test_shard_indexes_are_dense_and_zero_based() -> None:
    nodeids = [f"tests/{name}.py::test_one" for name in ("a", "b", "c")]
    assert [shard.index for shard in plan_shards(nodeids, 3)] == [0, 1, 2]
