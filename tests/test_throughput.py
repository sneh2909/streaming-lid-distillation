import math

import pytest

from streaming_lid.throughput import (
    THROUGHPUT_METRIC,
    aggregate_processes,
    measure_dual_clock,
    reciprocal_interval,
    summarize_process_runs,
    validate_timed_scope,
    validate_worker_runtime,
)


def _process(
    index: int,
    median: float,
    *,
    binding: dict | None = None,
    runs: list[float] | None = None,
) -> dict:
    wall = runs if runs is not None else [median] * 10
    return {
        "process_index": index,
        "binding": binding or {"identity": "same"},
        "wall_rtf_runs": wall,
        "process_cpu_seconds_per_audio_second_runs": [value * 5 for value in wall],
    }


def test_aggregator_uses_seven_process_medians_not_seventy_sweeps() -> None:
    medians = [0.0100, 0.0101, 0.0102, 0.0103, 0.0104, 0.0105, 0.0106]
    processes = [_process(index, value) for index, value in enumerate(medians)]
    result = aggregate_processes(processes)

    assert result["n_processes"] == 7
    assert result["n_independent_replicates"] == 7
    assert not result["within_process_sweeps_are_independent_replicates"]
    assert result["median_wall_rtf"] == pytest.approx(0.0103)
    assert result["process_median_wall_rtfs"] == pytest.approx(medians)
    assert result["throughput_stability_passed"]
    assert result["x_realtime"] == pytest.approx(1 / 0.0103)

    permuted = [
        _process(index, value, runs=[value] * 10)
        for index, value in enumerate(medians)
    ]
    assert aggregate_processes(permuted)["median_wall_rtf"] == result[
        "median_wall_rtf"
    ]


def test_dispersion_and_half_drift_boundaries_are_inclusive() -> None:
    binding = {"identity": "boundary"}
    exact_ratio = [
        _process(index, 1.0 if index < 6 else 1.2, binding=binding)
        for index in range(7)
    ]
    assert aggregate_processes(exact_ratio)["dispersion_gate_passed"]
    exact_ratio[-1] = _process(
        6, math.nextafter(1.2, math.inf), binding=binding
    )
    assert not aggregate_processes(exact_ratio)["dispersion_gate_passed"]

    exact_drift = summarize_process_runs([10.0] * 5 + [11.0] * 5, [1.0] * 10)
    assert exact_drift["half_drift_passed"]
    over_drift = summarize_process_runs(
        [10.0] * 5 + [math.nextafter(11.0, math.inf)] * 5,
        [1.0] * 10,
    )
    assert not over_drift["half_drift_passed"]


def test_historical_opportunistic_fixture_fails_seven_process_protocol() -> None:
    medians = [
        0.0883287678,
        0.0171434637,
        0.0058373628,
        0.0066097918,
    ]
    processes = [_process(index, value) for index, value in enumerate(medians)]
    result = aggregate_processes(
        processes,
        expected_processes=7,
        require_complete_protocol=False,
    )

    assert result["historical_opportunistic_n"] == 4
    assert result["median_wall_rtf"] == pytest.approx(0.01187662775)
    assert result["min_wall_rtf"] == pytest.approx(0.0058373628)
    assert result["max_wall_rtf"] == pytest.approx(0.0883287678)
    assert result["max_min_ratio"] == pytest.approx(15.1316221, rel=1e-7)
    assert not result["protocol_compliant"]
    assert not result["throughput_stability_passed"]
    assert result["x_realtime"] is None


def test_aggregation_rejects_identity_or_output_binding_changes() -> None:
    processes = [_process(index, 0.01) for index in range(7)]
    processes[4] = _process(4, 0.01, binding={"identity": "changed-output"})
    with pytest.raises(ValueError, match="identity, output, affinity, or threading"):
        aggregate_processes(processes)


@pytest.mark.parametrize(
    "values",
    [[], [0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]],
)
def test_invalid_timing_arrays_fail(values: list[float]) -> None:
    with pytest.raises(ValueError, match="positive finite|empty"):
        summarize_process_runs(values, values)


def test_worker_runtime_contract_is_exact() -> None:
    environment = {
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
    }
    contract = validate_worker_runtime(
        declared_affinity=[0, 2, 4, 6, 8, 10],
        observed_affinity=[10, 8, 6, 4, 2, 0],
        torch_intraop=6,
        torch_interop=1,
        environment=environment,
    )
    assert contract["torch_intraop"] == 6
    assert contract["torch_interop"] == 1

    with pytest.raises(RuntimeError, match="observed affinity"):
        validate_worker_runtime(
            declared_affinity=[0, 2, 4, 6, 8, 10],
            observed_affinity=[0, 2, 4, 6, 8],
            torch_intraop=6,
            torch_interop=1,
            environment=environment,
        )
    with pytest.raises(RuntimeError, match="thread counts"):
        validate_worker_runtime(
            declared_affinity=[0, 2, 4, 6, 8, 10],
            observed_affinity=[0, 2, 4, 6, 8, 10],
            torch_intraop=6,
            torch_interop=2,
            environment=environment,
        )


def test_dual_clock_measurement_and_reciprocal_interval() -> None:
    wall_values = iter([1_000_000_000, 3_000_000_000])
    cpu_values = iter([4_000_000_000, 5_000_000_000])
    called: list[bool] = []
    result = measure_dual_clock(
        lambda: called.append(True),
        total_audio_seconds=2.0,
        wall_clock_ns=lambda: next(wall_values),
        process_clock_ns=lambda: next(cpu_values),
    )
    assert called == [True]
    assert result["wall_ns"] == 2_000_000_000
    assert result["process_cpu_ns"] == 1_000_000_000
    assert result["wall_rtf"] == 1.0
    assert result["process_cpu_seconds_per_audio_second"] == 0.5
    assert reciprocal_interval(0.25, 0.5) == (2.0, 4.0)


def test_named_scope_rejects_teacher_construction() -> None:
    validate_timed_scope(metric=THROUGHPUT_METRIC, loaded_modules=["torch"])
    with pytest.raises(RuntimeError, match="offline teacher"):
        validate_timed_scope(
            metric=THROUGHPUT_METRIC,
            loaded_modules=["torch", "speechbrain.inference.classifiers"],
        )
    with pytest.raises(ValueError, match="unsupported throughput metric"):
        validate_timed_scope(metric="frontend_only", loaded_modules=[])
