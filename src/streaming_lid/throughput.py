"""Pure validation and aggregation for offline-replay throughput results."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any


THROUGHPUT_SCHEMA_VERSION = 1
THROUGHPUT_METRIC = "offline_replay_frontend_student_wall_rtf"
THROUGHPUT_SCOPE = "whole_waveform_frontend_plus_uncached_streaming_replay"
EXPECTED_FRESH_PROCESSES = 7
EXPECTED_WARMUP_SWEEPS = 2
EXPECTED_MEASURED_SWEEPS = 10
MAX_PROCESS_MEDIAN_RATIO = 1.20
MAX_HALF_DRIFT = 0.10


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value for content-derived identities."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_finite(values: Sequence[float], *, name: str) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in converted):
        raise ValueError(f"{name} must contain only positive finite values")
    return converted


def _inclusive_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear inclusive quantile with endpoints at probabilities zero and one."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def summarize_process_runs(
    wall_rtf_runs: Sequence[float],
    process_cpu_seconds_per_audio_second_runs: Sequence[float],
) -> dict[str, Any]:
    """Summarize one fresh process without treating its sweeps as replicates."""
    wall = _positive_finite(wall_rtf_runs, name="wall_rtf_runs")
    process_cpu = _positive_finite(
        process_cpu_seconds_per_audio_second_runs,
        name="process_cpu_seconds_per_audio_second_runs",
    )
    if len(wall) != len(process_cpu):
        raise ValueError("wall and process-CPU sweep counts differ")
    midpoint = len(wall) // 2
    if midpoint == 0 or len(wall) - midpoint == 0:
        raise ValueError("at least two measured sweeps are required")
    first_half = wall[:midpoint]
    last_half = wall[midpoint:]
    first_median = float(statistics.median(first_half))
    last_median = float(statistics.median(last_half))
    half_drift = abs(last_median / first_median - 1.0)
    return {
        "measured_sweeps": len(wall),
        "median_wall_rtf": float(statistics.median(wall)),
        "median_process_cpu_seconds_per_audio_second": float(
            statistics.median(process_cpu)
        ),
        "first_half_median_wall_rtf": first_median,
        "last_half_median_wall_rtf": last_median,
        "absolute_half_drift_fraction": half_drift,
        "half_drift_passed": half_drift <= MAX_HALF_DRIFT,
    }


def aggregate_processes(
    processes: Sequence[Mapping[str, Any]],
    *,
    expected_processes: int = EXPECTED_FRESH_PROCESSES,
    expected_sweeps: int = EXPECTED_MEASURED_SWEEPS,
    require_complete_protocol: bool = True,
) -> dict[str, Any]:
    """Aggregate fresh-process medians and apply the predeclared smoke gates."""
    if not isinstance(expected_processes, int) or expected_processes <= 0:
        raise ValueError("expected_processes must be a positive integer")
    if not isinstance(expected_sweeps, int) or expected_sweeps <= 1:
        raise ValueError("expected_sweeps must be an integer greater than one")
    if not processes:
        raise ValueError("process rows must not be empty")
    if require_complete_protocol and len(processes) != expected_processes:
        raise ValueError(
            f"expected {expected_processes} fresh processes, got {len(processes)}"
        )

    first_binding = processes[0].get("binding")
    if not isinstance(first_binding, Mapping) or not first_binding:
        raise ValueError("every process must carry a non-empty binding")

    medians: list[float] = []
    process_summaries: list[dict[str, Any]] = []
    for expected_index, process in enumerate(processes):
        if process.get("process_index") != expected_index:
            raise ValueError("process rows must be in contiguous launch order")
        if process.get("binding") != first_binding:
            raise ValueError("process identity, output, affinity, or threading differs")
        summary = summarize_process_runs(
            process.get("wall_rtf_runs", ()),
            process.get("process_cpu_seconds_per_audio_second_runs", ()),
        )
        if require_complete_protocol and summary["measured_sweeps"] != expected_sweeps:
            raise ValueError(
                f"process {expected_index} has {summary['measured_sweeps']} sweeps; "
                f"expected {expected_sweeps}"
            )
        medians.append(summary["median_wall_rtf"])
        process_summaries.append(summary)

    ordered = sorted(medians)
    aggregate_median = float(statistics.median(ordered))
    minimum = float(ordered[0])
    maximum = float(ordered[-1])
    ratio = maximum / minimum
    deviations = sorted(abs(value - aggregate_median) for value in medians)
    protocol_compliant = (
        len(processes) == expected_processes
        and all(item["measured_sweeps"] == expected_sweeps for item in process_summaries)
    )
    dispersion_passed = ratio <= MAX_PROCESS_MEDIAN_RATIO
    half_drift_passed = all(item["half_drift_passed"] for item in process_summaries)
    stability_passed = protocol_compliant and dispersion_passed and half_drift_passed
    return {
        "outer_replication_unit": "fresh_process",
        "n_processes": len(processes),
        "n_independent_replicates": len(processes),
        "within_process_sweeps_are_independent_replicates": False,
        "expected_processes": expected_processes,
        "expected_sweeps_per_process": expected_sweeps,
        "protocol_compliant": protocol_compliant,
        "identity_bindings_equal": True,
        "process_median_wall_rtfs": medians,
        "median_wall_rtf": aggregate_median,
        "q1_wall_rtf": _inclusive_quantile(ordered, 0.25),
        "q3_wall_rtf": _inclusive_quantile(ordered, 0.75),
        "iqr_wall_rtf": (
            _inclusive_quantile(ordered, 0.75)
            - _inclusive_quantile(ordered, 0.25)
        ),
        "min_wall_rtf": minimum,
        "max_wall_rtf": maximum,
        "max_min_ratio": ratio,
        "median_absolute_deviation_wall_rtf": float(
            statistics.median(deviations)
        ),
        "max_process_median_ratio_gate": MAX_PROCESS_MEDIAN_RATIO,
        "max_half_drift_gate": MAX_HALF_DRIFT,
        "dispersion_gate_passed": dispersion_passed,
        "all_half_drift_gates_passed": half_drift_passed,
        "throughput_stability_passed": stability_passed,
        "x_realtime": (1.0 / aggregate_median) if stability_passed else None,
        "scalar_speed_headline_permitted": stability_passed,
        "process_summaries": process_summaries,
        "quantile_method": "linear_inclusive_v1",
    }


def reciprocal_interval(lower_rtf: float, upper_rtf: float) -> tuple[float, float]:
    """Transform an RTF interval to x-real-time, reversing its endpoints."""
    values = _positive_finite((lower_rtf, upper_rtf), name="rtf_interval")
    if values[0] > values[1]:
        raise ValueError("RTF interval endpoints are reversed")
    return (1.0 / values[1], 1.0 / values[0])


def measure_dual_clock(
    operation: Callable[[], Any],
    *,
    total_audio_seconds: float,
    wall_clock_ns: Callable[[], int],
    process_clock_ns: Callable[[], int],
) -> dict[str, Any]:
    """Measure one operation with independent monotonic wall and process clocks."""
    if not math.isfinite(total_audio_seconds) or total_audio_seconds <= 0.0:
        raise ValueError("total_audio_seconds must be positive and finite")
    wall_start = int(wall_clock_ns())
    process_start = int(process_clock_ns())
    operation()
    process_stop = int(process_clock_ns())
    wall_stop = int(wall_clock_ns())
    wall_delta = wall_stop - wall_start
    process_delta = process_stop - process_start
    if wall_delta <= 0 or process_delta <= 0:
        raise ValueError("wall and process clocks must advance")
    return {
        "wall_start_ns": wall_start,
        "wall_stop_ns": wall_stop,
        "wall_ns": wall_delta,
        "process_cpu_start_ns": process_start,
        "process_cpu_stop_ns": process_stop,
        "process_cpu_ns": process_delta,
        "wall_rtf": (wall_delta / 1_000_000_000.0) / total_audio_seconds,
        "process_cpu_seconds_per_audio_second": (
            process_delta / 1_000_000_000.0
        )
        / total_audio_seconds,
    }


def validate_worker_runtime(
    *,
    declared_affinity: Sequence[int],
    observed_affinity: Sequence[int],
    torch_intraop: int,
    torch_interop: int,
    environment: Mapping[str, str | None],
) -> dict[str, Any]:
    """Fail before timing when affinity or thread controls differ from contract."""
    declared = tuple(sorted(int(value) for value in declared_affinity))
    observed = tuple(sorted(int(value) for value in observed_affinity))
    if len(declared) != 6 or len(set(declared)) != 6:
        raise ValueError("declared affinity must contain six distinct CPUs")
    if observed != declared:
        raise RuntimeError(
            f"observed affinity {observed} differs from declared {declared}"
        )
    expected_environment = {
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
    }
    if int(torch_intraop) != 6 or int(torch_interop) != 1:
        raise RuntimeError("Torch thread counts differ from the 6/1 contract")
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise RuntimeError(f"{name} differs from the benchmark contract")
    return {
        "torch_intraop": 6,
        "torch_interop": 1,
        "omp_num_threads": 6,
        "mkl_num_threads": 6,
        "omp_proc_bind": "TRUE",
        "omp_places": "cores",
        "binding_policy": "six_distinct_visible_cores_first_logical_cpu_v1",
    }


def validate_timed_scope(*, metric: str, loaded_modules: Sequence[str]) -> None:
    """Protect the named student-only timing scope from teacher construction."""
    if metric != THROUGHPUT_METRIC:
        raise ValueError(f"unsupported throughput metric: {metric}")
    forbidden = sorted(
        name
        for name in loaded_modules
        if name == "speechbrain" or name.startswith("speechbrain.")
    )
    if forbidden:
        raise RuntimeError(f"offline teacher entered the timed child: {forbidden[:3]}")

