#!/usr/bin/env python3
"""Run a source/input-bound, fresh-process offline-replay CPU benchmark."""

from __future__ import annotations

import os
import sys

_SOURCE_STAGE_CONTEXT = None
if __name__ == "__main__":
    from source_stage import (
        capture_source_snapshot,
        is_staged_child,
        prepare_staged_child,
        run_snapshot_child,
    )

    if not is_staged_child():
        from pathlib import Path

        workspace = Path(__file__).resolve().parents[1]
        snapshot = capture_source_snapshot(workspace)
        raise SystemExit(
            run_snapshot_child(
                snapshot,
                entrypoint="scripts/benchmark_throughput.py",
                argv=sys.argv[1:],
                workspace_root=workspace,
                extra_environment={
                    "OMP_NUM_THREADS": "6",
                    "MKL_NUM_THREADS": "6",
                    "OMP_PROC_BIND": "TRUE",
                    "OMP_PLACES": "cores",
                },
            )
        )
    _SOURCE_STAGE_CONTEXT = prepare_staged_child(
        "scripts/benchmark_throughput.py"
    )

import argparse
import hashlib
import io
import json
import math
import platform
import resource
import shutil
import stat
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_STAGE_SCHEMA_VERSION = 1
INPUT_STAGE_MANIFEST = ".throughput_input.json"
INPUT_STAGE_DIRECTORY = Path("data/generated/throughput_benchmark_inputs")
RESULT_DIRECTORY_NAME = "throughput"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _wav_properties(payload: bytes, *, label: str) -> dict[str, int]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            properties = {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "sample_count": handle.getnframes(),
            }
    except (wave.Error, EOFError) as error:
        raise RuntimeError(f"{label} is not a readable PCM WAV: {error}") from error
    if properties["channels"] != 1:
        raise RuntimeError(f"{label} must be mono")
    if properties["sample_width_bytes"] != 2:
        raise RuntimeError(f"{label} must use 16-bit PCM")
    if properties["sample_rate"] != 16_000:
        raise RuntimeError(f"{label} must be 16 kHz")
    if properties["sample_count"] <= 0:
        raise RuntimeError(f"{label} must contain audio samples")
    return properties


def _safe_clip_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("held-out manifest row has an invalid clip ID")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise RuntimeError(f"clip ID is unsafe for a staged filename: {value!r}")
    return value


def _capture_input_stage(
    *,
    workspace: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    summary_path: Path,
) -> tuple[Path, dict[str, Any], bytes]:
    manifest_payload = manifest_path.read_bytes()
    summary_payload = summary_path.read_bytes()
    checkpoint_payload = checkpoint_path.read_bytes()
    summary = _read_json_bytes(summary_payload, label=str(summary_path))
    checkpoint_sha256 = _sha256(checkpoint_payload)
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("summary checkpoint hash differs from captured checkpoint bytes")

    records: list[dict[str, Any]] = []
    try:
        lines = manifest_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("manifest is not UTF-8") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = _read_json_bytes(line.encode("utf-8"), label=f"manifest line {line_number}")
        if value.get("split") == "heldout":
            records.append(value)
    if len(records) != 21:
        raise RuntimeError(f"expected 21 held-out clips, found {len(records)}")

    manifest_directory = manifest_path.resolve().parent
    captured_audio: list[tuple[str, bytes]] = []
    clip_ledger: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        clip_id = _safe_clip_id(record.get("id"))
        if clip_id in seen_ids:
            raise RuntimeError(f"duplicate held-out clip ID: {clip_id}")
        seen_ids.add(clip_id)
        audio_relative = record.get("audio_path")
        if not isinstance(audio_relative, str) or not audio_relative:
            raise RuntimeError(f"held-out clip {clip_id} has no audio path")
        source_audio = (manifest_directory / audio_relative).resolve()
        try:
            source_audio.relative_to(workspace)
        except ValueError as error:
            raise RuntimeError(f"held-out audio escapes workspace: {source_audio}") from error
        if source_audio.is_symlink() or not source_audio.is_file():
            raise RuntimeError(f"held-out audio is missing or unsafe: {source_audio}")
        payload = source_audio.read_bytes()
        properties = _wav_properties(payload, label=clip_id)
        staged_relative = f"audio/{index:02d}-{clip_id}.wav"
        sha256 = _sha256(payload)
        declared_sha256 = record.get("audio_sha256")
        if declared_sha256 != sha256:
            raise RuntimeError(f"manifest WAV hash differs for {clip_id}")
        captured_audio.append((staged_relative, payload))
        clip_ledger.append(
            {
                "index": index,
                "id": clip_id,
                "language": record.get("language"),
                "staged_path": staged_relative,
                "wav_sha256": sha256,
                "wav_bytes": len(payload),
                **properties,
            }
        )

    total_samples = sum(item["sample_count"] for item in clip_ledger)
    sample_rates = {item["sample_rate"] for item in clip_ledger}
    if sample_rates != {16_000}:
        raise RuntimeError("held-out clips do not share the expected sample rate")
    ordered_ids = [item["id"] for item in clip_ledger]
    ordered_wav_hashes = [item["wav_sha256"] for item in clip_ledger]
    input_identity = {
        "schema_version": INPUT_STAGE_SCHEMA_VERSION,
        "source_manifest_file_sha256": _sha256(manifest_payload),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": len(checkpoint_payload),
        "base_summary_sha256": _sha256(summary_payload),
        "expected_release": {
            "run_id": summary.get("run_id"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "model_state_sha256": summary.get("model_state_sha256"),
            "audio_files_sha256": summary.get("audio_files_sha256"),
            "pipeline_source_sha256": summary.get("pipeline_source_sha256"),
        },
        "n_clips": len(clip_ledger),
        "sample_rate": 16_000,
        "total_samples": total_samples,
        "total_audio_seconds": total_samples / 16_000,
        "ordered_clip_ids_sha256": _sha256(_canonical_json_bytes(ordered_ids)),
        "ordered_wav_sha256": _sha256(_canonical_json_bytes(ordered_wav_hashes)),
        "clips": clip_ledger,
    }
    root_sha256 = _sha256(_canonical_json_bytes(input_identity))
    input_manifest = {
        **input_identity,
        "input_stage_root_sha256": root_sha256,
        "checkpoint_staged_path": "checkpoint.pt",
    }
    parent = workspace / INPUT_STAGE_DIRECTORY
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / root_sha256
    if destination.exists():
        verified = _verify_input_stage(destination, expected_root_sha256=root_sha256)
        return destination, verified, summary_payload

    temporary = Path(tempfile.mkdtemp(prefix=".throughput-input-", dir=parent))
    try:
        (temporary / "audio").mkdir()
        (temporary / "checkpoint.pt").write_bytes(checkpoint_payload)
        for relative, payload in captured_audio:
            path = temporary / relative
            path.write_bytes(payload)
            path.chmod(0o444)
        (temporary / "checkpoint.pt").chmod(0o444)
        manifest_output = temporary / INPUT_STAGE_MANIFEST
        manifest_output.write_bytes(_canonical_json_bytes(input_manifest) + b"\n")
        manifest_output.chmod(0o444)
        _verify_input_stage(temporary, expected_root_sha256=root_sha256)
        try:
            temporary.rename(destination)
        except FileExistsError:
            _verify_input_stage(destination, expected_root_sha256=root_sha256)
        verified = _verify_input_stage(destination, expected_root_sha256=root_sha256)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination, verified, summary_payload


def _verify_input_stage(
    stage_root: Path, *, expected_root_sha256: str | None = None
) -> dict[str, Any]:
    root = stage_root.resolve()
    manifest_path = root / INPUT_STAGE_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("throughput input-stage manifest is missing or unsafe")
    manifest = _read_json_bytes(
        manifest_path.read_bytes(), label=str(manifest_path)
    )
    if manifest.get("schema_version") != INPUT_STAGE_SCHEMA_VERSION:
        raise RuntimeError("throughput input-stage schema is unsupported")
    declared_root = manifest.get("input_stage_root_sha256")
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"input_stage_root_sha256", "checkpoint_staged_path"}
    }
    recomputed_root = _sha256(_canonical_json_bytes(identity))
    if declared_root != recomputed_root:
        raise RuntimeError("throughput input-stage root contradicts its manifest")
    if expected_root_sha256 is not None and declared_root != expected_root_sha256:
        raise RuntimeError("throughput input-stage root differs from request")

    declared_files = {INPUT_STAGE_MANIFEST, "checkpoint.pt"}
    declared_files.update(item["staged_path"] for item in manifest.get("clips", []))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != declared_files:
        raise RuntimeError("throughput input-stage file set differs from manifest")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"throughput input stage contains a symlink: {path}")

    checkpoint_path = root / "checkpoint.pt"
    if _file_sha256(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("staged checkpoint hash changed")
    if checkpoint_path.stat().st_size != manifest.get("checkpoint_bytes"):
        raise RuntimeError("staged checkpoint byte count changed")
    total_samples = 0
    for item in manifest.get("clips", []):
        audio_path = root / item["staged_path"]
        if not audio_path.is_file() or audio_path.is_symlink():
            raise RuntimeError(f"staged audio is missing or unsafe: {item['id']}")
        payload = audio_path.read_bytes()
        if _sha256(payload) != item["wav_sha256"] or len(payload) != item["wav_bytes"]:
            raise RuntimeError(f"staged audio bytes changed: {item['id']}")
        properties = _wav_properties(payload, label=item["id"])
        for key, value in properties.items():
            if item.get(key) != value:
                raise RuntimeError(f"staged WAV metadata changed: {item['id']} {key}")
        total_samples += properties["sample_count"]
    if total_samples != manifest.get("total_samples"):
        raise RuntimeError("staged total sample count changed")
    return manifest


def _derive_six_core_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("benchmark requires Linux sched affinity support")
    allowed = sorted(os.sched_getaffinity(0))
    core_to_cpu: dict[tuple[str, str], int] = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError as error:
            raise RuntimeError(f"cannot derive topology for CPU {cpu}: {error}") from error
        core_to_cpu.setdefault((package, core), cpu)
    selected = list(core_to_cpu.values())[:6]
    if len(selected) != 6:
        raise RuntimeError(
            f"benchmark requires six distinct visible cores; found {len(core_to_cpu)}"
        )
    return selected


def _read_first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError, UnicodeDecodeError):
        return None


def _system_diagnostics(affinity: list[int]) -> dict[str, Any]:
    release = platform.release()
    environment_class = (
        "WSL2_observational" if "microsoft" in release.lower() else "Linux_observational"
    )
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    governors = {
        value
        for cpu in affinity
        if (value := _read_first_line(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")))
    }
    governor: str | list[str] | None
    if not governors:
        governor = None
    elif len(governors) == 1:
        governor = next(iter(governors))
    else:
        governor = sorted(governors)
    return {
        "environment_class": environment_class,
        "platform": platform.platform(),
        "kernel_release": release,
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "visible_affinity_before_binding": sorted(os.sched_getaffinity(0)),
        "affinity_cpu_ids": affinity,
        "affinity_distinct_physical_cores": True,
        "governor": governor,
        "host_power_mode": None,
        "host_power_mode_observable_from_guest": False,
        "load_average_before": list(os.getloadavg()),
    }


def _proc_status_snapshot() -> dict[str, int | None]:
    values: dict[str, int | None] = {
        "voluntary_context_switches": None,
        "nonvoluntary_context_switches": None,
    }
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("voluntary_ctxt_switches:"):
                values["voluntary_context_switches"] = int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                values["nonvoluntary_context_switches"] = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return values


def _thread_affinity_snapshot(declared_affinity: list[int]) -> dict[str, Any]:
    declared = set(declared_affinity)
    observed: list[list[int]] = []
    for task_path in sorted(
        Path("/proc/self/task").iterdir(), key=lambda path: int(path.name)
    ):
        if not task_path.name.isdigit():
            continue
        try:
            mask = sorted(os.sched_getaffinity(int(task_path.name)))
        except ProcessLookupError:
            continue
        if not mask or not set(mask).issubset(declared):
            raise RuntimeError(
                f"worker thread affinity {mask} escapes declared {sorted(declared)}"
            )
        observed.append(mask)
    union = sorted({cpu for mask in observed for cpu in mask})
    if union != sorted(declared):
        raise RuntimeError(
            f"worker thread affinity union {union} differs from declared "
            f"{sorted(declared)}"
        )
    return {
        "thread_count": len(observed),
        "unique_thread_affinity_masks": sorted({tuple(mask) for mask in observed}),
        "union_cpu_ids": union,
        "all_masks_within_declared_affinity": True,
        "union_equals_declared_affinity": True,
    }


def _worker(args: argparse.Namespace) -> None:
    if _SOURCE_STAGE_CONTEXT is None:
        raise RuntimeError("throughput worker must run from the verified source stage")
    affinity = [int(value) for value in args.affinity.split(",") if value]
    os.sched_setaffinity(0, set(affinity))
    observed_affinity_before_torch = sorted(os.sched_getaffinity(0))

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from source_stage import executed_source_identity, verify_source_snapshot
    from streaming_lid.audio import LogMelFrontend, load_audio
    from streaming_lid.model import CausalLIDStudent
    from streaming_lid.run_identity import model_state_sha256, run_id_for_identity
    from streaming_lid.throughput import (
        THROUGHPUT_METRIC,
        THROUGHPUT_SCOPE,
        measure_dual_clock,
        validate_timed_scope,
        validate_worker_runtime,
    )

    runtime_threading = validate_worker_runtime(
        declared_affinity=affinity,
        observed_affinity=observed_affinity_before_torch,
        torch_intraop=torch.get_num_threads(),
        torch_interop=torch.get_num_interop_threads(),
        environment={
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
        },
    )
    validate_timed_scope(metric=THROUGHPUT_METRIC, loaded_modules=tuple(sys.modules))
    input_stage = Path(args.input_stage).resolve()
    input_manifest = _verify_input_stage(
        input_stage, expected_root_sha256=args.input_stage_sha256
    )
    checkpoint_path = input_stage / input_manifest["checkpoint_staged_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, dict):
        raise RuntimeError("benchmark checkpoint has no run identity")
    if checkpoint.get("run_id") != run_id_for_identity(run_identity):
        raise RuntimeError("benchmark checkpoint run ID is invalid")
    state_sha256 = model_state_sha256(checkpoint.get("model_state", {}))
    expected_release = input_manifest["expected_release"]
    if checkpoint.get("run_id") != expected_release["run_id"]:
        raise RuntimeError("benchmark checkpoint run differs from captured summary")
    if state_sha256 != expected_release["model_state_sha256"]:
        raise RuntimeError("benchmark model state differs from captured summary")
    source_identity = executed_source_identity()
    checkpoint_source = run_identity.get("pipeline", {}).get("source")
    if not isinstance(checkpoint_source, dict):
        raise RuntimeError("benchmark checkpoint has no executed-source identity")
    performance_files = (
        "src/streaming_lid/__init__.py",
        "src/streaming_lid/audio.py",
        "src/streaming_lid/config.py",
        "src/streaming_lid/model.py",
    )
    for relative in performance_files:
        if checkpoint_source.get("files", {}).get(relative) != source_identity.get(
            "files", {}
        ).get(relative):
            raise RuntimeError(
                f"benchmark performance source differs from checkpoint: {relative}"
            )
    verify_source_snapshot(
        _SOURCE_STAGE_CONTEXT["stage_root"],
        expected_root_sha256=_SOURCE_STAGE_CONTEXT["snapshot_root_sha256"],
    )

    pipeline = run_identity["pipeline"]
    frontend_config = pipeline["frontend"]
    streaming_config = pipeline["streaming"]
    if int(frontend_config["sample_rate"]) != input_manifest["sample_rate"]:
        raise RuntimeError("benchmark sample rate differs from checkpoint frontend")
    chunk_frames = int(streaming_config["chunk_frames"])
    model = CausalLIDStudent(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = LogMelFrontend().eval()
    waveforms: list[tuple[str, torch.Tensor]] = []
    for item in input_manifest["clips"]:
        waveform = load_audio(input_stage / item["staged_path"])
        if len(waveform) != item["sample_count"]:
            raise RuntimeError(f"decoded sample count changed: {item['id']}")
        waveforms.append((item["id"], waveform))
    total_audio_seconds = float(input_manifest["total_audio_seconds"])

    def one_sweep() -> str:
        digest = hashlib.sha256()
        with torch.inference_mode():
            for clip_id, waveform in waveforms:
                features = frontend(waveform)
                logits = model.streaming_forward(
                    features, chunk_frames=chunk_frames
                ).detach().cpu().contiguous()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"non-finite benchmark output: {clip_id}")
                digest.update(clip_id.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(logits.dtype).encode("ascii"))
                digest.update(b"\0")
                digest.update(_canonical_json_bytes(list(logits.shape)))
                digest.update(b"\0")
                digest.update(logits.numpy().tobytes(order="C"))
        return digest.hexdigest()

    warmup_digests = [one_sweep() for _ in range(args.warmups)]
    if len(set(warmup_digests)) != 1:
        raise RuntimeError("full-corpus warm-up output digests differ")
    output_digest = warmup_digests[0]
    thread_affinity = _thread_affinity_snapshot(affinity)
    clocks: list[dict[str, Any]] = []
    measured_digests: list[str] = []
    for sweep_index in range(args.sweeps):
        holder: list[str] = []
        status_before = _proc_status_snapshot()
        load_before = list(os.getloadavg())
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        measured = measure_dual_clock(
            lambda: holder.append(one_sweep()),
            total_audio_seconds=total_audio_seconds,
            wall_clock_ns=time.perf_counter_ns,
            process_clock_ns=time.process_time_ns,
        )
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        status_after = _proc_status_snapshot()
        measured_digest = holder[0]
        if measured_digest != output_digest:
            raise RuntimeError("measured output digest differs from warmed output")
        measured_digests.append(measured_digest)
        clocks.append(
            {
                "sweep_index": sweep_index,
                **measured,
                "load_average_before": load_before,
                "load_average_after": list(os.getloadavg()),
                "voluntary_context_switch_delta": (
                    usage_after.ru_nvcsw - usage_before.ru_nvcsw
                ),
                "involuntary_context_switch_delta": (
                    usage_after.ru_nivcsw - usage_before.ru_nivcsw
                ),
                "proc_status_before": status_before,
                "proc_status_after": status_after,
                "output_digest_sha256": measured_digest,
            }
        )
    validate_timed_scope(metric=THROUGHPUT_METRIC, loaded_modules=tuple(sys.modules))
    binding = {
        "schema_version": 1,
        "metric": THROUGHPUT_METRIC,
        "scope": THROUGHPUT_SCOPE,
        "teacher_in_timed_region": False,
        "executed_source_root_sha256": source_identity["source_sha256"],
        "checkpoint_source_root_sha256": checkpoint_source["source_sha256"],
        "performance_source_files_equal_to_checkpoint": True,
        "benchmark_entrypoint_sha256": source_identity["files"][
            "scripts/benchmark_throughput.py"
        ]["sha256"],
        "checkpoint_sha256": input_manifest["checkpoint_sha256"],
        "run_id": checkpoint["run_id"],
        "model_state_sha256": state_sha256,
        "input_stage_root_sha256": input_manifest["input_stage_root_sha256"],
        "ordered_clip_ids_sha256": input_manifest["ordered_clip_ids_sha256"],
        "ordered_wav_sha256": input_manifest["ordered_wav_sha256"],
        "audio_files_sha256": expected_release["audio_files_sha256"],
        "n_clips": input_manifest["n_clips"],
        "total_samples": input_manifest["total_samples"],
        "sample_rate": input_manifest["sample_rate"],
        "total_audio_seconds": total_audio_seconds,
        "frontend": frontend_config,
        "chunk_frames": chunk_frames,
        "dtype": "torch.float32",
        "device": "cpu",
        "model_eval": True,
        "torch_inference_mode": True,
        "output_digest_sha256": output_digest,
        "affinity_cpu_ids": affinity,
        "threading": runtime_threading,
    }
    row = {
        "process_index": args.process_index,
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
        "full_corpus_warmups": args.warmups,
        "warmup_output_digests_sha256": warmup_digests,
        "wall_rtf_runs": [item["wall_rtf"] for item in clocks],
        "process_cpu_seconds_per_audio_second_runs": [
            item["process_cpu_seconds_per_audio_second"] for item in clocks
        ],
        "sweeps": clocks,
        "observed_affinity_cpu_ids_before_torch": observed_affinity_before_torch,
        "observed_main_thread_affinity_cpu_ids_after_torch": sorted(
            os.sched_getaffinity(0)
        ),
        "observed_thread_affinity_after_warmup": thread_affinity,
        "observed_torch_intraop": torch.get_num_threads(),
        "observed_torch_interop": torch.get_num_interop_threads(),
        "loaded_offline_teacher": False,
    }
    output_path = Path(args.worker_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(row) + b"\n")
    os.replace(temporary, output_path)


def _coordinator(args: argparse.Namespace) -> None:
    if _SOURCE_STAGE_CONTEXT is None:
        raise RuntimeError("benchmark coordinator must run from a verified source stage")
    from source_stage import verify_source_snapshot
    from streaming_lid.throughput import (
        EXPECTED_FRESH_PROCESSES,
        EXPECTED_MEASURED_SWEEPS,
        EXPECTED_WARMUP_SWEEPS,
        THROUGHPUT_METRIC,
        THROUGHPUT_SCHEMA_VERSION,
        THROUGHPUT_SCOPE,
        aggregate_processes,
        canonical_json_sha256,
    )

    if args.processes != EXPECTED_FRESH_PROCESSES:
        raise ValueError(f"controlled protocol requires {EXPECTED_FRESH_PROCESSES} processes")
    if args.warmups != EXPECTED_WARMUP_SWEEPS:
        raise ValueError(f"controlled protocol requires {EXPECTED_WARMUP_SWEEPS} warm-ups")
    if args.sweeps != EXPECTED_MEASURED_SWEEPS:
        raise ValueError(f"controlled protocol requires {EXPECTED_MEASURED_SWEEPS} sweeps")
    if args.threads != 6:
        raise ValueError("controlled protocol requires six intra-op threads")

    workspace = Path(_SOURCE_STAGE_CONTEXT["workspace_root"])
    manifest_path = (workspace / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    checkpoint_path = (workspace / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.resolve()
    results_dir = (workspace / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir.resolve()
    summary_path = results_dir / "summary.json"
    affinity = _derive_six_core_affinity()
    system = _system_diagnostics(affinity)
    input_stage, input_manifest, base_summary_payload = _capture_input_stage(
        workspace=workspace,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
    )

    worker_directory = Path(tempfile.mkdtemp(prefix="streaming-lid-throughput-"))
    rows: list[dict[str, Any]] = []
    try:
        for process_index in range(args.processes):
            output = worker_directory / f"process-{process_index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--input-stage",
                str(input_stage),
                "--input-stage-sha256",
                input_manifest["input_stage_root_sha256"],
                "--worker-output",
                str(output),
                "--process-index",
                str(process_index),
                "--affinity",
                ",".join(str(value) for value in affinity),
                "--threads",
                str(args.threads),
                "--warmups",
                str(args.warmups),
                "--sweeps",
                str(args.sweeps),
            ]
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"throughput worker {process_index} failed with code "
                    f"{completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
                )
            row = _read_json_bytes(
                output.read_bytes(), label=f"worker {process_index} output"
            )
            rows.append(row)
            print(
                f"completed throughput process {process_index + 1}/{args.processes}",
                flush=True,
            )
    finally:
        shutil.rmtree(worker_directory)

    aggregate = aggregate_processes(rows, require_complete_protocol=True)
    binding = rows[0]["binding"]
    protocol = {
        "fresh_processes": args.processes,
        "process_launch": "sequential",
        "full_corpus_warmups_per_process": args.warmups,
        "measured_sweeps_per_process": args.sweeps,
        "outer_replication_unit": "fresh_process",
        "pooled_sweeps_as_independent_runs": False,
    }
    benchmark_identity = {
        "schema_version": THROUGHPUT_SCHEMA_VERSION,
        "metric": THROUGHPUT_METRIC,
        "scope": THROUGHPUT_SCOPE,
        "binding": binding,
        "protocol": protocol,
        "environment_class": system["environment_class"],
    }
    benchmark_id = f"lidthroughput-{canonical_json_sha256(benchmark_identity)}"
    result = {
        "schema_version": THROUGHPUT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "metric": THROUGHPUT_METRIC,
        "scope": THROUGHPUT_SCOPE,
        "includes": [
            "whole_waveform_log_mel_frontend",
            "python_chunk_scheduler",
            "tensor_allocation",
            "uncached_streaming_overlap_recomputation",
            "student_forward",
        ],
        "excludes": [
            "wav_io",
            "checkpoint_load",
            "offline_teacher",
            "accuracy_scoring",
            "router",
            "incremental_waveform_ingress",
        ],
        "teacher_in_timed_region": False,
        "benchmark_identity": benchmark_identity,
        "input": {
            key: input_manifest[key]
            for key in (
                "input_stage_root_sha256",
                "source_manifest_file_sha256",
                "checkpoint_sha256",
                "base_summary_sha256",
                "n_clips",
                "sample_rate",
                "total_samples",
                "total_audio_seconds",
                "ordered_clip_ids_sha256",
                "ordered_wav_sha256",
                "clips",
            )
        },
        "identity": binding,
        "system": {
            **system,
            "load_average_after": list(os.getloadavg()),
        },
        "threading": binding["threading"],
        "protocol": protocol,
        "processes": rows,
        "aggregate": aggregate,
        "source_stage_verified_before_import": True,
        "source_stage_verified_before_publication": True,
        "input_stage_verified_before_workers": True,
        "input_stage_verified_before_publication": True,
        "publication": {
            "content_addressed": True,
            "older_results_overwritten": False,
        },
    }
    result_payload = _canonical_json_bytes(result) + b"\n"
    result_sha256 = _sha256(result_payload)
    result_relative = Path(RESULT_DIRECTORY_NAME) / f"{result_sha256}.json"
    result_path = results_dir / result_relative
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_result = result_path.parent / f".{result_sha256}.tmp"
    temporary_result.write_bytes(result_payload)
    _read_json_bytes(temporary_result.read_bytes(), label="temporary throughput result")

    verify_source_snapshot(
        _SOURCE_STAGE_CONTEXT["stage_root"],
        expected_root_sha256=_SOURCE_STAGE_CONTEXT["snapshot_root_sha256"],
    )
    _verify_input_stage(
        input_stage,
        expected_root_sha256=input_manifest["input_stage_root_sha256"],
    )
    if summary_path.read_bytes() != base_summary_payload:
        raise RuntimeError("summary changed during throughput benchmark")
    if _file_sha256(checkpoint_path) != input_manifest["checkpoint_sha256"]:
        raise RuntimeError("live checkpoint changed during throughput benchmark")
    if result_path.exists():
        if result_path.read_bytes() != result_payload:
            raise RuntimeError("content-addressed throughput result collision")
        temporary_result.unlink()
    else:
        os.replace(temporary_result, result_path)

    summary = _read_json_bytes(base_summary_payload, label=str(summary_path))
    controlled_summary = {
        "schema_version": THROUGHPUT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "metric": THROUGHPUT_METRIC,
        "result_path": result_relative.as_posix(),
        "result_sha256": result_sha256,
        "fresh_processes": aggregate["n_processes"],
        "measured_sweeps_per_process": args.sweeps,
        "process_median_wall_rtfs": aggregate["process_median_wall_rtfs"],
        "median_wall_rtf": aggregate["median_wall_rtf"],
        "q1_wall_rtf": aggregate["q1_wall_rtf"],
        "q3_wall_rtf": aggregate["q3_wall_rtf"],
        "min_wall_rtf": aggregate["min_wall_rtf"],
        "max_wall_rtf": aggregate["max_wall_rtf"],
        "max_min_ratio": aggregate["max_min_ratio"],
        "throughput_stability_passed": aggregate["throughput_stability_passed"],
        "x_realtime": aggregate["x_realtime"],
        "environment_class": system["environment_class"],
    }
    summary["controlled_throughput"] = controlled_summary
    summary_payload = json.dumps(summary, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_bytes(summary_payload)
    _read_json_bytes(temporary_summary.read_bytes(), label="updated summary")
    verify_source_snapshot(
        _SOURCE_STAGE_CONTEXT["stage_root"],
        expected_root_sha256=_SOURCE_STAGE_CONTEXT["snapshot_root_sha256"],
    )
    _verify_input_stage(
        input_stage,
        expected_root_sha256=input_manifest["input_stage_root_sha256"],
    )
    if summary_path.read_bytes() != base_summary_payload:
        raise RuntimeError("summary changed before controlled result publication")
    os.replace(temporary_summary, summary_path)

    if aggregate["throughput_stability_passed"]:
        message = (
            f"controlled median RTF={aggregate['median_wall_rtf']:.6f}; "
            f"x-real-time={aggregate['x_realtime']:.2f}"
        )
    else:
        message = (
            f"controlled process-median RTF range="
            f"{aggregate['min_wall_rtf']:.6f}--{aggregate['max_wall_rtf']:.6f}; "
            "scalar speed headline suppressed"
        )
    print(f"{message}; wrote {result_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/student.pt")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--processes", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-stage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--input-stage-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--process-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--affinity", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("processes", "warmups", "sweeps", "threads"):
        value = getattr(args, name)
        if not isinstance(value, int) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if args.worker:
        required = (
            "input_stage",
            "input_stage_sha256",
            "worker_output",
            "process_index",
            "affinity",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"worker arguments are missing: {missing}")
        if args.process_index < 0:
            parser.error("--process-index must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker(args)
    else:
        _coordinator(args)


if __name__ == "__main__":
    main()
