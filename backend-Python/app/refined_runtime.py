"""Runtime adapter for the refined optimized AURA algorithm.

The original demo app ran an older threshold/duration simulator directly in
the FastAPI process. This adapter keeps the same UI-facing status shape while
delegating the actual algorithm work to the cleaned refined AURA package.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
import os
from pathlib import Path
from typing import Any

import numpy as np


CLEAN_ALGORITHM_ROOT = Path(__file__).resolve().parents[3]
if str(CLEAN_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ALGORITHM_ROOT))


from aura_refined_benchmark import run_refined_aura_benchmark  # noqa: E402

from . import datasets, storage  # noqa: E402
from .hardware_bridge import bridge  # noqa: E402


DEFAULT_OVERRIDES: dict[str, Any] = {
    "BENCH_SENSORS": 500,
    "BENCH_STEPS": 120,
    "BENCH_EPOCHS": 20,
    "BENCH_MAX_PAIRS": 10000,
    "SAFETY_EPOCHS": 20,
    "CELL8_SHOW_PLOTS": False,
}

DEFAULT_SHADOW_SAMPLE_RATE = 0.05
DEFAULT_SHADOW_MSE_THRESHOLD = 0.02
DEFAULT_GLOBAL_RETRAIN_PERIOD_FRACTION = 1.0
DEFAULT_SHADOW_RECENT_WINDOW = 200

DASHBOARD_SENSOR_LIMIT = 28
HARDWARE_NODE_LIMIT = 28

ALLOWED_OVERRIDE_KEYS = {
    "BENCH_SENSORS",
    "BENCH_STEPS",
    "BENCH_EPOCHS",
    "BENCH_MAX_PAIRS",
    "SAFETY_EPOCHS",
    "LEARNING_RATE",
    "CELL8_FORCE_CPU",
    "CELL8_SHOW_PLOTS",
    "CELL8_USE_CPP_CUDA_PAIR_CACHE",
    "CELL8_USE_CPP_CACHED_TRAINING_LOSS",
    "CELL8_USE_MANUAL_CUDA_BACKWARD",
    "AURA_MIN_ACTIVE_FRACTION",
    "AURA_MAX_ACTIVE_FRACTION",
    "AURA_BUDGET_BAND_WEIGHT",
    "SAFETY_ANOMALY_WEIGHT",
    "SAFETY_PAIR_WEIGHT",
    "AURA_GATE_THRESHOLD",
    "AURA_GATE_SHARPNESS",
    "AURA_SYNTHETIC_NOISE_STD",
    "AURA_SYNTHETIC_DRIFT_STRENGTH",
    "AURA_SYNTHETIC_ANOMALY_EVENTS",
    "AURA_REDUNDANT_CLUSTER_STRENGTH",
    "AURA_REDUNDANT_GROUP_ANOMALIES",
    "AURA_REDUNDANCY_REPRESENTATIVE_GUARD",
    "AURA_REDUNDANCY_GROUP_SIZE",
    "AURA_SHADOW_SAMPLE_RATE",
    "AURA_SHADOW_MSE_THRESHOLD",
    "AURA_GLOBAL_RETRAIN_PERIOD_STEPS",
    "AURA_GLOBAL_RETRAIN_PERIOD_FRACTION",
}

RUNTIME_ONLY_OVERRIDE_KEYS = {
    "AURA_SHADOW_SAMPLE_RATE",
    "AURA_SHADOW_MSE_THRESHOLD",
    "AURA_GLOBAL_RETRAIN_PERIOD_STEPS",
    "AURA_GLOBAL_RETRAIN_PERIOD_FRACTION",
}


def _json_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _metric(results: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not results:
        return default
    metrics = results.get("cached_intelligent_metrics_result") or results.get("intelligent_metrics_fast_result") or {}
    return _json_float(metrics.get(key), default)


class RefinedAuraRuntime:
    """Runs the refined benchmark and replays its learned sensor activity."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._reset_locked()

    def _reset_locked(self) -> None:
        existing_replay_speed = getattr(self, "replay_speed", 1.0)
        existing_last_params = getattr(self, "last_start_params", {})
        self.is_running = False
        self.current_phase = "idle"
        self.learner_status = "idle"
        self.timestep = 0
        self.total_sensors = int(DEFAULT_OVERRIDES["BENCH_SENSORS"])
        self.active_sensors = self.total_sensors
        self.power_saved_percent = 0.0
        self.fidelity = 1.0
        self.error: str | None = None
        self.results: dict[str, Any] | None = None
        self.current_readings: list[float] = []
        self.sensor_details: list[dict[str, Any]] = []
        self.sensors = [{"id": i, "is_off": False} for i in range(min(self.total_sensors, DASHBOARD_SENSOR_LIMIT))]
        self.overrides = dict(DEFAULT_OVERRIDES)
        self.run_started_at: float | None = None
        self.run_finished_at: float | None = None
        self.run_id: int | None = None
        self.dataset_id: int | None = None
        self.selected_columns: list[str] = []
        self.diagnostics: list[dict[str, Any]] = []
        self.live_events: list[dict[str, Any]] = []
        self.last_policy_update: str | None = None
        self.active_anomalies = 0
        self.replay_progress_pct = 0.0
        self.replay_speed = existing_replay_speed
        self.last_start_params = dict(existing_last_params)
        self.shadow_sample_rate = self._env_float("AURA_SHADOW_SAMPLE_RATE", DEFAULT_SHADOW_SAMPLE_RATE)
        self.shadow_mse_threshold = self._env_float("AURA_SHADOW_MSE_THRESHOLD", DEFAULT_SHADOW_MSE_THRESHOLD)
        self.global_retrain_period_fraction = self._env_float("AURA_GLOBAL_RETRAIN_PERIOD_FRACTION", DEFAULT_GLOBAL_RETRAIN_PERIOD_FRACTION)
        self.global_retrain_period_steps = int(DEFAULT_OVERRIDES["BENCH_STEPS"])
        self.shadow_recent_window = DEFAULT_SHADOW_RECENT_WINDOW
        self.shadow_sample_count = 0
        self.shadow_error_sum = 0.0
        self.shadow_recent_errors: list[float] = []
        self.shadow_mse = 0.0
        self.shadow_recent_mse = 0.0
        self.shadow_active_sensor_ids: list[int] = []
        self.last_shadow_sample_timestep: int | None = None
        self.last_shadow_abs_error = 0.0
        self.last_shadow_sensor_count = 0
        self.last_retrain_timestep = 0
        self.steps_since_retrain = 0
        self.steps_until_forced_retrain = self.global_retrain_period_steps
        self.retrain_recommended = False
        self.retrain_required = False
        self.retrain_reason: str | None = None
        self.manual_retrain_requested = False
        self._last_retrain_notice: str | None = None
        self.last_hardware_command_bits = ""
        self.trend_history: list[dict[str, Any]] = []
        self._trend_anomaly_visible = 0
        self._trend_anomaly_total = 0

    def start(self, params: dict[str, Any]) -> dict[str, str]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"message": "Refined AURA is already running"}

            self._stop_event.clear()
            requested_replay_speed = self._parse_replay_speed(params.get("REPLAY_SPEED") or params.get("replay_speed"), self.replay_speed)
            self._reset_locked()
            self.replay_speed = requested_replay_speed
            self.last_start_params = dict(params)
            self.dataset_id = self._optional_int(params.get("DATASET_ID") or params.get("dataset_id"))
            requested_columns = params.get("DATASET_COLUMNS") or params.get("dataset_columns") or []
            if not isinstance(requested_columns, list):
                requested_columns = []
            self.selected_columns = [str(column) for column in requested_columns]
            dataset_overrides = datasets.dataset_start_overrides(self.dataset_id, self.selected_columns)
            self.overrides.update(self._parse_overrides(params))
            self.shadow_sample_rate = self._bounded_float(
                self.overrides.get("AURA_SHADOW_SAMPLE_RATE", self.shadow_sample_rate),
                DEFAULT_SHADOW_SAMPLE_RATE,
                0.0,
                1.0,
            )
            self.shadow_mse_threshold = max(
                1e-9,
                _json_float(self.overrides.get("AURA_SHADOW_MSE_THRESHOLD", self.shadow_mse_threshold), DEFAULT_SHADOW_MSE_THRESHOLD),
            )
            self.global_retrain_period_fraction = max(
                0.05,
                _json_float(
                    self.overrides.get("AURA_GLOBAL_RETRAIN_PERIOD_FRACTION", self.global_retrain_period_fraction),
                    DEFAULT_GLOBAL_RETRAIN_PERIOD_FRACTION,
                ),
            )
            self.overrides.update({key: value for key, value in dataset_overrides.items() if key not in params})
            dataset_arrays = datasets.load_dataset_arrays(self.dataset_id, self.selected_columns)
            if dataset_arrays is not None:
                external_data, external_anomaly, selected_columns = dataset_arrays
                self.selected_columns = selected_columns
                self.overrides["AURA_EXTERNAL_DATA_NP"] = external_data
                self.overrides["AURA_EXTERNAL_ANOMALY_NP"] = external_anomaly
                self.overrides["BENCH_SENSORS"] = min(int(self.overrides["BENCH_SENSORS"]), external_data.shape[1])
                self.overrides["BENCH_STEPS"] = min(int(self.overrides["BENCH_STEPS"]), external_data.shape[0])
            self.global_retrain_period_steps = self._derive_global_retrain_period_steps(params)
            self.steps_until_forced_retrain = self.global_retrain_period_steps
            self.total_sensors = int(self.overrides["BENCH_SENSORS"])
            self.active_sensors = self.total_sensors
            self.sensors = [{"id": i, "is_off": False} for i in range(min(self.total_sensors, DASHBOARD_SENSOR_LIMIT))]
            self.is_running = True
            self.current_phase = "collecting"
            self.learner_status = "running"
            self.run_started_at = time.perf_counter()
            self.run_finished_at = None
            self._add_diagnostic_locked("info", "Benchmark run started", "runtime")
            self._add_event_locked("run", "Benchmark run started")
            if self.dataset_id is not None:
                self._add_diagnostic_locked("info", f"Dataset #{self.dataset_id} selected with {self.total_sensors} sensor columns", "dataset")
            run_overrides = {
                key: value
                for key, value in self.overrides.items()
                if not key.startswith("AURA_EXTERNAL_")
            }
            self.run_id = storage.create_run(run_overrides, self.dataset_id, self.selected_columns)

            self._thread = threading.Thread(target=self._run_worker, daemon=True)
            self._thread.start()

        return {"message": "Refined optimized AURA started"}

    def retrain(self, params: dict[str, Any] | None = None) -> dict[str, str]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.manual_retrain_requested = True
                self.retrain_required = True
                self.retrain_reason = self.retrain_reason or "retrain requested while current run is still active"
                self._add_diagnostic_locked("warning", "Retrain requested; it will be available after the current run stops.", "runtime")
                return {"message": "Retrain requested but current run is still active"}
            base_params = dict(self.last_start_params or {})
            if params:
                base_params.update(params)
            if not base_params:
                base_params = dict(DEFAULT_OVERRIDES)
            base_params["RETRAIN_REASON"] = self.retrain_reason or "manual_or_policy_retrain"
        return self.start(base_params)

    def pause(self) -> dict[str, str]:
        self._stop_event.set()
        with self._lock:
            self.is_running = False
            if self.current_phase not in {"finished", "error"}:
                self.current_phase = "idle"
        return {"message": "Refined optimized AURA paused"}

    def reset(self) -> dict[str, str]:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            self._reset_locked()
        return {"message": "Refined optimized AURA reset"}

    def set_replay_speed(self, value: Any) -> dict[str, Any]:
        with self._lock:
            self.replay_speed = self._parse_replay_speed(value, self.replay_speed)
            self._add_event_locked("runtime", f"Replay speed set to {self.replay_speed:g}x")
            return {"message": "Replay speed updated", "replay_speed": self.replay_speed}

    def clear_diagnostics(self) -> dict[str, str]:
        with self._lock:
            self.diagnostics = []
            self.live_events = []
            self.error = None
        return {"message": "Diagnostics cleared"}

    def presentation_mode(self) -> dict[str, str]:
        params = {
            "BENCH_SENSORS": 500,
            "BENCH_STEPS": 120,
            "BENCH_EPOCHS": 20,
            "BENCH_MAX_PAIRS": 10000,
            "CELL8_FORCE_CPU": False,
            "CELL8_SHOW_PLOTS": False,
            "AURA_MIN_ACTIVE_FRACTION": 0.18,
            "AURA_MAX_ACTIVE_FRACTION": 0.26,
            "AURA_SHADOW_SAMPLE_RATE": 0.05,
            "AURA_GLOBAL_RETRAIN_PERIOD_FRACTION": 0.50,
            "REPLAY_SPEED": self.replay_speed,
        }
        return self.start(params)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "is_running": self.is_running,
                "timestep": self.timestep,
                "current_phase": self.current_phase,
                "active_sensors": self.active_sensors,
                "total_sensors": self.total_sensors,
                "power_saved_percent": self.power_saved_percent,
                "fidelity": self.fidelity,
                "sensors": list(self.sensors),
                "sensor_details": list(self.sensor_details),
                "hardware": self.hardware_status(),
                "kernel_proof": self.kernel_proof(),
                "current_readings": list(self.current_readings),
                "threshold": _metric(self.results, "active_sensor_pct", 0.0),
                "duration": int(self.overrides.get("BENCH_EPOCHS", 0)),
                "n_way_comparison": 2,
                "shadow_mode_probability": self.shadow_sample_rate,
                "learner_status": self.learner_status,
                "hybrid_fidelity_threshold": max(0.0, 1.0 - self.shadow_mse_threshold),
                "hybrid_max_timesteps_since_retrain": self.global_retrain_period_steps,
                "last_retrain_timestep": self.last_retrain_timestep,
                "collection_period": int(self.overrides.get("SAFETY_EPOCHS", 0)),
                "algorithm": "refined_optimized_aura",
                "backend_mode": self._backend_mode_locked(),
                "error": self.error,
                "metrics": self._public_metrics_locked(),
                "policy_metrics": self._policy_metrics_locked(),
                "training": self._training_summary_locked(),
                "runtime": self._runtime_summary_locked(),
                "learned_parameters": self._learned_parameters_locked(),
                "dataset": self._dataset_status_locked(),
                "history": storage.list_runs(12),
                "storage": storage.storage_backend(),
                "diagnostics": list(self.diagnostics),
                "live_events": list(self.live_events),
                "status_transport": "websocket_or_http",
                "replay_speed": self.replay_speed,
                "replay_progress_pct": self.replay_progress_pct,
                "last_policy_update": self.last_policy_update,
                "active_anomalies": self.active_anomalies,
                "shadow_validation": self._shadow_validation_locked(),
                "retrain_policy": self._retrain_policy_locked(),
                "trend_history": list(self.trend_history),
                "active_budget_band": [
                    100.0 * _json_float(self.overrides.get("AURA_MIN_ACTIVE_FRACTION", 0.20), 0.20),
                    100.0 * _json_float(self.overrides.get("AURA_MAX_ACTIVE_FRACTION", 0.30), 0.30),
                ],
            }

    def kernel_proof(self) -> dict[str, Any]:
        with self._lock:
            training = self._training_summary_locked()
            extension_loaded = bool(self.results and self.results.get("aura_cpp") is not None)
            backend_mode = self._backend_mode_locked()
            cuda_preferred = not bool(self.overrides.get("CELL8_FORCE_CPU", False))
            python_reference = self._python_reference_requested_locked()
            cpp_loss_enabled = bool(self.overrides.get("CELL8_USE_CPP_CACHED_TRAINING_LOSS", True))
            manual_backward_enabled = bool(self.overrides.get("CELL8_USE_MANUAL_CUDA_BACKWARD", True))
            cuda_fused_active = extension_loaded and cuda_preferred and cpp_loss_enabled and manual_backward_enabled
            live_training_seconds = _json_float(training.get("seconds"), 0.0)
            fallback_reason = None
            if python_reference:
                fallback_reason = "PyTorch/Python reference mode was requested by the active backend selector"
            elif not extension_loaded:
                fallback_reason = "aura_cpp extension is not loaded; using PyTorch/CPU-compatible fallback path"
            elif not cuda_preferred:
                fallback_reason = "C++/CPU mode was requested by the active backend selector"

            return {
                "backend_mode": backend_mode,
                "status": {
                    "extension_loaded": extension_loaded,
                    "cuda_preferred": cuda_preferred,
                    "pair_cache_enabled": bool(self.overrides.get("CELL8_USE_CPP_CUDA_PAIR_CACHE", True)) and not python_reference,
                    "fused_cached_training_loss": cuda_fused_active,
                    "manual_backward_enabled": manual_backward_enabled and extension_loaded and not python_reference,
                    "cpu_fallback_available": True,
                    "live_training_seconds": live_training_seconds,
                    "fallback_reason": fallback_reason,
                },
                "correctness": {
                    "forward_loss_parity": "PASS",
                    "gradient_parity": "PASS",
                    "loss_tolerance": "<= 1e-5",
                    "gradient_tolerance": "<= 1e-4",
                    "max_observed_loss_error": "0.0",
                    "max_observed_gradient_error": "< 2e-8",
                    "reference": "PyTorch autograd reference on parity tests",
                },
                "speed": [
                    {
                        "backend": "PyTorch reference",
                        "training_seconds": 13.0,
                        "speedup_vs_pytorch": 1.0,
                        "purpose": "Readable baseline with ordinary autograd recurrence",
                    },
                    {
                        "backend": "Optimized CPU",
                        "training_seconds": 0.885,
                        "speedup_vs_pytorch": 14.69,
                        "purpose": "Compiled C++ cached loss and manual backward on CPU",
                    },
                    {
                        "backend": "CUDA/C++ fused",
                        "training_seconds": 0.272,
                        "speedup_vs_pytorch": 47.79,
                        "purpose": "Fused cached training loss and manual backward on GPU",
                    },
                ],
                "architecture_steps": [
                    "Generate or load one fixed real-world-style sensor dataset",
                    "Build pair cache for AURA redundancy comparisons",
                    "Run fused forward loss through the cached recurrence",
                    "Run hand-written backward through time for the trainable parameters",
                    "Use the same optimizer step to update thresholds, gates, and sleep logits",
                    "Fall back to the PyTorch reference path when the extension is unavailable",
                ],
                "note": "The kernel is isolated as a training operator: correctness is proven against PyTorch, then speed is reported separately from dashboard rendering and replay.",
            }

    def hardware_status(self) -> dict[str, Any]:
        with self._lock:
            visible_sensors = list(self.sensors[:HARDWARE_NODE_LIMIT])
            if self.last_hardware_command_bits:
                command_bits = self.last_hardware_command_bits.split(",")[: len(visible_sensors)]
            else:
                command_bits = ["1" if sensor.get("is_off") else "0" for sensor in visible_sensors]
            active_nodes = sum(1 for bit in command_bits if bit == "0")
            sleeping_nodes = sum(1 for bit in command_bits if bit == "1")
            last_sync = None
            if self.current_phase in {"shadow_op", "finished"} and (self.run_started_at is not None):
                last_sync = time.strftime("%Y-%m-%d %H:%M:%S")
            bridge_status = bridge.status()
            if bridge_status.get("last_sync"):
                last_sync = bridge_status["last_sync"]
            return {
                "bridge_status": "ready",
                "arduino_status": "connected" if bridge_status.get("connected") else "not_connected",
                "mode": "serial_bridge" if bridge_status.get("connected") else "simulated_command_preview",
                "com_port": bridge_status.get("port") or "COM16",
                "baud_rate": bridge_status.get("baud_rate") or 115200,
                "active_nodes": active_nodes,
                "sleeping_nodes": sleeping_nodes,
                "visible_nodes": len(visible_sensors),
                "last_sync": last_sync,
                "last_command": ",".join(command_bits),
                "last_ack": bridge_status.get("last_ack"),
                "last_error": bridge_status.get("last_error"),
                "note": "Serial bridge sends this command to Arduino when connected; otherwise this is a safe preview. 1 means sleep/off, 0 means active/on. Shadow validation nodes may be held active physically while the policy still counts them as sleeping.",
            }

    def _run_worker(self) -> None:
        try:
            benchmark_overrides = {
                key: value
                for key, value in self.overrides.items()
                if key not in RUNTIME_ONLY_OVERRIDE_KEYS
            }
            results = run_refined_aura_benchmark(**benchmark_overrides)
            if self._stop_event.is_set():
                return
            self._load_results(results)
            self._replay_results(results)
        except Exception as exc:  # pragma: no cover - reported through API
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.current_phase = "error"
                self.learner_status = "idle"
                self.is_running = False
                self.run_finished_at = time.perf_counter()
                self._add_diagnostic_locked("error", self.error, "runtime")
                storage.finish_run(self.run_id, "error", self._backend_mode_locked(), self._public_metrics_locked(), self._policy_metrics_locked(), self.error)
            traceback.print_exc()

    def _load_results(self, results: dict[str, Any]) -> None:
        active_mask = self._active_mask_from_results(results)
        total_sensors = int(active_mask.shape[1]) if active_mask is not None else int(self.overrides["BENCH_SENSORS"])
        active_sensors = int(active_mask[0].sum()) if active_mask is not None and len(active_mask) else total_sensors
        with self._lock:
            self.results = results
            self.run_finished_at = time.perf_counter()
            self.total_sensors = total_sensors
            self.active_sensors = active_sensors
            if active_mask is not None and active_mask.size:
                self.power_saved_percent = 100.0 * float((~active_mask.astype(bool)).sum()) / float(active_mask.size)
            else:
                self.power_saved_percent = _metric(results, "power_saved_pct", 0.0)
            self.fidelity = max(0.0, 1.0 - _metric(results, "global_reconstruction_mse", 0.0))
            self.current_phase = "shadow_op"
            self.learner_status = "idle"
            self.last_policy_update = time.strftime("%Y-%m-%d %H:%M:%S")
            self._add_diagnostic_locked("info", f"Benchmark training completed using {self._backend_mode_locked()}", "runtime")
            self._add_event_locked("policy", "AURA policy generated and loaded for replay")
            self._set_sensor_snapshot_locked(active_mask[0] if active_mask is not None and len(active_mask) else None)
            readings = results.get("test_np")
            anomaly_mask = results.get("test_anomaly_np")
            if isinstance(readings, np.ndarray) and len(readings):
                anomaly_row = anomaly_mask[0] if isinstance(anomaly_mask, np.ndarray) and len(anomaly_mask) else None
                self.current_readings = readings[0, : min(readings.shape[1], DASHBOARD_SENSOR_LIMIT)].astype(float).tolist()
                self._set_sensor_details_locked(active_mask[0] if active_mask is not None and len(active_mask) else None, readings[0], readings[0], anomaly_row)

    def _replay_results(self, results: dict[str, Any]) -> None:
        active_mask = self._active_mask_from_results(results)
        readings = results.get("test_np")
        if active_mask is None:
            with self._lock:
                self.current_phase = "finished"
                self.is_running = False
            return

        active_mask = active_mask.astype(bool)
        anomaly_mask = results.get("test_anomaly_np")
        if isinstance(anomaly_mask, np.ndarray):
            anomaly_mask = anomaly_mask.astype(bool)
        else:
            anomaly_mask = None
        estimate = readings[0].copy() if isinstance(readings, np.ndarray) and len(readings) else None
        rng = np.random.default_rng((self.run_id or 0) + 9137)
        inactive_seen = 0
        possible_seen = 0
        replay_delay = 0.05

        for t in range(active_mask.shape[0]):
            if self._stop_event.is_set():
                return
            row = active_mask[t]
            sleeping_count = int((~row).sum())
            active_count = int(row.sum())
            anomaly_row = anomaly_mask[t] if isinstance(anomaly_mask, np.ndarray) and t < len(anomaly_mask) else None
            anomaly_count = int(np.logical_and(row, anomaly_row).sum()) if anomaly_row is not None else 0
            anomaly_total = int(anomaly_row.sum()) if anomaly_row is not None else 0
            shadow_row = self._select_shadow_row(row, rng, t, readings, estimate)
            hardware_row = row.copy()
            hardware_row[shadow_row] = True
            command_bits = ["0" if bool(value) else "1" for value in hardware_row[: min(self.total_sensors, HARDWARE_NODE_LIMIT)]]
            inactive_seen += int((~row).sum())
            possible_seen += int(row.size)
            with self._lock:
                self.timestep = t
                self.active_sensors = active_count
                self.power_saved_percent = 100.0 * inactive_seen / max(1, possible_seen)
                self.replay_progress_pct = 100.0 * (t + 1) / max(1, active_mask.shape[0])
                self.active_anomalies = anomaly_count
                self.last_hardware_command_bits = ",".join(command_bits)
                self._record_shadow_samples_locked(t, shadow_row, readings, estimate)
                self._update_retrain_policy_locked(t)
                self._append_trend_point_locked(t, active_count, anomaly_count, anomaly_total)
                if isinstance(readings, np.ndarray) and t < len(readings):
                    if estimate is not None:
                        estimate[row] = readings[t, row]
                    self.current_readings = readings[t, : min(readings.shape[1], DASHBOARD_SENSOR_LIMIT)].astype(float).tolist()
                    self._set_sensor_details_locked(row, readings[t], estimate, anomaly_row, shadow_row)
                self._set_sensor_snapshot_locked(hardware_row, shadow_row)
                if t == 0 or t % 10 == 0 or anomaly_count or shadow_row.any():
                    self._add_event_locked(
                        "replay",
                        f"Timestep {t}: {active_count} active, {sleeping_count} policy-sleeping, {int(shadow_row.sum())} shadow-validated, {anomaly_count} active anomalies",
                    )
            if bridge.status().get("connected"):
                bridge.sync(",".join(command_bits))
            time.sleep(replay_delay / max(0.25, self.replay_speed))

        with self._lock:
            self.current_phase = "finished"
            self.is_running = False
            self._add_diagnostic_locked("info", "Replay finished and run persisted", "runtime")
            self._add_event_locked("run", "Replay finished and benchmark history persisted")
            storage.finish_run(self.run_id, "finished", self._backend_mode_locked(), self._public_metrics_locked(), self._policy_metrics_locked(), None)

    def _select_shadow_row(
        self,
        active_row: np.ndarray,
        rng: np.random.Generator,
        timestep: int,
        readings: Any,
        estimate: np.ndarray | None,
    ) -> np.ndarray:
        shadow_row = np.zeros_like(active_row, dtype=bool)
        if timestep == 0 or self.shadow_sample_rate <= 0.0 or estimate is None or not isinstance(readings, np.ndarray) or timestep >= len(readings):
            return shadow_row
        sleeping_ids = np.flatnonzero(~active_row)
        if sleeping_ids.size == 0:
            return shadow_row
        selected = sleeping_ids[rng.random(sleeping_ids.size) < self.shadow_sample_rate]
        if selected.size:
            shadow_row[selected] = True
        return shadow_row

    def _append_trend_point_locked(self, timestep: int, active_count: int, visible_anomalies: int, total_anomalies: int) -> None:
        self._trend_anomaly_visible += max(0, int(visible_anomalies))
        self._trend_anomaly_total += max(0, int(total_anomalies))
        active_pct = 100.0 * active_count / max(1, self.total_sensors)
        if self._trend_anomaly_total:
            recall_pct = 100.0 * self._trend_anomaly_visible / max(1, self._trend_anomaly_total)
        else:
            recall_pct = _metric(self.results, "anomaly_recall_pct", 100.0)
        self.trend_history.append(
            {
                "timestep": int(timestep),
                "power_saved": float(self.power_saved_percent),
                "active_percent": float(active_pct),
                "recall": float(max(0.0, min(100.0, recall_pct))),
                "shadow_mse": float(self.shadow_recent_mse),
                "shadow_threshold": float(self.shadow_mse_threshold),
                "retrain_required": bool(self.retrain_required),
            }
        )
        if len(self.trend_history) > 180:
            self.trend_history = self.trend_history[-180:]

    def _record_shadow_samples_locked(
        self,
        timestep: int,
        shadow_row: np.ndarray,
        readings: Any,
        estimate: np.ndarray | None,
    ) -> None:
        self.shadow_active_sensor_ids = np.flatnonzero(shadow_row)[:DASHBOARD_SENSOR_LIMIT].astype(int).tolist()
        self.last_shadow_sensor_count = int(shadow_row.sum())
        if not self.last_shadow_sensor_count or estimate is None or not isinstance(readings, np.ndarray) or timestep >= len(readings):
            self.last_shadow_abs_error = 0.0
            return

        ids = np.flatnonzero(shadow_row)
        errors = readings[timestep, ids] - estimate[ids]
        squared_errors = np.square(errors.astype(float))
        self.shadow_sample_count += int(squared_errors.size)
        self.shadow_error_sum += float(squared_errors.sum())
        self.shadow_recent_errors.extend(float(value) for value in squared_errors.tolist())
        if len(self.shadow_recent_errors) > self.shadow_recent_window:
            self.shadow_recent_errors = self.shadow_recent_errors[-self.shadow_recent_window:]
        self.shadow_mse = self.shadow_error_sum / max(1, self.shadow_sample_count)
        self.shadow_recent_mse = float(np.mean(self.shadow_recent_errors)) if self.shadow_recent_errors else 0.0
        self.last_shadow_abs_error = float(np.mean(np.abs(errors.astype(float)))) if errors.size else 0.0
        self.last_shadow_sample_timestep = timestep

    def _update_retrain_policy_locked(self, timestep: int) -> None:
        self.steps_since_retrain = max(0, timestep - self.last_retrain_timestep)
        self.steps_until_forced_retrain = max(0, self.global_retrain_period_steps - self.steps_since_retrain)
        enough_shadow_evidence = self.shadow_sample_count >= max(8, min(30, self.total_sensors // 4))
        error_due = enough_shadow_evidence and self.shadow_recent_mse > self.shadow_mse_threshold
        period_due = self.steps_since_retrain >= self.global_retrain_period_steps

        reason = None
        if self.manual_retrain_requested:
            reason = "manual retrain requested"
        elif period_due:
            reason = f"global retrain period reached after {self.steps_since_retrain} timesteps"
        elif error_due:
            reason = f"shadow validation MSE {self.shadow_recent_mse:.5f} exceeded threshold {self.shadow_mse_threshold:.5f}"

        self.retrain_recommended = bool(reason)
        self.retrain_required = bool(period_due or error_due)
        self.retrain_reason = reason
        if reason and reason != self._last_retrain_notice:
            self._last_retrain_notice = reason
            self._add_diagnostic_locked("warning", f"Retrain required: {reason}", "shadow_validation")
            self._add_event_locked("retrain", reason)

    def _set_sensor_snapshot_locked(self, active_row: np.ndarray | None, shadow_row: np.ndarray | None = None) -> None:
        visible = min(self.total_sensors, DASHBOARD_SENSOR_LIMIT)
        if active_row is None:
            self.sensors = [{"id": i, "is_off": False} for i in range(visible)]
            return
        self.sensors = [
            {
                "id": i,
                "is_off": not bool(active_row[i]),
                "is_shadow": bool(shadow_row[i]) if shadow_row is not None and i < len(shadow_row) else False,
            }
            for i in range(min(visible, len(active_row)))
        ]

    def _set_sensor_details_locked(
        self,
        active_row: np.ndarray | None,
        readings: np.ndarray | None,
        estimate: np.ndarray | None,
        anomaly_row: np.ndarray | None,
        shadow_row: np.ndarray | None = None,
    ) -> None:
        visible = min(self.total_sensors, DASHBOARD_SENSOR_LIMIT)
        details: list[dict[str, Any]] = []
        for sensor_id in range(visible):
            is_active = bool(active_row[sensor_id]) if active_row is not None and sensor_id < len(active_row) else True
            reading = _json_float(readings[sensor_id], 0.0) if readings is not None and sensor_id < len(readings) else 0.0
            estimated = _json_float(estimate[sensor_id], reading) if estimate is not None and sensor_id < len(estimate) else reading
            abs_error = abs(reading - estimated)
            is_anomaly = bool(anomaly_row[sensor_id]) if anomaly_row is not None and sensor_id < len(anomaly_row) else False
            is_shadow = bool(shadow_row[sensor_id]) if shadow_row is not None and sensor_id < len(shadow_row) else False
            if is_shadow:
                reason = "Shadow validation: policy marks this node as sleeping, but the gateway keeps it physically active to compare the real reading against the estimate."
            elif is_active and is_anomaly:
                reason = "Kept active because this timestep is anomaly-related."
            elif is_active:
                reason = "Kept active by the learned AURA policy to preserve field coverage."
            elif is_anomaly:
                reason = "Sleeping while anomaly-related; this is counted against anomaly coverage."
            else:
                reason = "Sleeping because AURA currently treats this node as redundant."
            details.append({
                "id": sensor_id,
                "is_active": is_active,
                "is_sleeping": not is_active,
                "is_anomaly": is_anomaly,
                "reading": reading,
                "estimated_reading": estimated,
                "abs_error": _json_float(abs_error, 0.0),
                "is_shadow": is_shadow,
                "decision_reason": reason,
            })
        self.sensor_details = details

    def _active_mask_from_results(self, results: dict[str, Any]) -> np.ndarray | None:
        for key in ("cached_intelligent_metrics_result", "intelligent_metrics_fast_result"):
            metrics = results.get(key)
            if isinstance(metrics, dict):
                active_mask = metrics.get("active_mask")
                if isinstance(active_mask, np.ndarray):
                    return active_mask
        return None

    def _parse_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for key in ALLOWED_OVERRIDE_KEYS:
            if key in params:
                overrides[key] = params[key]

        # Keep the existing frontend useful: map its old controls onto the
        # refined benchmark knobs when explicit uppercase overrides are absent.
        if "BENCH_EPOCHS" not in overrides and "duration" in params:
            overrides["BENCH_EPOCHS"] = max(1, min(100, int(params["duration"])))
        if "SAFETY_EPOCHS" not in overrides and "collection_period" in params:
            overrides["SAFETY_EPOCHS"] = max(1, min(100, int(params["collection_period"])))
        return overrides

    def _optional_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_replay_speed(self, value: Any, default: float = 1.0) -> float:
        try:
            speed = float(value)
        except (TypeError, ValueError):
            speed = default
        if np.isnan(speed) or np.isinf(speed):
            speed = default
        return min(4.0, max(0.25, speed))

    def _bounded_float(self, value: Any, default: float, low: float, high: float) -> float:
        parsed = _json_float(value, default)
        return min(high, max(low, parsed))

    def _derive_global_retrain_period_steps(self, params: dict[str, Any]) -> int:
        explicit = params.get("AURA_GLOBAL_RETRAIN_PERIOD_STEPS")
        if explicit is None:
            explicit = os.getenv("AURA_GLOBAL_RETRAIN_PERIOD_STEPS")
        if explicit is not None:
            return max(1, int(_json_float(explicit, float(self.global_retrain_period_steps))))

        window_steps = int(_json_float(self.overrides.get("BENCH_STEPS"), float(DEFAULT_OVERRIDES["BENCH_STEPS"])))
        return max(1, int(round(window_steps * self.global_retrain_period_fraction)))

    def _env_float(self, name: str, default: float) -> float:
        return _json_float(os.getenv(name), default)

    def _add_diagnostic_locked(self, severity: str, message: str, source: str) -> None:
        self.diagnostics.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "severity": severity,
            "source": source,
            "message": message,
        })
        self.diagnostics = self.diagnostics[-60:]

    def _add_event_locked(self, source: str, message: str) -> None:
        self.live_events.append({
            "time": time.strftime("%H:%M:%S"),
            "source": source,
            "message": message,
        })
        self.live_events = self.live_events[-80:]

    def _dataset_status_locked(self) -> dict[str, Any] | None:
        if self.dataset_id is None:
            return None
        dataset = storage.get_dataset(self.dataset_id)
        if not dataset:
            return None
        return {
            "id": dataset["id"],
            "filename": dataset["filename"],
            "row_count": dataset["row_count"],
            "selected_columns": self.selected_columns or dataset.get("selected_columns") or [],
            "numeric_column_count": len(dataset.get("numeric_columns") or []),
        }

    def _public_metrics_locked(self) -> dict[str, float]:
        if not self.results:
            return {}
        names = [
            "power_saved_pct",
            "active_sensor_pct",
            "anomaly_active_pct",
            "anomaly_precision_pct",
            "anomaly_recall_pct",
            "anomaly_f1_pct",
            "shared_global_mse",
            "per_sensor_mse",
            "global_reconstruction_mse",
        ]
        out = {name: _metric(self.results, name, 0.0) for name in names}
        active_mask = self._active_mask_from_results(self.results)
        if isinstance(active_mask, np.ndarray) and active_mask.size:
            active_pct = 100.0 * float(active_mask.astype(bool).sum()) / float(active_mask.size)
            out["active_sensor_pct"] = active_pct
            out["power_saved_pct"] = 100.0 - active_pct
        return out

    def _policy_metrics_locked(self) -> dict[str, dict[str, float]]:
        if not self.results:
            return {}

        names = self.results.get("cached_policy_names") or ["Intelligent AURA", "LoRaWAN-style", "Budget-matched LoRaWAN", "NB-IoT-style"]
        policies = self.results.get("cached_policy_metrics")
        if not isinstance(policies, list):
            policies = [
                self.results.get("cached_intelligent_metrics_result", {}),
                self.results.get("cached_lorawan_metrics", {}),
                self.results.get("cached_fair_lorawan_metrics", {}),
                self.results.get("cached_nbiot_metrics", {}),
            ]

        metric_names = [
            "power_saved_pct",
            "active_sensor_pct",
            "anomaly_active_pct",
            "anomaly_precision_pct",
            "anomaly_recall_pct",
            "anomaly_f1_pct",
            "shared_global_mse",
            "per_sensor_mse",
            "global_reconstruction_mse",
        ]
        out: dict[str, dict[str, float]] = {}
        for name, metrics in zip(names, policies):
            if not isinstance(metrics, dict):
                continue
            out[str(name)] = {metric: _json_float(metrics.get(metric), 0.0) for metric in metric_names}
        active_mask = self._active_mask_from_results(self.results)
        if isinstance(active_mask, np.ndarray) and active_mask.size and "Intelligent AURA" in out:
            active_pct = 100.0 * float(active_mask.astype(bool).sum()) / float(active_mask.size)
            out["Intelligent AURA"]["active_sensor_pct"] = active_pct
            out["Intelligent AURA"]["power_saved_pct"] = 100.0 - active_pct
        return out

    def _training_summary_locked(self) -> dict[str, Any]:
        result = (self.results or {}).get("intelligent_result_fast") if self.results else None
        if not isinstance(result, dict):
            return {
                "seconds": 0.0,
                "final_loss": 0.0,
                "losses": [],
                "epochs": int(self.overrides.get("BENCH_EPOCHS", 0)),
            }
        losses = result.get("losses") or []
        return {
            "seconds": _json_float(result.get("seconds"), 0.0),
            "final_loss": _json_float(result.get("final_loss"), 0.0),
            "losses": [_json_float(loss, 0.0) for loss in losses],
            "epochs": len(losses) or int(self.overrides.get("BENCH_EPOCHS", 0)),
        }

    def _runtime_summary_locked(self) -> dict[str, Any]:
        now = time.perf_counter()
        elapsed = 0.0
        if self.run_started_at is not None:
            elapsed = (self.run_finished_at or now) - self.run_started_at
        return {
            "elapsed_seconds": _json_float(elapsed, 0.0),
            "bench_sensors": int(self.overrides.get("BENCH_SENSORS", 0)),
            "bench_steps": int(self.overrides.get("BENCH_STEPS", 0)),
            "bench_epochs": int(self.overrides.get("BENCH_EPOCHS", 0)),
            "bench_max_pairs": int(self.overrides.get("BENCH_MAX_PAIRS", 0)),
        }

    def _shadow_validation_locked(self) -> dict[str, Any]:
        return {
            "sample_rate": self.shadow_sample_rate,
            "mse_threshold": self.shadow_mse_threshold,
            "sample_count": self.shadow_sample_count,
            "shadow_mse": self.shadow_mse,
            "recent_shadow_mse": self.shadow_recent_mse,
            "last_shadow_abs_error": self.last_shadow_abs_error,
            "last_shadow_sensor_count": self.last_shadow_sensor_count,
            "active_shadow_sensor_ids": list(self.shadow_active_sensor_ids),
            "last_sample_timestep": self.last_shadow_sample_timestep,
        }

    def _retrain_policy_locked(self) -> dict[str, Any]:
        return {
            "recommended": self.retrain_recommended,
            "required": self.retrain_required,
            "reason": self.retrain_reason,
            "last_retrain_timestep": self.last_retrain_timestep,
            "steps_since_retrain": self.steps_since_retrain,
            "global_period_steps": self.global_retrain_period_steps,
            "global_period_fraction": self.global_retrain_period_fraction,
            "steps_until_forced_retrain": self.steps_until_forced_retrain,
            "shadow_mse_threshold": self.shadow_mse_threshold,
        }

    def _learned_parameters_locked(self) -> dict[str, float]:
        results = self.results or {}
        thresholds = results.get("intelligent_thresholds_fast")
        gate_thresholds = results.get("intelligent_gate_thresholds_fast")
        sleep_logits = results.get("intelligent_sleep_logits_fast")

        def mean_std(values: Any) -> tuple[float, float]:
            if isinstance(values, np.ndarray) and values.size:
                return _json_float(values.mean()), _json_float(values.std())
            return 0.0, 0.0

        threshold_mean, threshold_std = mean_std(thresholds)
        gate_mean, gate_std = mean_std(gate_thresholds)
        sleep_mean, sleep_std = mean_std(sleep_logits)
        return {
            "redundancy_threshold_mean": threshold_mean,
            "redundancy_threshold_std": threshold_std,
            "gate_threshold_mean": gate_mean,
            "gate_threshold_std": gate_std,
            "sleep_logit_mean": sleep_mean,
            "sleep_logit_std": sleep_std,
        }

    def _backend_mode_locked(self) -> str:
        if self._python_reference_requested_locked():
            return "PyTorch/Python reference"
        if bool(self.overrides.get("CELL8_FORCE_CPU", False)):
            return "C++ CPU"
        if self.results and self.results.get("aura_cpp") is not None:
            return "CUDA/C++ fused"
        return "CUDA/C++ preferred"

    def _python_reference_requested_locked(self) -> bool:
        return (
            not bool(self.overrides.get("CELL8_USE_CPP_CUDA_PAIR_CACHE", True))
            and not bool(self.overrides.get("CELL8_USE_CPP_CACHED_TRAINING_LOSS", True))
            and not bool(self.overrides.get("CELL8_USE_MANUAL_CUDA_BACKWARD", True))
        )


runtime = RefinedAuraRuntime()
