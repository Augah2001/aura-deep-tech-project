from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .hardware_bridge import bridge
from .refined_runtime import runtime


router = APIRouter(prefix="/api/v1", tags=["sensor integration"])
_lock = threading.RLock()
_latest_readings: dict[str, dict[str, Any]] = {}


class SensorReadingBatch(BaseModel):
    network_id: str = Field(default="default", description="External network or deployment identifier.")
    node_ids: list[int] | None = Field(default=None, description="Optional node ids matching the reading order.")
    readings: list[float] | dict[str, float] = Field(description="Sensor readings as an ordered list or keyed object.")
    timestamp: str | None = Field(default=None, description="Optional upstream timestamp.")
    units: str | None = Field(default=None, description="Optional physical unit such as Celsius, humidity, lux, etc.")


class PolicyRequest(BaseModel):
    network_id: str = "default"
    readings: list[float] | dict[str, float] | None = None
    node_ids: list[int] | None = None
    sync_hardware: bool = False


def _require_api_key(x_aura_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("AURA_API_KEY")
    if expected and x_aura_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing AURA API key")


def _coerce_readings(readings: list[float] | dict[str, float], node_ids: list[int] | None = None) -> tuple[list[int], list[float]]:
    if isinstance(readings, dict):
        pairs: list[tuple[int, float]] = []
        for index, (key, value) in enumerate(readings.items()):
            try:
                node_id = int(key)
            except (TypeError, ValueError):
                node_id = index
            pairs.append((node_id, float(value)))
        pairs.sort(key=lambda item: item[0])
        return [node_id for node_id, _ in pairs], [value for _, value in pairs]

    values = [float(value) for value in readings]
    if node_ids and len(node_ids) == len(values):
        return [int(node_id) for node_id in node_ids], values
    return list(range(len(values))), values


def _policy_from_runtime(node_ids: list[int]) -> dict[str, Any]:
    status = runtime.status()
    sensor_states = {int(sensor["id"]): bool(sensor.get("is_off")) for sensor in status.get("sensors", [])}
    shadow_states = {int(sensor["id"]): bool(sensor.get("is_shadow")) for sensor in status.get("sensors", [])}
    hardware = status.get("hardware") or {}
    default_active = not sensor_states

    policies = []
    hardware_bits = (hardware.get("last_command") or "").split(",")
    command_bits = []
    for node_id in node_ids:
        is_sleeping = sensor_states.get(node_id, False if default_active else False)
        bit = hardware_bits[node_id].strip() if node_id < len(hardware_bits) and hardware_bits[node_id].strip() in {"0", "1"} else ("1" if is_sleeping else "0")
        is_shadow = shadow_states.get(node_id, False)
        command_bits.append(bit)
        policies.append(
            {
                "node_id": node_id,
                "command_bit": bit,
                "policy_state": "sleep" if is_sleeping else "active",
                "hardware_state": "active" if bit == "0" else "sleep",
                "shadow_validation": is_shadow,
                "reason": "shadow_validation_holdout" if is_shadow else ("learned_runtime_mask" if node_id in sensor_states else "default_active_until_visible_in_runtime"),
            }
        )

    if not node_ids:
        command = hardware.get("last_command") or ""
    else:
        command = ",".join(command_bits)

    return {
        "algorithm": status.get("algorithm", "refined_optimized_aura"),
        "phase": status.get("current_phase"),
        "policy_source": "learned_runtime_mask" if sensor_states else "default_active",
        "network_power_saved_percent": status.get("power_saved_percent", 0.0),
        "active_budget_band": status.get("active_budget_band"),
        "command_convention": {"0": "active/on", "1": "sleep/off"},
        "command_bits": command,
        "node_policies": policies,
        "hardware": {
            "mode": hardware.get("mode"),
            "arduino_status": hardware.get("arduino_status"),
            "last_ack": hardware.get("last_ack"),
            "last_error": hardware.get("last_error"),
        },
        "shadow_validation": status.get("shadow_validation"),
        "retrain_policy": status.get("retrain_policy"),
    }


@router.get("/health")
async def integration_health(_: None = Depends(_require_api_key)):
    status = runtime.status()
    return {
        "ok": True,
        "service": "aura-sensor-integration-api",
        "phase": status.get("current_phase"),
        "storage": status.get("storage"),
        "hardware_mode": status.get("hardware", {}).get("mode"),
    }


@router.post("/readings")
async def submit_readings(batch: SensorReadingBatch, _: None = Depends(_require_api_key)):
    node_ids, readings = _coerce_readings(batch.readings, batch.node_ids)
    with _lock:
        _latest_readings[batch.network_id] = {
            "network_id": batch.network_id,
            "node_ids": node_ids,
            "readings": readings,
            "timestamp": batch.timestamp,
            "units": batch.units,
            "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    return {
        "accepted": True,
        "network_id": batch.network_id,
        "sensor_count": len(readings),
        "received_at": _latest_readings[batch.network_id]["received_at"],
    }


@router.get("/readings/latest")
async def latest_readings(network_id: str = "default", _: None = Depends(_require_api_key)):
    with _lock:
        reading = _latest_readings.get(network_id)
    if not reading:
        raise HTTPException(status_code=404, detail="no readings have been submitted for this network")
    return reading


@router.get("/policy/latest")
async def latest_policy(network_id: str = "default", _: None = Depends(_require_api_key)):
    with _lock:
        reading = _latest_readings.get(network_id)
    node_ids = list(reading.get("node_ids", [])) if reading else [sensor["id"] for sensor in runtime.status().get("sensors", [])]
    policy = _policy_from_runtime(node_ids)
    policy["network_id"] = network_id
    policy["reading_timestamp"] = reading.get("timestamp") if reading else None
    return policy


@router.post("/policy/evaluate")
async def evaluate_policy(request: PolicyRequest, _: None = Depends(_require_api_key)):
    if request.readings is not None:
        node_ids, readings = _coerce_readings(request.readings, request.node_ids)
        with _lock:
            _latest_readings[request.network_id] = {
                "network_id": request.network_id,
                "node_ids": node_ids,
                "readings": readings,
                "timestamp": None,
                "units": None,
                "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
    else:
        with _lock:
            reading = _latest_readings.get(request.network_id)
        node_ids = list(reading.get("node_ids", [])) if reading else []

    policy = _policy_from_runtime(node_ids)
    policy["network_id"] = request.network_id
    if request.sync_hardware:
        policy["hardware_sync"] = bridge.sync(policy["command_bits"])
    return policy


@router.post("/hardware/sync")
async def integration_hardware_sync(request: Request, _: None = Depends(_require_api_key)):
    body = await request.json()
    command = str(body.get("command_bits") or runtime.hardware_status().get("last_command") or "")
    return bridge.sync(command)
