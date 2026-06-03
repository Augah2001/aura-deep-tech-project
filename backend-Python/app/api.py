import asyncio
import contextlib
import io
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import datasets, storage
from .hardware_bridge import bridge
from .integration_api import router as integration_router
from .refined_runtime import runtime


app = FastAPI(title="AURA Refined Optimized Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://localhost:3005",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(integration_router)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@app.on_event("startup")
async def startup_event():
    storage.init_storage()
    print("Refined optimized AURA backend ready.")


@app.get("/status")
async def get_status():
    return runtime.status()


@app.get("/hardware/status")
async def get_hardware_status():
    return runtime.hardware_status()


@app.get("/hardware/ports")
async def get_hardware_ports():
    return bridge.list_ports()


@app.get("/kernel/status")
async def get_kernel_status():
    return runtime.kernel_proof()


def _fallback_explanation(status: dict, question: str = "") -> dict:
    policies = status.get("policy_metrics") or {}
    aura = policies.get("Intelligent AURA") or {}
    shadow = status.get("shadow_validation") or {}
    retrain = status.get("retrain_policy") or {}
    active_pct = aura.get("active_sensor_pct")
    if active_pct is None:
        total = status.get("total_sensors") or 0
        active = status.get("active_sensors") or 0
        active_pct = (100.0 * active / total) if total else 0.0
    power = aura.get("power_saved_pct", status.get("power_saved_percent", 0.0))
    recall = aura.get("anomaly_recall_pct", (status.get("metrics") or {}).get("anomaly_recall_pct", 0.0))
    mse = aura.get("global_reconstruction_mse", (status.get("metrics") or {}).get("global_reconstruction_mse", 0.0))
    retrain_state = "required" if retrain.get("required") else "recommended" if retrain.get("recommended") else "not required"
    focus = f" In response to the question, {question.strip()}" if question.strip() else ""
    summary = (
        f"AURA is currently saving {power:.2f}% power while keeping {active_pct:.2f}% of sensors active. "
        f"Anomaly recall is {recall:.2f}% and reconstruction MSE is {mse:.6f}. "
        f"Shadow mode is sampling {(shadow.get('sample_rate') or 0.0) * 100:.1f}% of policy-sleeping sensors; "
        f"recent shadow MSE is {(shadow.get('recent_shadow_mse') or 0.0):.6f}. "
        f"Retraining is {retrain_state}."
        f"{focus}"
    )
    bullets = [
        "Green nodes are active, grey nodes are policy-sleeping, red/orange nodes indicate anomaly pressure, and blue-outlined nodes are shadow-validated.",
        "Use the what-if sliders to trade power saving against recall, active-sensor budget, anomaly strictness, and shadow sampling.",
        "Use Challenge AURA to inject noise, drift, anomaly pressure, sensor loss, redundancy, or a backend switch before starting a new run.",
    ]
    if retrain.get("reason"):
        bullets.insert(1, f"Retraining signal: {retrain.get('reason')}")
    return {"mode": "local_explainer", "summary": summary, "bullets": bullets}


@app.post("/assistant/explain")
async def explain_status(request: Request):
    body = await request.json()
    status = body.get("status") if isinstance(body.get("status"), dict) else runtime.status()
    question = str(body.get("question") or "")
    fallback = _fallback_explanation(status, question)
    endpoint = os.getenv("AURA_LLM_API_URL")
    api_key = os.getenv("AURA_LLM_API_KEY")
    if not endpoint or not api_key:
        return fallback

    prompt = {
        "instruction": "Explain the AURA sensor-network result clearly for a project demonstration. Do not invent metrics. Use only the JSON values.",
        "question": question,
        "status": {
            "backend_mode": status.get("backend_mode"),
            "total_sensors": status.get("total_sensors"),
            "active_sensors": status.get("active_sensors"),
            "policy_metrics": status.get("policy_metrics"),
            "shadow_validation": status.get("shadow_validation"),
            "retrain_policy": status.get("retrain_policy"),
            "active_budget_band": status.get("active_budget_band"),
            "runtime": status.get("runtime"),
            "dataset": status.get("dataset"),
        },
    }
    payload = json.dumps({"prompt": prompt, "fallback": fallback}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            text = (
                data.get("summary")
                or data.get("answer")
                or data.get("response")
                or data.get("content")
                or data.get("text")
                or data.get("message")
            )
            if not text and isinstance(data.get("choices"), list) and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        text = message.get("content")
                    text = text or choice.get("text")
            bullets = data.get("bullets") if isinstance(data.get("bullets"), list) else fallback["bullets"]
            if text:
                return {"mode": "llm", "summary": str(text), "bullets": [str(item) for item in bullets[:5]]}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return fallback
    return fallback


def _max_pairs_for(sensor_count: int) -> int:
    return min(4096, max(120, sensor_count * (sensor_count - 1) // 2))


def _parse_log_value(log_text: str, prefix: str) -> str | None:
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix.lower()):
            return stripped.split("=", 1)[1].strip() if "=" in stripped else None
    return None


def _run_acceleration_backend(label: str, runner, overrides: dict, sensor_count: int, step_count: int, max_pairs: int, epochs: int) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    base_overrides = {
        "SAFETY_EPOCHS": epochs,
        "CELL7_GATE_THRESHOLDS": [0.10],
        "CELL7_GATE_SHARPNESSES": [40.0],
        "CELL7_LORAWAN_PERIODS": [5],
        "CELL7_LORAWAN_THRESHOLDS": [0.10],
        "CELL8_SHOW_PLOTS": False,
        "CELL8_DISABLE_TORCH_COMPILE": True,
        "CELL8_PREGENERATE_TRAINING_DATASETS": True,
        "CELL8_FIXED_TRAINING_DATASET": True,
        "CELL8_USE_CACHED_FIXED_DATASET_TRAINING": True,
        "CELL8_FINAL_CACHED_REPORT_ONLY": False,
        "BENCH_SENSORS": sensor_count,
        "BENCH_STEPS": step_count,
        "BENCH_MAX_PAIRS": max_pairs,
    }
    stream = io.StringIO()
    run_overrides = {**base_overrides, **overrides}
    with contextlib.redirect_stdout(stream):
        result = runner(**run_overrides)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    training = result.get("intelligent_result_fast", {})
    return {
        "backend": label,
        "training_seconds": float(training.get("seconds") or 0.0),
        "final_loss": float(training.get("final_loss") or 0.0),
        "device": _parse_log_value(stream.getvalue(), "device") or "unknown",
        "status": "ok",
    }


def _run_acceleration_comparison(sensor_count: int = 128, step_count: int = 2000, max_pairs: int = 10000, epochs: int = 20) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from aura_refined_benchmark import run_refined_aura_benchmark
    from python_pytorch_baseline import run_python_pytorch_baseline

    specs = [
        (
            "Python/PyTorch",
            run_python_pytorch_baseline,
            {
                "CELL8_FORCE_CPU": True,
                "CELL8_USE_CACHED_FIXED_DATASET_TRAINING": False,
                "CELL8_PREGENERATE_TRAINING_DATASETS": False,
                "CELL8_FIXED_TRAINING_DATASET": False,
            },
            "Readable baseline with ordinary autograd recurrence",
        ),
        (
            "Optimized C++ CPU",
            run_refined_aura_benchmark,
            {
                "CELL8_FORCE_CPU": True,
                "CELL8_USE_CPP_CUDA_PAIR_CACHE": True,
                "CELL8_USE_CPP_CACHED_TRAINING_LOSS": True,
                "CELL8_USE_MANUAL_CUDA_BACKWARD": True,
            },
            "Compiled C++ cached loss and manual backward on CPU",
        ),
        (
            "CUDA/C++ fused",
            run_refined_aura_benchmark,
            {
                "CELL8_FORCE_CPU": False,
                "CELL8_USE_CPP_CUDA_PAIR_CACHE": True,
                "CELL8_USE_CPP_CACHED_TRAINING_LOSS": True,
                "CELL8_USE_MANUAL_CUDA_BACKWARD": True,
            },
            "Fused cached training loss and manual backward on GPU",
        ),
    ]
    rows = []
    for label, runner, overrides, purpose in specs:
        try:
            row = _run_acceleration_backend(label, runner, overrides, sensor_count, step_count, max_pairs, epochs)
        except Exception as exc:
            row = {
                "backend": label,
                "training_seconds": float("nan"),
                "final_loss": float("nan"),
                "device": "failed",
                "status": f"{type(exc).__name__}: {exc}",
            }
        row["purpose"] = purpose
        rows.append(row)

    python_time = next((row["training_seconds"] for row in rows if row["backend"] == "Python/PyTorch" and math.isfinite(row["training_seconds"]) and row["training_seconds"] > 0), float("nan"))
    for row in rows:
        seconds = row["training_seconds"]
        row["speedup_vs_pytorch"] = float(python_time / seconds) if math.isfinite(python_time) and math.isfinite(seconds) and seconds > 0 else float("nan")
    return {"sensor_count": sensor_count, "step_count": step_count, "max_pairs": max_pairs, "epochs": epochs, "rows": rows}


@app.post("/runtime/acceleration-comparison")
async def runtime_acceleration_comparison(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    sensor_count = max(16, min(128, int(body.get("sensor_count", 128))))
    step_count = max(80, min(3000, int(body.get("step_count", 2000))))
    max_pairs = max(120, min(20000, int(body.get("max_pairs", 10000))))
    epochs = max(1, min(60, int(body.get("epochs", 20))))
    return await asyncio.to_thread(_run_acceleration_comparison, sensor_count, step_count, max_pairs, epochs)


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runtime.status())
            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        return


@app.get("/history")
async def get_history():
    return {"runs": storage.list_runs(30)}


@app.get("/storage/status")
async def get_storage_status():
    return storage.storage_backend()


@app.get("/datasets")
async def list_datasets():
    return {"datasets": storage.list_datasets()}


@app.post("/datasets/upload")
async def upload_dataset(request: Request):
    filename = request.headers.get("x-filename", "dataset.csv")
    content = await request.body()
    if not content:
        return {"error": "empty upload"}
    return datasets.save_uploaded_dataset(filename, content)


@app.post("/datasets/{dataset_id}/selection")
async def update_dataset_selection(dataset_id: int, request: Request):
    body = await request.json()
    selected_columns = body.get("selected_columns", [])
    if not isinstance(selected_columns, list):
        selected_columns = []
    dataset = storage.get_dataset(dataset_id)
    if not dataset:
        return {"error": "dataset not found"}
    allowed = set(dataset.get("numeric_columns") or [])
    selected = [str(column) for column in selected_columns if str(column) in allowed]
    return storage.update_dataset_selection(dataset_id, selected)


@app.post("/hardware/connect")
async def connect_hardware(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return bridge.connect(body.get("port"), body.get("baud_rate"))


@app.post("/hardware/disconnect")
async def disconnect_hardware():
    return bridge.disconnect()


@app.post("/hardware/sync")
async def sync_hardware():
    command = runtime.hardware_status().get("last_command", "")
    return bridge.sync(command)


@app.post("/start")
async def start_simulation(request: Request):
    params = await request.json()
    return runtime.start(params)


@app.post("/retrain")
async def retrain_simulation(request: Request):
    try:
        params = await request.json()
    except Exception:
        params = {}
    return runtime.retrain(params)


@app.post("/pause")
async def pause_simulation():
    return runtime.pause()


@app.post("/replay/speed")
async def set_replay_speed(request: Request):
    body = await request.json()
    return runtime.set_replay_speed(body.get("speed"))


@app.post("/presentation")
async def start_presentation_mode():
    return runtime.presentation_mode()


@app.post("/diagnostics/clear")
async def clear_diagnostics():
    return runtime.clear_diagnostics()


@app.post("/reset")
async def reset_simulation():
    return runtime.reset()
