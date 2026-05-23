import asyncio

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
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(integration_router)


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


@app.get("/kernel/status")
async def get_kernel_status():
    return runtime.kernel_proof()


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


@app.post("/pause")
async def pause_simulation():
    return runtime.pause()


@app.post("/reset")
async def reset_simulation():
    return runtime.reset()
