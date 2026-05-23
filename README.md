
# AURA: Refined Optimized Sensor Network

AURA (Autonomous Unsupervised Retraining Algorithm) is a smart sensor-network benchmark and visualization project. This copy is wired to the cleaned refined AURA algorithm package in the parent `aura_algorithm_clean` folder.

The backend now launches the refined optimized benchmark through `aura_refined_benchmark.run_refined_aura_benchmark()`, using the CUDA/C++ fused training path when available and the optimized CPU/PyTorch fallback otherwise. The frontend still uses the same `/start`, `/pause`, `/reset`, and `/status` API, but the status stream is now a replay of the refined algorithm's learned sensor activity and benchmark metrics.

The simulation is visualized and controlled through a sleek, real-time web interface built with Next.js, offering a comprehensive view of the network's state, performance, and the AURA algorithm in action.

## Key Features

- **Intelligent Power Saving:** Runs the refined Intelligent AURA policy with a target active-sensor budget band.
- **Optimized Training:** Uses the cleaned PyTorch/C++/CUDA implementation from the parent algorithm package.
- **Real-time Visualization:** A rich Next.js frontend provides a live 3D view of the sensor farm, detailed metrics, and interactive charts.
- **FastAPI Runtime Adapter:** The backend runs the refined benchmark in a background thread and exposes frontend-compatible status.
- **Plug-and-play Sensor API:** External gateways can submit readings and request AURA sleep/wake command bits through `/api/v1`.
- **Persistent Evidence Store:** Uploaded datasets and benchmark history are persisted in PostgreSQL when configured, with SQLite fallback.
- **Benchmarking Suite:** Includes Jupyter notebooks for analyzing and evaluating the performance of the AURA algorithm.

## System Architecture

The project is composed of three main components:

1.  **Backend (Python/FastAPI):**
    - Serves the refined optimized AURA runtime.
    - Calls the cleaned algorithm package in the parent folder.
    - Replays learned active/off sensor masks for the UI.
    - Exposes dashboard APIs and versioned `/api/v1` integration APIs for external sensor systems.

2.  **Frontend (Next.js/React):**
    - Provides a user-friendly web interface for controlling and monitoring the simulation.
    - Features a 3D visualization of the sensor network using React Three Fiber.
    - Displays real-time data and performance metrics through interactive charts.

3.  **Arduino:**
    - Contains the code for the physical hardware component of the system, which syncs with the simulation.

## Getting Started

To get the AURA simulation running on your local machine, follow these steps.

### Prerequisites

- **Node.js and npm:** Required for the Next.js frontend.
- **Python 3.8+ and pip:** Required for the FastAPI backend.

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend-Python
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the backend server:**
    ```bash
    python run.py
    ```
    The backend will be running at `http://127.0.0.1:8000`.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend-Next.js
    ```

2.  **Install the required npm packages:**
    ```bash
    npm install
    ```

3.  **Start the frontend development server:**
    ```bash
    npm run dev
    ```
    The frontend will be running at `http://localhost:3000`.

## How to Use the Simulation

Once both the backend and frontend servers are running, open your web browser and navigate to `http://localhost:3000`.

- **Start/Pause:** Use the "Start" button to begin the simulation. The button will change to "Pause," allowing you to halt the simulation at any time.
- **Reset:** The "Reset" button stops the simulation and resets it to its initial state.
- **Parameters:** You can adjust the core AURA parameters and the autonomous retraining triggers before starting the simulation.
- **Charts:** Click the "Show Charts" button to view real-time graphs of the system's fidelity and power-saving performance.

## The Refined AURA Algorithm

The active backend entry point is `backend-Python/app/refined_runtime.py`. It imports the parent clean package and runs:

```python
from aura_refined_benchmark import run_refined_aura_benchmark
```

Useful `/start` overrides can be sent as JSON:

```json
{
  "BENCH_SENSORS": 500,
  "BENCH_STEPS": 120,
  "BENCH_EPOCHS": 20,
  "BENCH_MAX_PAIRS": 10000,
  "CELL8_FORCE_CPU": false
}
```

The older Numba threshold/duration simulator files are still present for reference, but the FastAPI app no longer uses them.

## Plug-and-play Sensor API

External systems can integrate with AURA without using the dashboard. The stable integration surface is:

```text
http://127.0.0.1:8000/api/v1
```

Common endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/health` | GET | Check service, storage, runtime, and hardware mode |
| `/api/v1/readings` | POST | Submit sensor readings for a named network |
| `/api/v1/policy/latest` | GET | Get the latest AURA command policy |
| `/api/v1/policy/evaluate` | POST | Submit readings and receive command bits in one request |
| `/api/v1/hardware/sync` | POST | Send command bits through the Arduino serial bridge |

Example request:

```json
{
  "network_id": "farm-zone-1",
  "node_ids": [0, 1, 2, 3],
  "readings": [0.42, 0.44, 0.91, 0.43]
}
```

The response includes a `command_bits` string where `0` means active/on and `1` means sleep/off. Full details are in `../docs/aura_sensor_integration_api.md`.

## Benchmarking

The `AURA_benchmarks.ipynb` notebook in the `backend-Python` directory provides a suite for testing and analyzing the performance of the AURA algorithm under various conditions.

## Future Work

- **Hardware Integration:** Fully integrate the Arduino component for a physical demonstration of the sensor network.
- **Advanced Data Sources:** Connect the simulation to live, real-world data streams.
- **Algorithm Expansion:** Experiment with different machine learning models for the learner module.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
