# AURA: Refined Optimized Sensor Network

AURA (Autonomous Unsupervised Retraining Algorithm) is a smart sensor-network benchmark and visualization project. This copy is wired to the cleaned refined AURA algorithm package in the parent `aura_algorithm_clean` folder.

The backend now launches the refined optimized benchmark through `aura_refined_benchmark.run_refined_aura_benchmark()`, using the CUDA/C++ fused training path when available and the optimized CPU/PyTorch fallback otherwise. The frontend still uses the same `/start`, `/pause`, `/reset`, and `/status` API, but the status stream is now a replay of the refined algorithm's learned sensor activity and benchmark metrics.

AURA is also a bio-inspired algorithmic framework for optimizing resource usage in large-scale distributed data networks. In this prototype it demonstrates more than 75% power savings while maintaining 99.26% data fidelity across benchmark scenarios.

This project was submitted to the **AlgoFest Hackathon 2025**.

---

## The Universal Problem: The High Cost of Redundancy

In systems that generate multiple parallel data streams, such as algorithmic trading signals, cloud infrastructure metrics, logistics trackers, and IoT sensor data, systemic data redundancy wastes compute, network bandwidth, and energy.

The core problem is the lack of an intelligent, lightweight, and scalable algorithm that can autonomously identify complex, multi-variable redundancies and act on them dynamically at the source.

## Key Features

- **Intelligent Power Saving:** Runs the refined Intelligent AURA policy with a target active-sensor budget band.
- **Optimized Training:** Uses the cleaned PyTorch/C++/CUDA implementation from the parent algorithm package.
- **Real-time Visualization:** A rich Next.js frontend provides a live 3D view of the sensor farm, detailed metrics, and interactive charts.
- **FastAPI Runtime Adapter:** The backend runs the refined benchmark in a background thread and exposes frontend-compatible status.
- **Plug-and-play Sensor API:** External gateways can submit readings and request AURA sleep/wake command bits through `/api/v1`.
- **Persistent Evidence Store:** Uploaded datasets and benchmark history are persisted in PostgreSQL when configured, with SQLite fallback.
- **Benchmarking Suite:** Includes Jupyter notebooks for analyzing and evaluating the performance of the AURA algorithm.

## System Architecture

1. **Backend (Python/FastAPI):**
   - Serves the refined optimized AURA runtime.
   - Calls the cleaned algorithm package in the parent folder.
   - Replays learned active/off sensor masks for the UI.
   - Exposes dashboard APIs and versioned `/api/v1` integration APIs for external sensor systems.

2. **Frontend (Next.js/React):**
   - Provides the web interface for controlling and monitoring the simulation.
   - Features a 3D visualization of the sensor network using React Three Fiber.
   - Displays real-time data and performance metrics through interactive charts.

3. **Arduino:**
   - Contains the code for the physical hardware component of the system, which syncs with the simulation.

## The Solution Approach: A Bio-Inspired Algorithmic Core

AURA (**A**utonomous, **R**esource-**A**ware) treats redundancy optimization as a bio-inspired network problem.

### The Core Innovation: The AURA Index

The brain of the system is a domain-agnostic formula called the **AURA Index (A)**. Inspired by Hebbian learning ("neurons that fire together, wire together"), it treats network nodes as biological neurons and quantifies their co-movement or informational redundancy.

```text
AURA Index = sum(sin^2(pi * s_i / sum(s_j))) / (n * sin^2(pi / n))
```

Where `s_i` and `s_j` are normalized data-stream values, and `n` is the number of streams being compared.

This formula is useful because it can evaluate redundancy across any number of streams, is lightweight enough for real-time edge execution, and returns an interpretable value from 0 to 1.

### Intelligent Architecture: The Operator and The Learner

The algorithm is implemented within a dual-process architecture designed for continuous, autonomous optimization:

1. **The Operator:** Fetches data, computes the AURA Index, and executes real-time optimization decisions.
2. **The Learner:** Uses a Differential Evolution algorithm to discover better parameters for the Operator as data patterns change.

---

## The Concrete Proof: A Real-World Implementation

The prototype implements and benchmarks AURA in a self-optimizing IoT sensor network.

### The Full Prototype

- **Backend:** A high-performance simulation and runtime adapter built with Python and FastAPI.
- **Frontend:** A real-time, interactive 3D visualization built with Next.js and Three.js.
- **Hardware:** Arduino Mega code that mirrors the digital simulation state.

### Performance Benchmarks

| Benchmark Scenario | Average Power Savings | Average Data Fidelity |
| :--- | :---: | :---: |
| Baseline (No AURA) | 0% | 100% |
| **AURA Hybrid System** | **75.94%** | **99.26%** |

![AURA Performance Benchmark Plot](images/benchmark_plot.png)

---

## Instructions to Run

To get the AURA simulation running on your local machine, follow these steps.

### Prerequisites

- Node.js and npm
- Python 3.8+ and pip

### Backend Setup

1. Navigate to the `backend-Python` directory:

   ```bash
   cd backend-Python
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   On Windows, use:

   ```powershell
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:

   ```bash
   python run.py
   ```

   The backend will be running at `http://127.0.0.1:8000`.

### Frontend Setup

1. Navigate to the `frontend-Next.js` directory:

   ```bash
   cd frontend-Next.js
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

   The frontend will be running at `http://localhost:3000`.

## How to Use the Simulation

Once both servers are running, open your browser to `http://localhost:3000`.

- **Controls:** Use the main controls to Start, Pause, or Reset the simulation.
- **Parameters:** Adjust the AURA algorithm's core parameters and the Learner's retraining triggers before starting.
- **Charts:** Click "Show Charts" to view real-time graphs of fidelity and power-saving performance.

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
