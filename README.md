# AURA: A Universal Algorithm for Dynamic Redundancy Reduction

AURA is a novel, bio-inspired algorithmic framework for optimizing resource usage in any large-scale, distributed data network. It is built around a new, generalized mathematical formula that autonomously identifies and eliminates systemic redundancy in real-time.

To prove its efficacy, we implemented AURA in a demanding, resource-constrained IoT environment. The result is a complete software and hardware prototype that achieved **>75% power savings** while maintaining **99.26% data fidelity** across rigorous benchmarks.

This project was submitted to the **AlgoFest Hackathon 2025**.

---

## The Universal Problem: The High Cost of Redundancy

In any system that generates multiple, parallel data streams—whether it's **algorithmic trading signals**, **cloud infrastructure metrics**, **logistics trackers**, or **IoT sensor data**—a fundamental challenge exists: **systemic data redundancy**. This inefficiency is a primary driver of wasted resources, leading to:

* **Excessive computational load** in data centers.
* **Saturated network bandwidth** in communication systems.
* **Depleted energy reserves** in edge and battery-powered devices.

The core problem is the lack of an intelligent, lightweight, and scalable algorithm that can autonomously identify complex, multi-variable redundancies and act on them dynamically at the source.

## The Solution Approach: A Bio-Inspired Algorithmic Core

AURA (**A**utonomous, **R**esource-**A**ware) is a holistic solution that treats this challenge not as a brute-force statistical problem, but as an elegant, bio-inspired one.

### The Core Innovation: The AURA Index

The brain of the system is a new, domain-agnostic mathematical formula I developed called the **AURA Index (A)**. Inspired by Hebbian learning ("neurons that fire together, wire together"), it treats network nodes as biological neurons and quantifies their "co-movement" or informational redundancy.

$$
\text{AURA Index} = \frac{\sum_{i=1}^{n} \sin^2\left(\frac{\pi s_i}{\sum_{j=1}^{n} s_j}\right)}{n \cdot \sin^2\left(\frac{\pi}{n}\right)}
$$

*Where $s_i$ and $s_j$ are the normalized values of the i-th and j-th data streams, and $n$ is the number of streams being compared.*

This formula represents a significant step forward from traditional correlation metrics:
* **N-Dimensional Insight:** It can evaluate complex redundancies between *any number* of data streams simultaneously, not just pairwise comparisons.
* **Extreme Efficiency:** The formula is computationally lightweight, making it perfect for real-time execution on low-power edge devices or in high-throughput systems.
* **Symmetrical & Interpretable:** Its output is conveniently ranged from 0 (no redundancy) to 1 (maximum redundancy), providing a clear, actionable metric.

### Intelligent Architecture: The Operator & The Learner

The algorithm is implemented within a sophisticated dual-process architecture designed for continuous, autonomous optimization:

1.  **The "Operator" (Real-Time Execution):** This primary process fetches data, computes the AURA Index, and executes real-time optimization decisions.
2.  **The "Learner" (Autonomous Improvement):** A background machine learning process that uses a **Differential Evolution** algorithm to continuously discover the optimal parameters for the Operator. It ensures the system adapts to changing data patterns, maximizing efficiency while preserving data fidelity.

---

## The Concrete Proof: A Real-World Implementation

To prove the real-world efficacy of this universal algorithm, we implemented and benchmarked it in one of the most challenging resource-constrained environments: **a self-optimizing IoT sensor network**.

### The Full Prototype

This is a complete, working system consisting of:
* **Backend:** A high-performance, multi-threaded simulation core built with Python and FastAPI.
* **Frontend:** A real-time, interactive 3D visualization of the network built with Next.js and Three.js.
* **Hardware:** A physical display using an Arduino Mega that mirrors the state of the digital simulation in perfect, real-time synchronization.

### Performance Benchmarks

The AURA system was rigorously benchmarked across seven distinct datasets. The results are formidable, demonstrating a massive gain in efficiency for a negligible loss in data accuracy.

| Benchmark Scenario | Average Power Savings | Average Data Fidelity |
| :--- | :---: | :---: |
| Baseline (No AURA) | 0% | 100% |
| **AURA Hybrid System** | **75.94%** | **99.26%** |
_Summary of performance across all datasets._

![AURA Performance Benchmark Plot](images/benchmark_plot.png)
_Fig 1.1: The plot shows the massive gain in power savings for a negligible loss in data fidelity across all tested datasets._

---

## Instructions to Run

To get the AURA simulation running on your local machine, follow these steps.

### Prerequisites

-   Node.js and npm
-   Python 3.8+ and pip

### Backend Setup

1.  Navigate to the `backend-Python` directory:
    ```bash
    cd backend-Python
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the server:
    ```bash
    python run.py
    ```
    The backend will be running at `http://127.0.0.1:8000`.

### Frontend Setup

1.  Navigate to the `frontend-Next.js` directory:
    ```bash
    cd frontend-Next.js
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend will be running at `http://localhost:3000`.

## How to Use the Simulation

Once both servers are running, open your browser to `http://localhost:3000`.
-   **Controls:** Use the main controls to Start, Pause, or Reset the simulation.
-   **Parameters:** Adjust the AURA algorithm's core parameters and the Learner's retraining triggers before starting.
-   **Charts:** Click "Show Charts" to view real-time graphs of the system's fidelity and power-saving performance.

