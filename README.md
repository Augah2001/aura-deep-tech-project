
# AURA: An Autonomous, Self-Optimizing Sensor Network

AURA (Autonomous Unsupervised Retraining Algorithm) is a sophisticated simulation of a smart sensor network that intelligently manages its own power consumption. It features a cutting-edge, self-optimizing control system that continuously learns from and adapts to the data it processes, ensuring maximum efficiency and predictive accuracy without human intervention.

This project demonstrates a "hybrid model" approach where the system operates in a power-saving "shadow mode" and automatically triggers a retraining pipeline when its performance degrades or after a set interval. This ensures the network remains robust and efficient over time.

The simulation is visualized and controlled through a sleek, real-time web interface built with Next.js, offering a comprehensive view of the network's state, performance, and the AURA algorithm in action.

## Key Features

- **Intelligent Power Saving:** Implements the AURA algorithm to dynamically deactivate and reactivate sensors, significantly reducing power consumption.
- **Autonomous Retraining:** The system automatically detects performance degradation and retrains itself on fresh data to optimize its internal parameters.
- **Real-time Visualization:** A rich Next.js frontend provides a live 3D view of the sensor farm, detailed metrics, and interactive charts.
- **Advanced Simulation Core:** The backend, built with Python and FastAPI, runs a sophisticated, multi-threaded simulation of the sensor network.
- **Differential Evolution Learner:** Utilizes a powerful optimization algorithm to discover the most effective parameters for the power-saving logic during retraining cycles.
- **Benchmarking Suite:** Includes Jupyter notebooks for analyzing and evaluating the performance of the AURA algorithm.

## System Architecture

The project is composed of three main components:

1.  **Backend (Python/FastAPI):**
    - Serves the core simulation logic.
    - Manages the state of the sensor network.
    - Implements the AURA algorithm and the autonomous retraining pipeline.
    - Exposes a REST API for the frontend to interact with the simulation.

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

## The AURA Algorithm

The core of the power-saving logic is the AURA (Autonomous Unsupervised Retraining Algorithm) index. This mathematical function measures the degree of informational redundancy among a small group of sensors. When the AURA index for a group of sensors exceeds a certain threshold, it indicates that their readings are highly correlated. The system then deactivates the "noisiest" sensor in that group for a set duration, saving power without a significant loss of information.

## Autonomous Retraining

AURA's most advanced feature is its ability to self-optimize. The system operates in one of two main phases:

1.  **Shadow Operation:** The default power-saving mode. During this phase, the system occasionally performs "undercover" quality checks by predicting a sensor's value and comparing it to the true reading. This allows it to calculate its own predictive accuracy (fidelity).

2.  **Collecting & Learning:** If the system's fidelity drops below a set threshold, or if a maximum time interval has passed, it automatically enters a data collection phase. It reactivates all sensors to gather a fresh, high-quality dataset. This data is then passed to the "learner" module, which uses a Differential Evolution algorithm to find a new, optimized set of parameters (threshold and duration) for the AURA algorithm. Once complete, the new parameters are seamlessly deployed, and the system returns to shadow operation.

## Benchmarking

The `AURA_benchmarks.ipynb` notebook in the `backend-Python` directory provides a suite for testing and analyzing the performance of the AURA algorithm under various conditions.

## Future Work

- **Hardware Integration:** Fully integrate the Arduino component for a physical demonstration of the sensor network.
- **Advanced Data Sources:** Connect the simulation to live, real-world data streams.
- **Algorithm Expansion:** Experiment with different machine learning models for the learner module.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
