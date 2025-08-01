# To run this:
# 1. Install libraries: 
#    pip install fastapi uvicorn python-multipart pandas numpy scipy aiohttp numba
# 2. Save this file as main.py
# 3. Run the server: 
#    uvicorn main:app --reload --port 8000

import asyncio
import threading
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from scipy.optimize import differential_evolution
from numba import njit
from itertools import combinations

# --- App & Global State ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust for your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Default Simulation State (Now dynamic) ---
def get_default_state():
    return {
        "is_running": False,
        "timestep": 0,
        "current_phase": "shadow_op", # Start in shadow mode
        
        # Core AURA parameters
        "threshold": 0.98, 
        "duration": 40,
        "n_way_comparison": 2,

        # Shadow mode configuration
        "shadow_mode_probability": 0.05,
        
        # Hybrid model triggers
        "hybrid_fidelity_threshold": 0.97,
        "hybrid_max_timesteps_since_retrain": 2880, # e.g., retrain at least every 2 days if 1 step = 1 minute
        "last_retrain_timestep": 0,
        "collection_period": 200,

        # Performance metrics
        "total_power_saved_steps": 0,
        "last_learner_fidelity": 1.0,
        "shadow_fidelity_error": 0.0,
        "shadow_fidelity_count": 0,
        "shadow_samples": [],
        
        # Sensor info will be populated dynamically after data load
        "sensors": [],
        "total_sensors": 0,
        "data": None,
        "lock": threading.Lock(),
        "learner_status": "idle"
    }

SIMULATION_STATE = get_default_state()
operator_thread = None
learner_thread = None

# --- Core Simulation & Learner Logic ---

@njit
def generalized_redundancy_metric(readings, n):
    if n < 2: return 0.0
    total = np.sum(readings)
    if total <= 1e-9: return 0.0
    denominator = n * (np.sin(np.pi / n)**2)
    if denominator < 1e-9: return 0.0
    numerator = np.sum(np.sin(np.pi * readings / total)**2)
    return numerator / denominator

@njit
def run_simulation_for_learner(threshold_R, duration, n_way, data_input):
    n_sensors = data_input.shape[1]
    current_timesteps = data_input.shape[0]
    if current_timesteps < 2: return 0.0, 1.0
    
    deactivated_storage = np.full((n_sensors, 3), -1, dtype=np.int32)
    num_deactivated = 0
    sensor_noise_variance = np.zeros(n_sensors, dtype=np.float64)
    last_readings = data_input[0].copy()
    power_saved, total_squared_error, fidelity_count = 0.0, 0.0, 0.0

    for t in range(1, current_timesteps):
        temp_storage = np.full((n_sensors, 3), -1, dtype=np.int32)
        temp_num_deactivated = 0
        is_sensor_off = np.zeros(n_sensors, dtype=np.bool_)
        
        active_deactivations = 0
        for k in range(num_deactivated):
            deactivated_id, _, end_time = deactivated_storage[k]
            if t < end_time:
                temp_storage[active_deactivations] = deactivated_storage[k]
                is_sensor_off[deactivated_id] = True
                active_deactivations += 1
        deactivated_storage, num_deactivated = temp_storage, active_deactivations
        power_saved += num_deactivated
        
        readings = data_input[t]
        deltas = readings - last_readings
        sensor_noise_variance = 0.99 * sensor_noise_variance + 0.01 * (deltas**2)
        last_readings = readings.copy()
        
        active_sensor_indices = np.where(~is_sensor_off)[0]
        
        if len(active_sensor_indices) < n_way: continue

        indices = np.arange(n_way)
        while True:
            combo_indices = active_sensor_indices[indices]
            combo_readings = readings[combo_indices]
            aura_index = generalized_redundancy_metric(combo_readings, n_way)

            if aura_index > threshold_R:
                combo_noise = sensor_noise_variance[combo_indices]
                max_noise_idx_in_combo = np.argmax(combo_noise)
                sensor_to_deactivate_id = combo_indices[max_noise_idx_in_combo]

                if not is_sensor_off[sensor_to_deactivate_id]:
                    peer_indices = np.delete(combo_indices, max_noise_idx_in_combo)
                    estimated_reading = np.mean(readings[peer_indices])
                    true_reading = readings[sensor_to_deactivate_id]
                    total_squared_error += (true_reading - estimated_reading)**2
                    fidelity_count += 1
                    
                    deactivated_storage[num_deactivated] = [sensor_to_deactivate_id, -1, t + duration]
                    is_sensor_off[sensor_to_deactivate_id] = True
                    num_deactivated += 1
            
            i = n_way - 1
            while i >= 0 and indices[i] == i + len(active_sensor_indices) - n_way: i -= 1
            if i < 0: break
            indices[i] += 1
            for j in range(i + 1, n_way): indices[j] = indices[j - 1] + 1

    mse = total_squared_error / fidelity_count if fidelity_count > 0 else 0.0
    fidelity_score = max(0.0, 1.0 - mse)
    power_saved_percentage = power_saved / (n_sensors * current_timesteps) if current_timesteps > 0 else 0.0
    return power_saved_percentage, fidelity_score

def objective_function(params, training_data, n_way):
    threshold_R, duration = params[0], int(round(params[1]))
    power_saved, fidelity = run_simulation_for_learner(threshold_R, duration, n_way, training_data)
    # The negative sign is because optimizers minimize, and we want to maximize this score
    return -((fidelity ** 10) * (power_saved ** 0.1))

def learner_task(collected_data, n_way):
    global SIMULATION_STATE
    
    with SIMULATION_STATE["lock"]:
        SIMULATION_STATE["learner_status"] = "running"
    
    split_index = int(len(collected_data) * 0.8)
    train_set, test_set = collected_data[:split_index], collected_data[split_index:]

    if not len(train_set) or not len(test_set):
        print("Not enough collected data for train/test split. Aborting learner.")
        with SIMULATION_STATE["lock"]: SIMULATION_STATE["learner_status"] = "idle"
        return

    print(f"Learner started. Training on {len(train_set)}, testing on {len(test_set)}.")
    
    bounds = [(0.9, 1.0), (10.0, 200.0)]
    result = differential_evolution(objective_function, bounds, args=(train_set, n_way), maxiter=30, popsize=10, tol=0.02, disp=False)
    new_threshold, new_duration = result.x[0], int(round(result.x[1]))

    _, final_fidelity = run_simulation_for_learner(new_threshold, new_duration, n_way, test_set)

    with SIMULATION_STATE["lock"]:
        SIMULATION_STATE["threshold"] = new_threshold
        SIMULATION_STATE["duration"] = new_duration
        SIMULATION_STATE["last_learner_fidelity"] = final_fidelity
        SIMULATION_STATE["learner_status"] = "idle"
        # Reset shadow stats after retraining
        SIMULATION_STATE["shadow_fidelity_error"] = 0.0
        SIMULATION_STATE["shadow_fidelity_count"] = 0
        SIMULATION_STATE["shadow_samples"] = []
        print(f"Learner finished. New params deployed: T={new_threshold:.4f}, D={new_duration}. Tested Fidelity: {final_fidelity:.4f}")

def operator_loop():
    global learner_thread
    while True:
        time.sleep(0.05)
        with SIMULATION_STATE["lock"]:
            if not SIMULATION_STATE["is_running"]: break
            
            t = SIMULATION_STATE["timestep"]
            if SIMULATION_STATE["data"] is None or t >= len(SIMULATION_STATE["data"]) - 1:
                SIMULATION_STATE["is_running"] = False
                SIMULATION_STATE["current_phase"] = "finished"
                break

            # --- Hybrid Model Logic ---
            # 1. Check if a retrain should be triggered
            current_fidelity = 1.0
            if SIMULATION_STATE["shadow_fidelity_count"] > 0:
                mse = SIMULATION_STATE["shadow_fidelity_error"] / SIMULATION_STATE["shadow_fidelity_count"]
                current_fidelity = max(0, 1.0 - mse)
            
            time_since_last_retrain = t - SIMULATION_STATE["last_retrain_timestep"]
            
            is_learner_busy = learner_thread is not None and learner_thread.is_alive()
            
            # Trigger conditions
            trigger_by_fidelity = current_fidelity < SIMULATION_STATE["hybrid_fidelity_threshold"]
            trigger_by_interval = time_since_last_retrain > SIMULATION_STATE["hybrid_max_timesteps_since_retrain"]
            
            if (trigger_by_fidelity or trigger_by_interval) and not is_learner_busy and SIMULATION_STATE["current_phase"] != "collecting":
                SIMULATION_STATE["current_phase"] = "collecting"
                SIMULATION_STATE["last_retrain_timestep"] = t # Mark the start of the collection cycle
                print(f"Retrain triggered at timestep {t}. Reason: Fidelity drop ({trigger_by_fidelity}), Interval exceeded ({trigger_by_interval}).")

            # 2. Execute logic based on the current phase
            phase = SIMULATION_STATE["current_phase"]
            
            if phase == "collecting":
                for s in SIMULATION_STATE["sensors"]: s["is_off"] = False
                
                collection_progress = t - SIMULATION_STATE["last_retrain_timestep"]
                if collection_progress >= SIMULATION_STATE["collection_period"]:
                    # Collection finished, start learner and switch back to shadow mode
                    start_learn = max(0, t - SIMULATION_STATE["collection_period"] + 1)
                    data_chunk = SIMULATION_STATE["data"][start_learn:t+1]
                    n_way = SIMULATION_STATE["n_way_comparison"]
                    learner_thread = threading.Thread(target=learner_task, args=(data_chunk, n_way))
                    learner_thread.start()
                    SIMULATION_STATE["current_phase"] = "shadow_op"
                    SIMULATION_STATE["last_retrain_timestep"] = t # Update to mark end of this cycle
            
            elif phase == "shadow_op":
                # Standard shadow mode operation
                num_deactivated = sum(1 for s in SIMULATION_STATE["sensors"] if s["is_off"] and t < s["end_time"])
                for s in SIMULATION_STATE["sensors"]:
                    if s["is_off"] and t >= s["end_time"]: s["is_off"] = False

                readings = SIMULATION_STATE["data"][t]
                last_readings = SIMULATION_STATE["data"][t - 1] if t > 0 else readings
                
                active_sensors = [s for s in SIMULATION_STATE["sensors"] if not s["is_off"]]
                for s in active_sensors:
                    delta = readings[s["id"]] - last_readings[s["id"]]
                    s["noise_variance"] = 0.99 * s["noise_variance"] + 0.01 * (delta ** 2)

                if len(active_sensors) >= SIMULATION_STATE["n_way_comparison"]:
                    for combo in combinations(active_sensors, SIMULATION_STATE["n_way_comparison"]):
                        if generalized_redundancy_metric(np.array([readings[s["id"]] for s in combo]), SIMULATION_STATE["n_way_comparison"]) > SIMULATION_STATE["threshold"]:
                            noisiest_sensor = max(combo, key=lambda s: s["noise_variance"])
                            if not noisiest_sensor["is_off"]:
                                if np.random.rand() < SIMULATION_STATE["shadow_mode_probability"]:
                                    # Undercover quality check
                                    peer_readings = [readings[s["id"]] for s in combo if s["id"] != noisiest_sensor["id"]]
                                    estimated = np.mean(np.array(peer_readings)) if peer_readings else readings[noisiest_sensor["id"]]
                                    true = readings[noisiest_sensor["id"]]
                                    SIMULATION_STATE["shadow_fidelity_error"] += (true - estimated) ** 2
                                    SIMULATION_STATE["shadow_fidelity_count"] += 1
                                    SIMULATION_STATE["shadow_samples"].append((true, estimated))
                                else:
                                    # Normal deactivation
                                    noisiest_sensor["is_off"] = True
                                    noisiest_sensor["end_time"] = t + SIMULATION_STATE["duration"]
                                    num_deactivated += 1
                
                SIMULATION_STATE["total_power_saved_steps"] += num_deactivated
            
            SIMULATION_STATE["timestep"] += 1

# --- Data Loading & API Endpoints ---
@app.on_event("startup")
async def startup_event():
    def load_data():
        global SIMULATION_STATE
        try:
            # Try loading the new dataset first
            df = pd.read_csv('GlobalWeather.csv').iloc[:, :50]
            # Drop the date column if it exists, as it's not a sensor reading
            if 'date' in df.columns:
                df = df.drop(columns=['date'])
            raw_data = df.values
        except FileNotFoundError:
            try:
                # Fallback to the original dataset
                df_sensor = pd.read_csv('GlobalWeather.csv')
                raw_data = df_sensor.drop(columns=['date']).values
            except FileNotFoundError:
                print("No dataset found (cleaned_iot_data.csv or GlobalWeather.csv). Using dummy data.")
                raw_data = np.random.rand(20000, 30)
                 # Default to 10 sensors for dummy data
        
        # Normalize data
        min_vals, max_vals = raw_data.min(axis=0), raw_data.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-6
        normalized_data = (raw_data - min_vals) / range_vals
        
        # DYNAMICALLY configure the simulation state
        num_sensors = normalized_data.shape[1]
        print(f"Dataset loaded and normalized. Detected {num_sensors} sensors. Shape: {normalized_data.shape}")
        
        with SIMULATION_STATE["lock"]:
            SIMULATION_STATE["data"] = normalized_data
            SIMULATION_STATE["total_sensors"] = num_sensors
            SIMULATION_STATE["sensors"] = [{"id": i, "is_off": False, "master_id": -1, "end_time": -1, "noise_variance": 0} for i in range(num_sensors)]

    threading.Thread(target=load_data).start()

@app.get('/status')
async def get_status():
    with SIMULATION_STATE["lock"]:
        state_copy = {k: v for k, v in SIMULATION_STATE.items() if k not in ["lock", "data", "shadow_samples"]}
        
        if state_copy["shadow_fidelity_count"] > 0:
            mse = state_copy["shadow_fidelity_error"] / state_copy["shadow_fidelity_count"]
            state_copy["fidelity"] = max(0, 1.0 - mse)
        else:
            state_copy["fidelity"] = state_copy["last_learner_fidelity"]
        
        active_count = sum(1 for s in SIMULATION_STATE["sensors"] if not s["is_off"])
        t_step = state_copy["timestep"]
        total_sensors = state_copy["total_sensors"]
        total_possible = t_step * total_sensors if t_step > 0 else 1
        power_saved_percent = (state_copy["total_power_saved_steps"] / total_possible) * 100 if total_possible > 0 else 0
        
        response = {**state_copy}
        response["active_sensors"] = active_count
        response["power_saved_percent"] = power_saved_percent
        response["collected_shadow_samples"] = len(SIMULATION_STATE["shadow_samples"])
        
        current_readings = []
        if SIMULATION_STATE.get("data") is not None and t_step < len(SIMULATION_STATE["data"]):
            current_readings = SIMULATION_STATE["data"][t_step].tolist()
        response["current_readings"] = current_readings
        
        return response

@app.post('/start')
async def start_simulation(request: Request):
    global operator_thread, SIMULATION_STATE
    with SIMULATION_STATE["lock"]:
        if not SIMULATION_STATE["is_running"]:
            params = await request.json()
            
            # Preserve essential loaded data properties during reset
            loaded_data = SIMULATION_STATE.get("data")
            total_sensors = SIMULATION_STATE.get("total_sensors", 0)
            
            # Get a fresh default state
            SIMULATION_STATE = get_default_state()
            
            # Restore the essential properties
            SIMULATION_STATE["data"] = loaded_data
            SIMULATION_STATE["total_sensors"] = total_sensors
            if total_sensors > 0:
                SIMULATION_STATE["sensors"] = [{"id": i, "is_off": False, "master_id": -1, "end_time": -1, "noise_variance": 0} for i in range(total_sensors)]

            SIMULATION_STATE.update({
                "threshold": params.get('threshold', SIMULATION_STATE['threshold']),
                "duration": params.get('duration', SIMULATION_STATE['duration']),
                "n_way_comparison": params.get('n_way_comparison', SIMULATION_STATE['n_way_comparison']),
                "shadow_mode_probability": params.get('shadow_mode_probability', SIMULATION_STATE['shadow_mode_probability']),
                "hybrid_fidelity_threshold": params.get('hybrid_fidelity_threshold', SIMULATION_STATE['hybrid_fidelity_threshold']),
                "hybrid_max_timesteps_since_retrain": params.get('hybrid_max_timesteps_since_retrain', SIMULATION_STATE['hybrid_max_timesteps_since_retrain']),
                "is_running": True
            })
            
            if operator_thread is None or not operator_thread.is_alive():
                operator_thread = threading.Thread(target=operator_loop)
                operator_thread.start()
            
            print("Simulation started with Hybrid Model.")
    return {"message": "Simulation started"}

@app.post('/pause')
async def pause_simulation():
    with SIMULATION_STATE["lock"]:
        SIMULATION_STATE["is_running"] = False
    return {"message": "Simulation paused"}

@app.post('/reset')
async def reset_simulation():
    global operator_thread, SIMULATION_STATE
    with SIMULATION_STATE["lock"]:
        SIMULATION_STATE["is_running"] = False
        if operator_thread and operator_thread.is_alive():
            operator_thread.join(timeout=1.0)
        
        # Preserve essential loaded data properties
        loaded_data = SIMULATION_STATE.get("data")
        total_sensors = SIMULATION_STATE.get("total_sensors", 0)
        
        # Get a fresh default state
        SIMULATION_STATE = get_default_state()
        
        # Restore the essential properties
        SIMULATION_STATE["data"] = loaded_data
        SIMULATION_STATE["total_sensors"] = total_sensors
        if total_sensors > 0:
            SIMULATION_STATE["sensors"] = [{"id": i, "is_off": False, "master_id": -1, "end_time": -1, "noise_variance": 0} for i in range(total_sensors)]
    return {"message": "Simulation reset"}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
