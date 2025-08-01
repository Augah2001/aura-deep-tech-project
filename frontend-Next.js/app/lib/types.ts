// --- TYPE DEFINITIONS (TypeScript) ---

export interface Sensor {
    id: number;
    is_off: boolean;
}

export interface Status {
    is_running: boolean;
    timestep: number;
    current_phase: 'idle' | 'collecting' | 'shadow_op' | 'finished';
    active_sensors: number;
    total_sensors: number;
    power_saved_percent: number;
    fidelity: number;
    sensors: Sensor[];
    current_readings: number[];
    threshold: number;
    duration: number;
    n_way_comparison: number;
    shadow_mode_probability: number;
    learner_status: 'idle' | 'running';
    // Hybrid model params from backend
    hybrid_fidelity_threshold: number;
    hybrid_max_timesteps_since_retrain: number;
    last_retrain_timestep: number;
    collection_period: number;
}

export interface ChartDataPoint {
    timestep: number;
    fidelity: number;
    powerSaved: number;
}
