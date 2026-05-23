// --- TYPE DEFINITIONS (TypeScript) ---

export interface Sensor {
    id: number;
    is_off: boolean;
}

export interface SensorDetail {
    id: number;
    is_active: boolean;
    is_sleeping: boolean;
    is_anomaly: boolean;
    reading: number;
    estimated_reading: number;
    abs_error: number;
    decision_reason: string;
}

export interface PolicyMetrics {
    power_saved_pct: number;
    active_sensor_pct: number;
    anomaly_active_pct: number;
    anomaly_precision_pct: number;
    anomaly_recall_pct: number;
    anomaly_f1_pct: number;
    shared_global_mse: number;
    per_sensor_mse: number;
    global_reconstruction_mse: number;
}

export interface HardwareStatus {
    bridge_status: string;
    arduino_status: string;
    mode: string;
    com_port: string;
    baud_rate: number;
    active_nodes: number;
    sleeping_nodes: number;
    visible_nodes: number;
    last_sync: string | null;
    last_command: string;
    last_ack?: string | null;
    last_error?: string | null;
    note: string;
}

export interface KernelSpeedRow {
    backend: string;
    training_seconds: number;
    speedup_vs_pytorch: number;
    purpose: string;
}

export interface KernelProof {
    backend_mode: string;
    status: {
        extension_loaded: boolean;
        cuda_preferred: boolean;
        pair_cache_enabled: boolean;
        fused_cached_training_loss: boolean;
        manual_backward_enabled: boolean;
        cpu_fallback_available: boolean;
        live_training_seconds: number;
        fallback_reason?: string | null;
    };
    correctness: {
        forward_loss_parity: string;
        gradient_parity: string;
        loss_tolerance: string;
        gradient_tolerance: string;
        max_observed_loss_error: string;
        max_observed_gradient_error: string;
        reference: string;
    };
    speed: KernelSpeedRow[];
    architecture_steps: string[];
    note: string;
}

export interface Status {
    is_running: boolean;
    timestep: number;
    current_phase: 'idle' | 'collecting' | 'shadow_op' | 'finished' | 'error';
    active_sensors: number;
    total_sensors: number;
    power_saved_percent: number;
    fidelity: number;
    sensors: Sensor[];
    sensor_details?: SensorDetail[];
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
    algorithm?: string;
    backend_mode?: string;
    error?: string | null;
    metrics?: Record<string, number>;
    policy_metrics?: Record<string, PolicyMetrics>;
    training?: {
        seconds: number;
        final_loss: number;
        losses: number[];
        epochs: number;
    };
    runtime?: {
        elapsed_seconds: number;
        bench_sensors: number;
        bench_steps: number;
        bench_epochs: number;
        bench_max_pairs: number;
    };
    learned_parameters?: Record<string, number>;
    active_budget_band?: [number, number];
    hardware?: HardwareStatus;
    kernel_proof?: KernelProof;
    dataset?: DatasetSummary | null;
    history?: BenchmarkRun[];
    storage?: StorageStatus;
    diagnostics?: DiagnosticEntry[];
    status_transport?: string;
}

export interface ChartDataPoint {
    timestep: number;
    fidelity: number;
    powerSaved: number;
}

export interface DatasetSummary {
    id: number;
    filename: string;
    row_count: number;
    columns?: string[];
    numeric_columns?: string[];
    selected_columns?: string[];
    numeric_column_count?: number;
    uploaded_at_iso?: string;
}

export interface BenchmarkRun {
    id: number;
    started_at_iso?: string;
    finished_at_iso?: string;
    phase: string;
    backend_mode?: string;
    dataset_id?: number | null;
    selected_columns?: string[];
    metrics?: Record<string, number>;
    error?: string | null;
}

export interface DiagnosticEntry {
    time: string;
    severity: string;
    source: string;
    message: string;
}

export interface StorageStatus {
    backend: string;
    database_url_configured: boolean;
    sqlite_path?: string;
}
