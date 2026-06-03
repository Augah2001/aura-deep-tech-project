'use client';

import { Suspense, FC, ReactNode, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import {
    Activity,
    AlertTriangle,
    AreaChart,
    BrainCircuit,
    CheckCircle2,
    Cpu,
    Database,
    Pause,
    Play,
    Radar,
    Satellite,
    Send,
    Upload,
    Zap,
} from 'lucide-react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    Line,
    LineChart,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { API_BASE_URL, useStatus } from './hooks/useStatus';
import type { BenchmarkRun, DatasetSummary, DiagnosticEntry, PolicyMetrics, Status, TrendPoint } from './lib/types';

const FarmScene = dynamic(() => import('./components/FarmScene'), {
    ssr: false,
    loading: () => <div className="grid h-full place-items-center bg-[#030712] text-sm text-slate-400">Loading digital twin</div>,
});

const policyColors: Record<string, string> = {
    'Intelligent AURA': '#38BDF8',
    'LoRaWAN-style': '#F97316',
    'Budget-matched LoRaWAN': '#A78BFA',
    'NB-IoT-style': '#14B8A6',
};

const demoPresets = {
    fastCuda: {
        label: 'Fast CUDA Demo',
        description: 'CUDA/C++ when available.',
        payload: {
            BENCH_SENSORS: 500,
            BENCH_STEPS: 120,
            BENCH_EPOCHS: 20,
            BENCH_MAX_PAIRS: 10000,
            CELL8_FORCE_CPU: false,
            CELL8_SHOW_PLOTS: false,
        },
    },
    cpu: {
        label: 'CPU Demo',
        description: 'CPU backend.',
        payload: {
            BENCH_SENSORS: 500,
            BENCH_STEPS: 120,
            BENCH_EPOCHS: 20,
            BENCH_MAX_PAIRS: 10000,
            CELL8_FORCE_CPU: true,
            CELL8_SHOW_PLOTS: false,
        },
    },
    quick: {
        label: 'Quick Smoke',
        description: 'Small fast run.',
        payload: {
            BENCH_SENSORS: 80,
            BENCH_STEPS: 48,
            BENCH_EPOCHS: 4,
            BENCH_MAX_PAIRS: 1200,
            SAFETY_EPOCHS: 4,
            CELL8_FORCE_CPU: false,
            CELL8_SHOW_PLOTS: false,
        },
    },
    stress: {
        label: 'Stress Test',
        description: 'Larger benchmark.',
        payload: {
            BENCH_SENSORS: 800,
            BENCH_STEPS: 140,
            BENCH_EPOCHS: 20,
            BENCH_MAX_PAIRS: 14000,
            CELL8_FORCE_CPU: false,
            CELL8_SHOW_PLOTS: false,
        },
    },
} as const;

const backendModes = {
    autoCuda: {
        label: 'Auto CUDA/C++',
        shortLabel: 'CUDA/C++',
        description: 'Prefer CUDA/C++.',
        payload: {
            CELL8_FORCE_CPU: false,
            CELL8_USE_CPP_CUDA_PAIR_CACHE: true,
            CELL8_USE_CPP_CACHED_TRAINING_LOSS: true,
            CELL8_USE_MANUAL_CUDA_BACKWARD: true,
        },
    },
    cpuCpp: {
        label: 'Force C++ CPU',
        shortLabel: 'C++ CPU',
        description: 'Use native CPU.',
        payload: {
            CELL8_FORCE_CPU: true,
            CELL8_USE_CPP_CUDA_PAIR_CACHE: true,
            CELL8_USE_CPP_CACHED_TRAINING_LOSS: true,
            CELL8_USE_MANUAL_CUDA_BACKWARD: true,
        },
    },
    pythonReference: {
        label: 'Python/PyTorch',
        shortLabel: 'Python',
        description: 'Use reference path.',
        payload: {
            CELL8_FORCE_CPU: true,
            CELL8_USE_CPP_CUDA_PAIR_CACHE: false,
            CELL8_USE_CPP_CACHED_TRAINING_LOSS: false,
            CELL8_USE_MANUAL_CUDA_BACKWARD: false,
        },
    },
} as const;

type DemoPresetKey = keyof typeof demoPresets;
type BackendModeKey = keyof typeof backendModes;
type AppTab = 'live' | 'playground' | 'benchmarks' | 'data' | 'algorithm' | 'kernel' | 'hardware' | 'diagnostics';
type HardwareTarget = 'mega_simulation' | 'esp32_experiment';
type WhatIfControls = {
    recallPriority: number;
    powerPriority: number;
    activeBudgetPct: number;
    anomalyStrictness: number;
    shadowSamplePct: number;
    globalRetrainPct: number;
};
type ChallengeSettings = {
    reduceSensorsPct: number;
    injectAnomaly: boolean;
    simulateDrift: boolean;
    addNoise: boolean;
    removeSensor: boolean;
    increaseRedundancy: boolean;
};
type ChallengeToggleKey = Exclude<keyof ChallengeSettings, 'reduceSensorsPct'>;
type PanelistQuestionKey = 'balanced' | 'morePower' | 'moreRecall' | 'sensorFailure' | 'whyRetrain' | 'cudaFast';
type SerialPortOption = {
    device: string;
    description?: string;
    hwid?: string;
};

const scenarioPresets = {
    balanced: {
        label: 'Balanced',
        detail: 'Keeps the default power/recall compromise for general demonstrations.',
        controls: { recallPriority: 55, powerPriority: 55, activeBudgetPct: 25, anomalyStrictness: 55, shadowSamplePct: 5, globalRetrainPct: 100 },
        challenge: { reduceSensorsPct: 0, injectAnomaly: false, simulateDrift: false, addNoise: false, removeSensor: false, increaseRedundancy: false },
        preset: 'fastCuda',
    },
    highPower: {
        label: 'High Power Saving',
        detail: 'Uses a tighter active-sensor budget and stronger budget pressure.',
        controls: { recallPriority: 45, powerPriority: 88, activeBudgetPct: 10, anomalyStrictness: 50, shadowSamplePct: 5, globalRetrainPct: 100 },
        challenge: { reduceSensorsPct: 0, injectAnomaly: false, simulateDrift: false, addNoise: false, removeSensor: false, increaseRedundancy: true },
        preset: 'fastCuda',
    },
    highRecall: {
        label: 'High Recall',
        detail: 'Raises anomaly protection and keeps a larger active set.',
        controls: { recallPriority: 92, powerPriority: 35, activeBudgetPct: 35, anomalyStrictness: 82, shadowSamplePct: 10, globalRetrainPct: 70 },
        challenge: { reduceSensorsPct: 0, injectAnomaly: true, simulateDrift: false, addNoise: false, removeSensor: false, increaseRedundancy: false },
        preset: 'fastCuda',
    },
    drift: {
        label: 'Failure/Drift Scenario',
        detail: 'Injects drift and increases shadow sampling so retraining evidence appears faster.',
        controls: { recallPriority: 80, powerPriority: 45, activeBudgetPct: 28, anomalyStrictness: 88, shadowSamplePct: 20, globalRetrainPct: 35 },
        challenge: { reduceSensorsPct: 0, injectAnomaly: true, simulateDrift: true, addNoise: true, removeSensor: false, increaseRedundancy: false },
        preset: 'quick',
    },
    smartCity: {
        label: 'Dense Smart City',
        detail: 'Large redundant network with CUDA/C++ preferred for the defence demo.',
        controls: { recallPriority: 65, powerPriority: 78, activeBudgetPct: 18, anomalyStrictness: 62, shadowSamplePct: 10, globalRetrainPct: 80 },
        challenge: { reduceSensorsPct: 0, injectAnomaly: false, simulateDrift: false, addNoise: false, removeSensor: false, increaseRedundancy: true },
        preset: 'stress',
    },
    industrial: {
        label: 'Industrial Network',
        detail: 'Stricter anomaly tolerance and shadow checks for safety-critical monitoring.',
        controls: { recallPriority: 90, powerPriority: 50, activeBudgetPct: 30, anomalyStrictness: 90, shadowSamplePct: 15, globalRetrainPct: 55 },
        challenge: { reduceSensorsPct: 10, injectAnomaly: true, simulateDrift: false, addNoise: true, removeSensor: true, increaseRedundancy: false },
        preset: 'fastCuda',
    },
    environment: {
        label: 'Environmental Monitoring',
        detail: 'Moderate budget with slow drift checks for weather and pollution sensors.',
        controls: { recallPriority: 70, powerPriority: 70, activeBudgetPct: 22, anomalyStrictness: 65, shadowSamplePct: 5, globalRetrainPct: 90 },
        challenge: { reduceSensorsPct: 0, injectAnomaly: false, simulateDrift: true, addNoise: true, removeSensor: false, increaseRedundancy: true },
        preset: 'fastCuda',
    },
} as const;

type ScenarioPresetKey = keyof typeof scenarioPresets;

const defaultWhatIfControls: WhatIfControls = scenarioPresets.balanced.controls;
const defaultChallengeSettings: ChallengeSettings = scenarioPresets.balanced.challenge;

const panelistQuestions: Record<PanelistQuestionKey, {
    label: string;
    controls: Partial<WhatIfControls>;
    challenge: Partial<ChallengeSettings>;
    backend?: BackendModeKey;
}> = {
    balanced: {
        label: 'Show balanced behavior',
        controls: scenarioPresets.balanced.controls,
        challenge: scenarioPresets.balanced.challenge,
    },
    morePower: {
        label: 'Can AURA save more power?',
        controls: { powerPriority: 92, recallPriority: 45, activeBudgetPct: 10, shadowSamplePct: 5 },
        challenge: { increaseRedundancy: true },
    },
    moreRecall: {
        label: 'What if recall matters most?',
        controls: { recallPriority: 95, powerPriority: 35, activeBudgetPct: 36, anomalyStrictness: 88, shadowSamplePct: 10 },
        challenge: { injectAnomaly: true },
    },
    sensorFailure: {
        label: 'What if sensors fail?',
        controls: { recallPriority: 78, activeBudgetPct: 28, shadowSamplePct: 15 },
        challenge: { removeSensor: true, addNoise: true, reduceSensorsPct: 15 },
    },
    whyRetrain: {
        label: 'Why retrain?',
        controls: { recallPriority: 82, anomalyStrictness: 90, shadowSamplePct: 20, globalRetrainPct: 35 },
        challenge: { injectAnomaly: true, simulateDrift: true, addNoise: true },
    },
    cudaFast: {
        label: 'Is CUDA faster?',
        controls: { activeBudgetPct: 25, recallPriority: 60, powerPriority: 60 },
        challenge: {},
        backend: 'autoCuda',
    },
};

const challengeDisplay: { key: ChallengeToggleKey; label: string }[] = [
    { key: 'injectAnomaly', label: 'Anomaly' },
    { key: 'simulateDrift', label: 'Drift' },
    { key: 'addNoise', label: 'Noise' },
    { key: 'removeSensor', label: 'Sensor off' },
    { key: 'increaseRedundancy', label: 'Redundancy' },
];

const appTabs: { key: AppTab; label: string; technicalOnly?: boolean }[] = [
    { key: 'live', label: 'Live' },
    { key: 'playground', label: 'Playground' },
    { key: 'benchmarks', label: 'Benchmarks' },
    { key: 'algorithm', label: 'Algorithm' },
    { key: 'kernel', label: 'Kernel' },
    { key: 'hardware', label: 'Hardware' },
    { key: 'data', label: 'Data', technicalOnly: true },
    { key: 'diagnostics', label: 'Diagnostics', technicalOnly: true },
];

const getInitialTab = (): AppTab => {
    if (typeof window === 'undefined') return 'live';
    const requested = new URLSearchParams(window.location.search).get('tab') as AppTab | null;
    return requested && appTabs.some(tab => tab.key === requested) ? requested : 'live';
};

const demoSteps = [
    {
        title: 'Train AURA',
        detail: 'Train the policy.',
    },
    {
        title: 'Replay Prototype',
        detail: 'Replay sensor states.',
    },
    {
        title: 'Inspect Decision',
        detail: 'Inspect one sensor.',
    },
    {
        title: 'Compare Baselines',
        detail: 'Compare baselines.',
    },
    {
        title: 'Review Evidence',
        detail: 'Review benchmark evidence.',
    },
] as const;

const fmt = (value?: number, digits = 2) => Number.isFinite(value) ? Number(value).toFixed(digits) : '0.00';
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const displayBackendMode = (mode?: string) => {
    if (!mode) return 'Pending';
    return mode.replace(/\s+preferred$/i, '');
};

const replaySpeedOptions = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2];

const normalizeReplaySpeed = (speed?: number) => {
    const reported = speed || 1;
    return replaySpeedOptions.some(item => Math.abs(item - reported) < 0.01) ? reported : 1;
};

const controlsToPayload = (
    controls: WhatIfControls,
    challenge: ChallengeSettings,
    baseSensors: number,
) => {
    const redundancyMode = challenge.increaseRedundancy;
    const activeFraction = Math.min(
        0.75,
        Math.max(0.05, (redundancyMode ? Math.min(controls.activeBudgetPct, 14) : controls.activeBudgetPct) / 100),
    );
    const strictness = controls.anomalyStrictness / 100;
    const recallWeight = 1.5 + (controls.recallPriority / 100) * 7.5;
    const budgetWeight = 2.0 + (controls.powerPriority / 100) * 9.0 + (redundancyMode ? 4.0 : 0.0);
    const reducedSensors = Math.max(12, Math.round(baseSensors * (1 - challenge.reduceSensorsPct / 100)) - (challenge.removeSensor ? 1 : 0));
    const gateThreshold = Math.max(0.045, 0.16 - strictness * 0.09);
    const gateSharpness = 40 + strictness * 70;
    return {
        BENCH_SENSORS: reducedSensors,
        AURA_MIN_ACTIVE_FRACTION: Math.max(0.03, activeFraction - 0.04),
        AURA_MAX_ACTIVE_FRACTION: Math.min(0.90, activeFraction + 0.04),
        AURA_BUDGET_BAND_WEIGHT: budgetWeight,
        SAFETY_ANOMALY_WEIGHT: recallWeight,
        AURA_GATE_THRESHOLD: gateThreshold,
        AURA_GATE_SHARPNESS: gateSharpness,
        AURA_SHADOW_SAMPLE_RATE: controls.shadowSamplePct / 100,
        AURA_SHADOW_MSE_THRESHOLD: Math.max(0.004, 0.045 - strictness * 0.032),
        AURA_GLOBAL_RETRAIN_PERIOD_FRACTION: Math.max(0.05, controls.globalRetrainPct / 100),
        AURA_SYNTHETIC_NOISE_STD: challenge.addNoise ? 0.045 : 0.0,
        AURA_SYNTHETIC_DRIFT_STRENGTH: challenge.simulateDrift ? 0.22 : 0.0,
        AURA_SYNTHETIC_ANOMALY_EVENTS: challenge.injectAnomaly ? 34 : (redundancyMode ? 8 : 18),
        AURA_REDUNDANT_CLUSTER_STRENGTH: challenge.increaseRedundancy ? 0.92 : 0.0,
        AURA_REDUNDANT_GROUP_ANOMALIES: challenge.increaseRedundancy && !challenge.injectAnomaly,
        AURA_REDUNDANCY_REPRESENTATIVE_GUARD: challenge.increaseRedundancy,
        AURA_REDUNDANCY_GROUP_SIZE: 18,
    };
};

const Panel: FC<{ title?: string; action?: ReactNode; children: ReactNode; className?: string }> = ({ title, action, children, className = '' }) => (
    <section className={`rounded-lg border border-[#223044] bg-[#0E1520] shadow-[0_18px_45px_rgba(0,0,0,0.25)] ${className}`}>
        {(title || action) && (
            <div className="flex items-center justify-between gap-3 border-b border-[#223044] px-4 py-3">
                {title && <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[#8EA3B8]">{title}</h2>}
                {action}
            </div>
        )}
        {children}
    </section>
);

const EmptyState: FC<{ title: string; detail: string }> = ({ title, detail }) => (
    <div className="grid min-h-[180px] place-items-center p-6 text-center">
        <div>
            <div className="text-sm font-semibold text-[#E6EDF5]">{title}</div>
            <p className="mt-2 max-w-md text-sm leading-6 text-[#8EA3B8]">{detail}</p>
        </div>
    </div>
);

const TelemetryCard: FC<{ label: string; value: ReactNode; icon: ReactNode; accent: string; sub?: ReactNode }> = ({ label, value, icon, accent, sub }) => (
    <div className="rounded-lg border border-[#223044] bg-[#121C2A] p-3">
        <div className="mb-2 flex items-center justify-between text-[#8EA3B8]">
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em]">{label}</span>
            <span style={{ color: accent }}>{icon}</span>
        </div>
        <div className="font-mono text-2xl font-semibold tabular-nums text-[#E6EDF5]" style={{ color: accent }}>{value}</div>
        {sub && <div className="mt-2 text-xs text-[#8EA3B8]">{sub}</div>}
    </div>
);

const BenchmarkTable: FC<{ policies: Record<string, PolicyMetrics> }> = ({ policies }) => {
    const rows = [
        ['Power saved', 'power_saved_pct', '%'],
        ['Bandwidth saved', 'power_saved_pct', '%'],
        ['Active sensors', 'active_sensor_pct', '%'],
        ['Anomaly recall', 'anomaly_recall_pct', '%'],
        ['Global MSE (lower)', 'global_reconstruction_mse', ''],
    ] as const;
    const names = Object.keys(policies);

    if (!names.length) {
        return (
            <EmptyState
                title="No benchmark yet"
                detail="Press Start."
            />
        );
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] border-collapse text-left text-sm">
                <thead>
                    <tr className="border-b border-[#223044] text-xs uppercase tracking-[0.12em] text-[#8EA3B8]">
                        <th className="px-4 py-3 font-semibold">Metric</th>
                        {names.map(name => <th key={name} className="px-4 py-3 font-semibold">{name}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {rows.map(([label, key, unit]) => (
                        <tr key={label} className="border-b border-[#182335] last:border-b-0">
                            <td className="px-4 py-3 text-[#8EA3B8]">{label}</td>
                            {names.map(name => (
                                <td key={name} className="px-4 py-3 font-mono tabular-nums text-[#E6EDF5]">
                                    {fmt(policies[name]?.[key], key.includes('mse') ? 6 : 2)}{unit}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

const PresentationImpactStrip: FC<{ status: Status; policies: Record<string, PolicyMetrics> }> = ({ status, policies }) => {
    const aura = policies['Intelligent AURA'];
    const sleepingNodes = Math.max(0, status.total_sensors - status.active_sensors);
    const bandwidthSaved = status.power_saved_percent;
    const recall = aura?.anomaly_recall_pct ?? status.metrics?.anomaly_recall_pct;

    return (
        <section className="rounded-lg border border-[#22C55E]/35 bg-[#0E1520] shadow-[0_18px_45px_rgba(0,0,0,0.25)]">
            <div className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1.25fr)_repeat(4,minmax(150px,0.55fr))]">
                <div className="rounded-md border border-[#22C55E]/30 bg-[#22C55E]/8 p-4">
                    <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-[#86EFAC]">AURA Result</div>
                    <div className="text-xl font-semibold leading-8 text-[#E6EDF5]">
                        <span className="text-[#22C55E]">{sleepingNodes}</span> / <span className="text-[#38BDF8]">{status.total_sensors}</span> sensors suppressed.
                    </div>
                </div>
                <TelemetryCard label="Power Saved" value={`${fmt(status.power_saved_percent)}%`} icon={<Zap size={18} />} accent="#22C55E" />
                <TelemetryCard label="Bandwidth Saved" value={`${fmt(bandwidthSaved)}%`} icon={<Activity size={18} />} accent="#14B8A6" />
                <TelemetryCard label="Anomaly Recall" value={`${fmt(recall)}%`} icon={<Radar size={18} />} accent="#F59E0B" />
                <TelemetryCard label="MSE" value={fmt(aura?.global_reconstruction_mse, 5)} icon={<BrainCircuit size={18} />} accent="#A78BFA" />
            </div>
        </section>
    );
};

const PolicyFaceoff: FC<{ policies: Record<string, PolicyMetrics>; status: Status }> = ({ policies, status }) => {
    const liveActivePct = status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : 0;
    const replayProgress = status.current_phase === 'finished'
        ? 1
        : Math.max(0, Math.min(1, (status.replay_progress_pct || 0) / 100));
    const aura = policies['Intelligent AURA']
        ? {
            ...policies['Intelligent AURA'],
            power_saved_pct: status.power_saved_percent,
            active_sensor_pct: liveActivePct,
        }
        : undefined;
    const baselines = [
        ['LoRaWAN-style', policies['LoRaWAN-style'], '#FDBA74'],
        ['Budgeted LoRaWAN', policies['Budget-matched LoRaWAN'], '#2DD4BF'],
    ].filter(([, metrics]) => metrics)
        .map(([name, metrics, color]) => [
            name,
            {
                ...(metrics as PolicyMetrics),
                power_saved_pct: ((metrics as PolicyMetrics).power_saved_pct || 0) * replayProgress,
            },
            color,
        ]) as [string, PolicyMetrics, string][];
    const rows: [string, (metrics?: PolicyMetrics) => string, string][] = [
        ['Power', (metrics?: PolicyMetrics) => fmt(metrics?.power_saved_pct), '%'],
        ['Bandwidth', (metrics?: PolicyMetrics) => fmt(metrics?.power_saved_pct), '%'],
        ['Recall', (metrics?: PolicyMetrics) => fmt(metrics?.anomaly_recall_pct), '%'],
        ['MSE', (metrics?: PolicyMetrics) => fmt(metrics?.global_reconstruction_mse, 5), ''],
    ];

    if (!aura || !baselines.length) {
        return <EmptyState title="Comparison pending" detail="Run a benchmark." />;
    }

    return (
        <div className="grid gap-3 p-4">
            <div className={`grid gap-3 ${baselines.length > 1 ? 'grid-cols-3' : 'grid-cols-2'}`}>
                <div className="rounded-md border border-[#38BDF8]/45 bg-[#38BDF8]/10 p-3">
                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[#7DD3FC]">Intelligent AURA</div>
                </div>
                {baselines.map(([name, , color]) => (
                    <div key={name} className="rounded-md border border-[#F97316]/45 bg-[#F97316]/10 p-3">
                        <div className="text-xs font-semibold uppercase tracking-[0.16em]" style={{ color }}>{name}</div>
                    </div>
                ))}
            </div>
            {rows.map(([label, formatValue, unit]) => (
                <div key={label} className={`grid items-center gap-2 rounded-md bg-[#0A111C] px-3 py-2 ${baselines.length > 1 ? 'grid-cols-[92px_1fr_1fr_1fr]' : 'grid-cols-[92px_1fr_1fr]'}`}>
                    <div className="text-xs uppercase tracking-[0.12em] text-[#5F7288]">{label}</div>
                    <div className="font-mono text-sm font-semibold text-[#38BDF8]">{formatValue(aura)}{unit}</div>
                    {baselines.map(([name, metrics, color]) => (
                        <div key={name} className="font-mono text-sm font-semibold" style={{ color }}>{formatValue(metrics)}{unit}</div>
                    ))}
                </div>
            ))}
        </div>
    );
};

const ArduinoCommandStream: FC<{ status: Status; startPin?: number; nodeCount?: number }> = ({ status, startPin = 22, nodeCount = 28 }) => {
    const hardware = status.hardware;
    const command = hardware?.last_command || 'pending';
    const ack = hardware?.last_ack || (hardware?.arduino_status === 'connected' ? 'waiting for acknowledgement' : 'preview mode');
    const commandBits = command === 'pending' ? [] : command.split(',').slice(0, nodeCount);

    return (
        <div className="grid gap-3 p-4">
            <div className="rounded-md border border-[#223044] bg-[#050A12] p-3 font-mono text-xs leading-6">
                <div className="text-[#38BDF8]">AURA -&gt; Hardware: <span className="break-all text-[#22C55E]">{command}</span></div>
                <div className="text-[#A78BFA]">Hardware -&gt; AURA: <span className="text-[#E6EDF5]">{ack}</span></div>
            </div>
            <div className="grid grid-cols-7 gap-1.5">
                {commandBits.map((bit, index) => {
                    const active = bit.trim() === '0';
                    return (
                        <div key={`${index}-${bit}`} className={`rounded border px-1.5 py-1 text-center font-mono text-[10px] ${active ? 'border-[#38BDF8]/50 bg-[#38BDF8]/15 text-[#7DD3FC]' : 'border-[#64748B]/35 bg-[#64748B]/10 text-[#94A3B8]'}`}>
                            <div>{startPin + index}</div>
                            <div className="text-[9px]">{active ? 'ON' : 'OFF'}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

const ReplaySpeedControl: FC<{ speed: number; disabled: boolean; onChange: (speed: number) => void }> = ({ speed, disabled, onChange }) => {
    const speeds = replaySpeedOptions;
    return (
        <label className="flex items-center gap-2 rounded-md border border-[#223044] bg-[#0A111C] px-2 py-1 text-xs font-semibold text-[#8EA3B8]">
            <span className="hidden uppercase tracking-[0.12em] md:inline">Speed</span>
            <select
                value={normalizeReplaySpeed(speed)}
                onChange={event => onChange(Number(event.target.value))}
                disabled={disabled}
                className="rounded bg-[#121C2A] px-2 py-1.5 font-mono text-xs text-[#E6EDF5] outline-none disabled:cursor-not-allowed disabled:opacity-50"
            >
                {speeds.map(item => (
                    <option key={item} value={item}>{item}x</option>
                ))}
            </select>
        </label>
    );
};

const LiveSensorStream: FC<{ status: Status }> = ({ status }) => {
    const rows = (status.sensor_details || []).slice(0, 8);
    if (!rows.length) return <EmptyState title="No live readings" detail="Run AURA." />;

    return (
        <div className="overflow-x-auto p-4">
            <table className="w-full min-w-[560px] border-collapse text-left text-xs">
                <thead>
                    <tr className="border-b border-[#223044] uppercase tracking-[0.12em] text-[#8EA3B8]">
                        <th className="px-2 py-2 font-semibold">Node</th>
                        <th className="px-2 py-2 font-semibold">State</th>
                        <th className="px-2 py-2 font-semibold">Reading</th>
                        <th className="px-2 py-2 font-semibold">Estimate</th>
                        <th className="px-2 py-2 font-semibold">Error</th>
                        <th className="px-2 py-2 font-semibold">Anomaly</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map(row => (
                        <tr key={row.id} className="border-b border-[#182335] last:border-b-0">
                            <td className="px-2 py-2 font-mono text-[#E6EDF5]">#{row.id + 1}</td>
                            <td className={`px-2 py-2 font-semibold ${row.is_shadow ? 'text-[#A78BFA]' : row.is_active ? 'text-[#38BDF8]' : 'text-[#94A3B8]'}`}>{row.is_shadow ? 'SHADOW' : row.is_active ? 'ACTIVE' : 'SLEEP'}</td>
                            <td className="px-2 py-2 font-mono text-[#B8C7D8]">{fmt(row.reading, 4)}</td>
                            <td className="px-2 py-2 font-mono text-[#B8C7D8]">{fmt(row.estimated_reading, 4)}</td>
                            <td className="px-2 py-2 font-mono text-[#A78BFA]">{fmt(row.abs_error, 4)}</td>
                            <td className={`px-2 py-2 font-semibold ${row.is_anomaly ? 'text-[#F59E0B]' : 'text-[#5F7288]'}`}>{row.is_anomaly ? 'YES' : 'NO'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

const LiveEventLog: FC<{ status: Status }> = ({ status }) => {
    const events = status.live_events?.length ? status.live_events.slice(-8).reverse() : [];
    if (!events.length) return <EmptyState title="No live events" detail="Start a run." />;
    return (
        <div className="grid gap-2 p-4">
            {events.map((event, index) => (
                <div key={`${event.time}-${index}`} className="grid grid-cols-[72px_82px_1fr] gap-2 rounded-md bg-[#0A111C] px-3 py-2 text-xs">
                    <span className="font-mono text-[#5F7288]">{event.time}</span>
                    <span className="font-semibold uppercase tracking-[0.12em] text-[#38BDF8]">{event.source}</span>
                    <span className="text-[#B8C7D8]">{event.message}</span>
                </div>
            ))}
        </div>
    );
};

const AnomalyAlert: FC<{ status: Status }> = ({ status }) => {
    const anomalyRows = (status.sensor_details || []).filter(sensor => sensor.is_anomaly);
    if (!anomalyRows.length && !status.active_anomalies) return null;
    return (
        <div className="absolute left-4 top-4 z-10 max-w-[calc(100%-8rem)] rounded-md border border-[#F59E0B]/55 bg-[#1F1606]/90 px-3 py-2 shadow-[0_12px_30px_rgba(0,0,0,0.35)] backdrop-blur">
            <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2 text-xs font-semibold text-[#FDE68A]">
                    <AlertTriangle size={16} />
                    Anomaly event active
                </div>
                <div className="font-mono text-[11px] text-[#FDE68A]">
                    {status.active_anomalies || 0} active anomaly nodes | {anomalyRows.length} visible anomaly nodes
                </div>
            </div>
        </div>
    );
};

const QualityGuardrails: FC<{ status: Status; policies: Record<string, PolicyMetrics> }> = ({ status, policies }) => {
    const aura = policies['Intelligent AURA'];
    const mse = aura?.global_reconstruction_mse ?? status.metrics?.global_reconstruction_mse ?? 0;
    const recall = aura?.anomaly_recall_pct ?? status.metrics?.anomaly_recall_pct ?? 0;
    const powerSaved = aura?.power_saved_pct ?? status.power_saved_percent;
    const activePct = aura?.active_sensor_pct ?? (status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : 0);
    const budget = status.active_budget_band || [20, 30];
    const checks = [
        {
            label: 'Power objective',
            value: `${fmt(powerSaved)}% saved`,
            pass: powerSaved >= 50,
        },
        {
            label: 'Anomaly guardrail',
            value: `${fmt(recall)}% recall`,
            pass: recall >= 80,
        },
        {
            label: 'Fidelity guardrail',
            value: `MSE ${fmt(mse, 5)}`,
            pass: mse <= 0.01,
        },
        {
            label: 'Active budget',
            value: `${fmt(activePct)}% active`,
            pass: activePct >= budget[0] - 5 && activePct <= budget[1] + 5,
        },
    ];

    return (
        <div className="grid gap-2 p-4">
            {checks.map(check => (
                <div key={check.label} className={`flex items-center justify-between gap-3 rounded-md border px-3 py-2 ${check.pass ? 'border-[#22C55E]/35 bg-[#22C55E]/8' : 'border-[#F59E0B]/35 bg-[#F59E0B]/10'}`}>
                    <div className="flex items-center gap-2">
                        {check.pass ? <CheckCircle2 size={16} className="text-[#22C55E]" /> : <AlertTriangle size={16} className="text-[#F59E0B]" />}
                        <span className="text-sm font-semibold text-[#E6EDF5]">{check.label}</span>
                    </div>
                    <span className="font-mono text-xs text-[#B8C7D8]">{check.value}</span>
                </div>
            ))}
        </div>
    );
};

type RuntimeComparisonRow = {
    mode: string;
    runtimeSeconds: number;
    trainingSeconds: number;
    speedup: number;
    powerSaved: number;
    recall: number;
    mse: number;
    backend: string;
    purpose?: string;
};

const RuntimeComparisonPanel: FC<{
    rows: RuntimeComparisonRow[];
    isRunning: boolean;
    onRun: () => void;
}> = ({ rows, isRunning, onRun }) => (
    <div className="grid gap-4 p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <p className="text-sm leading-6 text-[#8EA3B8]">Runs an extended acceleration stress workload: 128 sensors, 2,000 timesteps, 20 training epochs, and 10,000 pair comparisons.</p>
            <button
                onClick={onRun}
                disabled={isRunning}
                className="rounded-md border border-[#38BDF8]/45 bg-[#38BDF8]/10 px-4 py-2 text-sm font-semibold text-[#7DD3FC] transition hover:bg-[#38BDF8]/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
                {isRunning ? 'Running...' : 'Run acceleration comparison'}
            </button>
        </div>
        {rows.length ? (
            <div className="overflow-x-auto rounded-md border border-[#223044]">
                <table className="w-full min-w-[760px] border-collapse text-left text-sm">
                    <thead>
                        <tr className="border-b border-[#223044] text-xs uppercase tracking-[0.12em] text-[#8EA3B8]">
                            <th className="px-4 py-3 font-semibold">Backend</th>
                            <th className="px-4 py-3 font-semibold">Training Time</th>
                            <th className="px-4 py-3 font-semibold">Speedup</th>
                            <th className="px-4 py-3 font-semibold">Purpose</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map(row => (
                            <tr key={row.mode} className="border-b border-[#182335] last:border-b-0">
                                <td className="px-4 py-3 font-semibold text-[#E6EDF5]">{row.mode}</td>
                                <td className="px-4 py-3 font-mono text-[#E6EDF5]">{fmt(row.trainingSeconds, 3)}s</td>
                                <td className="px-4 py-3 font-mono text-[#22C55E]">{fmt(row.speedup, 2)}x</td>
                                <td className="px-4 py-3 text-[#9CC7F5]">{row.purpose || row.backend}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        ) : (
            <div className="rounded-md border border-[#223044] bg-[#0A111C] px-4 py-5 text-center">
                <div className="text-sm font-semibold text-[#E6EDF5]">No runtime comparison yet</div>
                <p className="mt-1 text-sm text-[#8EA3B8]">Run the dedicated acceleration comparison to fill this table.</p>
            </div>
        )}
    </div>
);

const SliderControl: FC<{
    label: string;
    value: number;
    min?: number;
    max?: number;
    step?: number;
    suffix?: string;
    detail: string;
    onChange: (value: number) => void;
}> = ({ label, value, min = 0, max = 100, step = 1, suffix = '', detail, onChange }) => (
    <label className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-3">
        <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">{label}</span>
            <span className="font-mono text-sm font-semibold text-[#7DD3FC]">{value}{suffix}</span>
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={event => onChange(Number(event.target.value))}
            className="w-full accent-[#38BDF8]"
        />
        <p className="mt-2 text-xs leading-5 text-[#5F7288]">{detail}</p>
    </label>
);

const WhatIfControlsPanel: FC<{
    controls: WhatIfControls;
    onChange: (controls: WhatIfControls) => void;
}> = ({ controls, onChange }) => {
    const update = (key: keyof WhatIfControls, value: number) => onChange({ ...controls, [key]: value });
    return (
        <div className="grid gap-3 p-4 md:grid-cols-2">
            <SliderControl label="Recall priority" value={controls.recallPriority} detail="Higher values keep anomaly-sensitive sensors active." onChange={value => update('recallPriority', value)} />
            <SliderControl label="Power priority" value={controls.powerPriority} detail="Higher values push AURA toward more sleeping sensors." onChange={value => update('powerPriority', value)} />
            <SliderControl label="Active sensor budget" value={controls.activeBudgetPct} min={5} max={60} suffix="%" detail="Try 10% to test a strict active-node cap." onChange={value => update('activeBudgetPct', value)} />
            <SliderControl label="Anomaly strictness" value={controls.anomalyStrictness} detail="Stricter gates make AURA more cautious around unusual readings." onChange={value => update('anomalyStrictness', value)} />
            <SliderControl label="Shadow sample" value={controls.shadowSamplePct} min={0} max={25} suffix="%" detail="Controls the silent validation sample: 5%, 10%, 20%, etc." onChange={value => update('shadowSamplePct', value)} />
            <SliderControl label="Global retrain period" value={controls.globalRetrainPct} min={10} max={150} suffix="%" detail="Relative to the current dataset window." onChange={value => update('globalRetrainPct', value)} />
        </div>
    );
};

const ScenarioPresetPanel: FC<{
    selected: ScenarioPresetKey;
    onSelect: (key: ScenarioPresetKey) => void;
}> = ({ selected, onSelect }) => (
    <div className="grid gap-3 p-4 md:grid-cols-2 xl:grid-cols-3">
        {(Object.keys(scenarioPresets) as ScenarioPresetKey[]).map(key => {
            const preset = scenarioPresets[key];
            const active = selected === key;
            return (
                <button
                    key={key}
                    onClick={() => onSelect(key)}
                    className={`rounded-md border px-3 py-3 text-left transition ${
                        active
                            ? 'border-[#38BDF8] bg-[#38BDF8]/15'
                            : 'border-[#223044] bg-[#0A111C] hover:border-[#64748B]'
                    }`}
                >
                    <div className={`text-sm font-semibold ${active ? 'text-[#7DD3FC]' : 'text-[#E6EDF5]'}`}>{preset.label}</div>
                    <p className="mt-2 text-xs leading-5 text-[#8EA3B8]">{preset.detail}</p>
                </button>
            );
        })}
    </div>
);

const ChallengePanel: FC<{
    settings: ChallengeSettings;
    backendMode: BackendModeKey;
    onSettings: (settings: ChallengeSettings) => void;
    onBackendMode: (mode: BackendModeKey) => void;
}> = ({ settings, backendMode, onSettings, onBackendMode }) => {
    const toggle = (key: ChallengeToggleKey) => onSettings({ ...settings, [key]: !settings[key] });
    const items: { key: ChallengeToggleKey; label: string; detail: string }[] = [
        { key: 'injectAnomaly', label: 'Inject anomaly', detail: 'Adds extra anomaly events.' },
        { key: 'simulateDrift', label: 'Simulate drift', detail: 'Gradually shifts sensor readings.' },
        { key: 'addNoise', label: 'Add noise', detail: 'Raises measurement uncertainty.' },
        { key: 'removeSensor', label: 'Remove sensor', detail: 'Drops one available node.' },
        { key: 'increaseRedundancy', label: 'Increase redundancy', detail: 'Creates dense similar sensor clusters.' },
    ];
    return (
        <div className="grid gap-4 p-4">
            <SliderControl
                label="Reduce available sensors"
                value={settings.reduceSensorsPct}
                min={0}
                max={70}
                suffix="%"
                detail="Shrinks the run before AURA starts."
                onChange={value => onSettings({ ...settings, reduceSensorsPct: value })}
            />
            <div className="grid gap-2 md:grid-cols-2">
                {items.map(item => (
                    <button
                        key={item.key}
                        onClick={() => toggle(item.key)}
                        className={`rounded-md border px-3 py-3 text-left transition ${
                            settings[item.key]
                                ? 'border-[#F59E0B]/55 bg-[#F59E0B]/12'
                                : 'border-[#223044] bg-[#0A111C] hover:border-[#64748B]'
                        }`}
                    >
                        <div className={`text-sm font-semibold ${settings[item.key] ? 'text-[#FDE68A]' : 'text-[#E6EDF5]'}`}>{item.label}</div>
                        <p className="mt-1 text-xs leading-5 text-[#8EA3B8]">{item.detail}</p>
                    </button>
                ))}
            </div>
            <div className="rounded-md border border-[#223044] bg-[#0A111C] p-3">
                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Backend challenge</div>
                <div className="flex flex-wrap gap-2">
                    {(Object.keys(backendModes) as BackendModeKey[]).map(key => (
                        <button
                            key={key}
                            onClick={() => onBackendMode(key)}
                            className={`rounded px-3 py-2 text-xs font-semibold transition ${
                                backendMode === key ? 'bg-[#38BDF8] text-[#03111C]' : 'bg-[#121C2A] text-[#8EA3B8] hover:text-[#E6EDF5]'
                            }`}
                        >
                            {backendModes[key].shortLabel}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

const AuraAssistantPanel: FC<{ status: Status }> = ({ status }) => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState<{ mode: string; summary: string; bullets: string[] } | null>(null);
    const [loading, setLoading] = useState(false);

    const ask = async (prompt = question) => {
        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/assistant/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: prompt, status }),
            });
            if (!response.ok) throw new Error('assistant request failed');
            setAnswer(await response.json());
        } catch (error) {
            const aura = status.policy_metrics?.['Intelligent AURA'];
            setAnswer({
                mode: 'browser_fallback',
                summary: `AURA is saving ${fmt(aura?.power_saved_pct ?? status.power_saved_percent)}% power with ${fmt(aura?.anomaly_recall_pct ?? status.metrics?.anomaly_recall_pct)}% anomaly recall. Shadow mode currently samples ${fmt((status.shadow_validation?.sample_rate || 0) * 100, 1)}% of sleeping decisions.`,
                bullets: ['The backend assistant endpoint was not reachable.', 'The displayed explanation is generated from the current dashboard status.', 'Run a challenge or adjust what-if controls, then ask again.'],
            });
        } finally {
            setLoading(false);
        }
    };

    const prompts = ['Explain this run', 'Why did AURA sleep sensors?', 'Should retraining happen?', 'Compare with baselines'];
    return (
        <div className="grid gap-3 p-4">
            <div className="flex flex-wrap gap-2">
                {prompts.map(prompt => (
                    <button key={prompt} onClick={() => { setQuestion(prompt); void ask(prompt); }} className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-2 text-xs font-semibold text-[#8EA3B8] hover:border-[#64748B] hover:text-[#E6EDF5]">
                        {prompt}
                    </button>
                ))}
            </div>
            <div className="flex gap-2">
                <input
                    value={question}
                    onChange={event => setQuestion(event.target.value)}
                    placeholder="Ask AURA why a result changed..."
                    className="min-w-0 flex-1 rounded-md border border-[#223044] bg-[#050A12] px-3 py-2 text-sm text-[#E6EDF5] outline-none focus:border-[#38BDF8]"
                />
                <button onClick={() => ask()} disabled={loading} className="inline-flex items-center gap-2 rounded-md bg-[#38BDF8] px-3 py-2 text-sm font-semibold text-[#03111C] disabled:opacity-50">
                    <Send size={15} /> {loading ? 'Thinking' : 'Ask'}
                </button>
            </div>
            <div className="rounded-md border border-[#223044] bg-[#0A111C] p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">AURA Assistant</span>
                    <span className="rounded bg-[#121C2A] px-2 py-1 font-mono text-[10px] uppercase text-[#5F7288]">{answer?.mode || 'ready'}</span>
                </div>
                <p className="text-sm leading-6 text-[#B8C7D8]">{answer?.summary || 'Ask a question or run a scenario. The assistant explains only from live AURA metrics and shadow/retraining evidence.'}</p>
                {!!answer?.bullets?.length && (
                    <div className="mt-3 grid gap-2">
                        {answer.bullets.map((item, index) => (
                            <div key={`${index}-${item}`} className="rounded bg-[#050A12] px-3 py-2 text-xs leading-5 text-[#8EA3B8]">{item}</div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

const PlaygroundRunPanel: FC<{
    status: Status;
    scenarioLabel: string;
    backendLabel: string;
    selectedPresetLabel: string;
    runtimeComparisonRunning: boolean;
    onRun: () => void;
    onRerun: () => void;
    onReset: () => void;
    onCompare: () => void;
}> = ({ status, scenarioLabel, backendLabel, selectedPresetLabel, runtimeComparisonRunning, onRun, onRerun, onReset, onCompare }) => {
    const aura = status.policy_metrics?.['Intelligent AURA'];
    const activePct = status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : aura?.active_sensor_pct || 0;
    const currentPowerSaved = Math.max(0, Math.min(100, 100 - activePct));
    const rows = [
        ['Scenario', scenarioLabel],
        ['Preset', selectedPresetLabel],
        ['Backend', backendLabel],
        ['Phase', status.current_phase],
        ['Power saved', `${fmt(currentPowerSaved)}%`],
        ['Recall', `${fmt(aura?.anomaly_recall_pct ?? status.metrics?.anomaly_recall_pct)}%`],
        ['Active sensors', `${fmt(activePct)}%`],
        ['Shadow sample', `${fmt((status.shadow_validation?.sample_rate || status.shadow_mode_probability || 0) * 100, 1)}%`],
    ];

    return (
        <div className="grid gap-4 p-4">
            <div className="grid gap-2 md:grid-cols-4">
                <button
                    onClick={onRun}
                    className="inline-flex items-center justify-center gap-2 rounded-md bg-[#38BDF8] px-4 py-3 text-sm font-semibold text-[#03111C] transition hover:bg-[#7DD3FC]"
                >
                    {status.is_running ? <Pause size={16} /> : <Play size={16} />}
                    {status.is_running ? 'Pause run' : 'Run playground'}
                </button>
                <button
                    onClick={onRerun}
                    className="rounded-md border border-[#22C55E]/45 bg-[#22C55E]/10 px-4 py-3 text-sm font-semibold text-[#86EFAC] transition hover:bg-[#22C55E]/18"
                >
                    Rerun scenario
                </button>
                <button
                    onClick={onCompare}
                    disabled={runtimeComparisonRunning || status.is_running}
                    className="rounded-md border border-[#A78BFA]/45 bg-[#A78BFA]/10 px-4 py-3 text-sm font-semibold text-[#C4B5FD] transition hover:bg-[#A78BFA]/18 disabled:cursor-not-allowed disabled:opacity-50"
                >
                    {runtimeComparisonRunning ? 'Comparing...' : 'Compare backends'}
                </button>
                <button
                    onClick={onReset}
                    className="rounded-md border border-[#223044] bg-[#0A111C] px-4 py-3 text-sm font-semibold text-[#8EA3B8] transition hover:border-[#64748B] hover:text-[#E6EDF5]"
                >
                    Reset
                </button>
            </div>
            <div className="grid gap-2 md:grid-cols-4">
                {rows.map(([label, value]) => (
                    <div key={label} className="rounded-md bg-[#0A111C] px-3 py-3">
                        <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">{label}</div>
                        <div className="mt-1 truncate font-mono text-sm font-semibold text-[#E6EDF5]">{value}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const trendSpecs = [
    { key: 'power_saved', label: 'Power', color: '#22C55E', suffix: '%' },
    { key: 'active_percent', label: 'Active', color: '#38BDF8', suffix: '%' },
    { key: 'recall', label: 'Recall', color: '#F59E0B', suffix: '%' },
    { key: 'shadow_mse', label: 'Shadow', color: '#A78BFA', suffix: '' },
] as const;

const LiveTrendCard: FC<{
    data: TrendPoint[];
    dataKey: (typeof trendSpecs)[number]['key'];
    label: string;
    color: string;
    suffix: string;
}> = ({ data, dataKey, label, color, suffix }) => {
    const latest = data.length ? data[data.length - 1] : null;
    const value = latest ? Number(latest[dataKey]) : 0;
    const threshold = dataKey === 'shadow_mse' ? latest?.shadow_threshold : undefined;
    return (
        <div className="rounded-lg border border-[#223044] bg-[#0A111C] p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
                <span className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">{label}</span>
                <span className="font-mono text-sm font-semibold" style={{ color }}>
                    {dataKey === 'shadow_mse' ? fmt(value, 5) : `${fmt(value)}${suffix}`}
                </span>
            </div>
            <div className="h-28">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid stroke="#1D2A3A" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="timestep" hide />
                        <YAxis hide domain={dataKey === 'shadow_mse' ? ['auto', 'auto'] : [0, 100]} />
                        <Tooltip
                            contentStyle={{ background: '#050A12', border: '1px solid #223044', borderRadius: 8, color: '#E6EDF5' }}
                            formatter={(tooltipValue: number | string) => dataKey === 'shadow_mse' ? fmt(Number(tooltipValue), 6) : `${fmt(Number(tooltipValue))}%`}
                            labelFormatter={labelValue => `Timestep ${labelValue}`}
                        />
                        <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2.4} dot={false} isAnimationActive={false} />
                        {typeof threshold === 'number' && (
                            <Line type="monotone" dataKey="shadow_threshold" stroke="#EF4444" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                        )}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const LiveTrendsPanel: FC<{ status: Status }> = ({ status }) => {
    const fallbackPoint: TrendPoint = {
        timestep: status.timestep || 0,
        power_saved: status.power_saved_percent || 0,
        active_percent: status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : 0,
        recall: status.policy_metrics?.['Intelligent AURA']?.anomaly_recall_pct ?? status.metrics?.anomaly_recall_pct ?? 0,
        shadow_mse: status.shadow_validation?.recent_shadow_mse || 0,
        shadow_threshold: status.shadow_validation?.mse_threshold || 0,
        retrain_required: Boolean(status.retrain_policy?.required),
    };
    const data = status.trend_history?.length ? status.trend_history : [fallbackPoint];
    return (
        <div className="rounded-xl border border-[#223044] bg-[#0E1520] p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-[#9CC7F5]">Live Trends</h3>
                    <div className="mt-1 text-xs text-[#5F7288]">Rolling replay evidence for power, activity, recall, and shadow validation.</div>
                </div>
                <span className={`rounded-full px-3 py-1 font-mono text-xs ${status.retrain_policy?.required ? 'bg-[#F59E0B]/15 text-[#FDE68A]' : 'bg-[#121C2A] text-[#5F7288]'}`}>
                    {status.retrain_policy?.required ? 'retrain signal' : `${data.length} points`}
                </span>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                {trendSpecs.map(spec => (
                    <LiveTrendCard key={spec.key} data={data} dataKey={spec.key} label={spec.label} color={spec.color} suffix={spec.suffix} />
                ))}
            </div>
        </div>
    );
};

const PlaygroundStudio: FC<{
    status: Status;
    controls: WhatIfControls;
    challenge: ChallengeSettings;
    selectedScenario: ScenarioPresetKey;
    selectedQuestion: PanelistQuestionKey;
    selectedBackendMode: BackendModeKey;
    selectedPreset: DemoPresetKey;
    runtimeRows: RuntimeComparisonRow[];
    runtimeComparisonRunning: boolean;
    advancedOpen: boolean;
    onScenario: (key: ScenarioPresetKey) => void;
    onQuestion: (key: PanelistQuestionKey) => void;
    onControls: (controls: WhatIfControls) => void;
    onChallenge: (settings: ChallengeSettings) => void;
    onBackendMode: (mode: BackendModeKey) => void;
    onAdvancedOpen: (open: boolean) => void;
    onRun: () => void;
    onRerun: () => void;
    onReset: () => void;
    onCompare: () => void;
}> = ({
    status,
    controls,
    challenge,
    selectedScenario,
    selectedQuestion,
    selectedBackendMode,
    selectedPreset,
    runtimeRows,
    runtimeComparisonRunning,
    advancedOpen,
    onScenario,
    onQuestion,
    onControls,
    onChallenge,
    onBackendMode,
    onAdvancedOpen,
    onRun,
    onRerun,
    onReset,
    onCompare,
}) => {
    const aura = status.policy_metrics?.['Intelligent AURA'];
    const activePct = status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : aura?.active_sensor_pct || 0;
    const powerSaved = status.power_saved_percent ?? aura?.power_saved_pct ?? Math.max(0, Math.min(100, 100 - activePct));
    const recall = aura?.anomaly_recall_pct ?? status.metrics?.anomaly_recall_pct ?? 0;
    const retrain = status.retrain_policy;
    const retrainLabel = retrain?.required ? 'Required' : retrain?.recommended ? 'Watch' : 'Stable';
    const retrainTone = retrain?.required ? 'text-[#FDE68A]' : retrain?.recommended ? 'text-[#C4B5FD]' : 'text-[#22C55E]';
    const activeChallenges = challengeDisplay.filter(item => challenge[item.key]).map(item => item.label);
    const recipe = [
        ['Scenario', scenarioPresets[selectedScenario].label],
        ['Question', panelistQuestions[selectedQuestion].label],
        ['Challenge', activeChallenges.length ? activeChallenges.join(' + ') : 'None'],
        ['Backend', backendModes[selectedBackendMode].shortLabel],
    ];

    const challengeToggle = (key: ChallengeToggleKey) => onChallenge({ ...challenge, [key]: !challenge[key] });
    const primaryScenarios = (['balanced', 'highPower', 'highRecall', 'drift'] as ScenarioPresetKey[]);

    return (
        <section className="grid gap-4">
            <div className="rounded-xl border border-[#223044] bg-[#0E1520] p-4 shadow-[0_18px_45px_rgba(0,0,0,0.25)]">
                <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                        <h2 className="text-2xl font-semibold tracking-tight text-white">AURA Playground</h2>
                        <div className="mt-2 flex flex-wrap gap-2">
                            {recipe.map(([label, value]) => (
                                <span key={label} className="rounded-full border border-[#223044] bg-[#0A111C] px-3 py-1.5 text-xs text-[#8EA3B8]">
                                    {label}: <span className="font-semibold text-[#E6EDF5]">{value}</span>
                                </span>
                            ))}
                        </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <button onClick={onRun} className="inline-flex items-center gap-2 rounded-lg bg-[#38BDF8] px-5 py-3 text-sm font-semibold text-[#03111C] transition hover:bg-[#7DD3FC]">
                            {status.is_running ? <Pause size={17} /> : <Play size={17} />}
                            {status.is_running ? 'Pause' : 'Run current test'}
                        </button>
                        <button onClick={onRerun} className="rounded-lg border border-[#22C55E]/45 bg-[#22C55E]/10 px-4 py-3 text-sm font-semibold text-[#86EFAC]">Rerun</button>
                        <button onClick={onCompare} disabled={runtimeComparisonRunning || status.is_running} className="rounded-lg border border-[#A78BFA]/45 bg-[#A78BFA]/10 px-4 py-3 text-sm font-semibold text-[#C4B5FD] disabled:cursor-not-allowed disabled:opacity-50">
                            {runtimeComparisonRunning ? 'Comparing' : 'Compare'}
                        </button>
                        <button onClick={onReset} className="rounded-lg border border-[#223044] bg-[#0A111C] px-4 py-3 text-sm font-semibold text-[#8EA3B8] hover:text-[#E6EDF5]">Reset</button>
                    </div>
                </div>

                <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.72fr)]">
                    <div className="grid gap-4">
                        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                            {primaryScenarios.map(key => (
                                <button
                                    key={key}
                                    onClick={() => onScenario(key)}
                                    className={`rounded-lg border px-4 py-4 text-left transition ${selectedScenario === key ? 'border-[#38BDF8] bg-[#38BDF8]/15' : 'border-[#223044] bg-[#0A111C] hover:border-[#64748B]'}`}
                                >
                                    <div className={`text-sm font-semibold ${selectedScenario === key ? 'text-[#7DD3FC]' : 'text-[#E6EDF5]'}`}>{scenarioPresets[key].label}</div>
                                </button>
                            ))}
                        </div>

                        <div className="rounded-lg border border-[#223044] bg-[#0A111C] p-3">
                            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Panelist question</div>
                            <div className="flex flex-wrap gap-2">
                                {(Object.keys(panelistQuestions) as PanelistQuestionKey[]).map(key => (
                                    <button
                                        key={key}
                                        onClick={() => onQuestion(key)}
                                        className={`rounded-full border px-3 py-2 text-sm font-semibold transition ${selectedQuestion === key ? 'border-[#38BDF8] bg-[#38BDF8]/15 text-[#7DD3FC]' : 'border-[#223044] bg-[#121C2A] text-[#8EA3B8] hover:text-[#E6EDF5]'}`}
                                    >
                                        {panelistQuestions[key].label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="rounded-lg border border-[#223044] bg-[#0A111C] p-3">
                            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Challenge</div>
                            <div className="flex flex-wrap gap-2">
                                {challengeDisplay.map(item => (
                                    <button
                                        key={item.key}
                                        onClick={() => challengeToggle(item.key)}
                                        className={`rounded-full border px-3 py-2 text-sm font-semibold transition ${challenge[item.key] ? 'border-[#F59E0B]/60 bg-[#F59E0B]/15 text-[#FDE68A]' : 'border-[#223044] bg-[#121C2A] text-[#8EA3B8] hover:text-[#E6EDF5]'}`}
                                    >
                                        {item.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <details className="rounded-lg border border-[#223044] bg-[#0A111C] p-3" open={advancedOpen} onToggle={event => onAdvancedOpen(event.currentTarget.open)}>
                            <summary className="cursor-pointer text-sm font-semibold text-[#B8C7D8]">Fine tune</summary>
                            <div className="mt-3 grid gap-4">
                                <div className="grid gap-2 sm:grid-cols-3">
                                    {(Object.keys(backendModes) as BackendModeKey[]).map(key => (
                                        <button key={key} onClick={() => onBackendMode(key)} className={`rounded-md px-3 py-2 text-sm font-semibold ${selectedBackendMode === key ? 'bg-[#38BDF8] text-[#03111C]' : 'bg-[#121C2A] text-[#8EA3B8]'}`}>
                                            {backendModes[key].shortLabel}
                                        </button>
                                    ))}
                                </div>
                                <WhatIfControlsPanel controls={controls} onChange={onControls} />
                                <ChallengePanel settings={challenge} backendMode={selectedBackendMode} onSettings={onChallenge} onBackendMode={onBackendMode} />
                            </div>
                        </details>
                    </div>

                    <div className="grid gap-4">
                        <div className="rounded-lg border border-[#223044] bg-[#0A111C] p-4">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <div className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Current result</div>
                                <span className="rounded-full bg-[#121C2A] px-3 py-1 font-mono text-xs text-[#5F7288]">{status.current_phase}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div className="rounded-md bg-[#050A12] p-3">
                                    <div className="text-[11px] uppercase tracking-[0.12em] text-[#5F7288]">Power saved</div>
                                    <div className="mt-1 font-mono text-xl font-semibold text-[#22C55E]">{fmt(powerSaved)}%</div>
                                </div>
                                <div className="rounded-md bg-[#050A12] p-3">
                                    <div className="text-[11px] uppercase tracking-[0.12em] text-[#5F7288]">Recall</div>
                                    <div className="mt-1 font-mono text-xl font-semibold text-[#F59E0B]">{fmt(recall)}%</div>
                                </div>
                                <div className="rounded-md bg-[#050A12] p-3">
                                    <div className="text-[11px] uppercase tracking-[0.12em] text-[#5F7288]">Active</div>
                                    <div className="mt-1 font-mono text-xl font-semibold text-[#38BDF8]">{fmt(activePct)}%</div>
                                </div>
                                <div className="rounded-md bg-[#050A12] p-3">
                                    <div className="text-[11px] uppercase tracking-[0.12em] text-[#5F7288]">Retrain</div>
                                    <div className={`mt-1 font-mono text-xl font-semibold ${retrainTone}`}>
                                        {retrainLabel}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <LiveTrendsPanel status={status} />

                        <AuraAssistantPanel status={status} />

                        {!!runtimeRows.length && (
                            <div className="rounded-lg border border-[#223044] bg-[#0A111C] p-3">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Backend comparison</div>
                                <div className="grid gap-2">
                                    {runtimeRows.map(row => (
                                        <div key={row.mode} className="flex items-center justify-between rounded-md bg-[#050A12] px-3 py-2 text-sm">
                                            <span className="font-semibold text-[#E6EDF5]">{row.mode}</span>
                                            <span className="font-mono text-[#22C55E]">{fmt(row.speedup, 2)}x</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </section>
    );
};

const AlgorithmState: FC<{ status: Status }> = ({ status }) => {
    const learned = status.learned_parameters || {};
    const runtime = status.runtime;
    const training = status.training;
    const budget = status.active_budget_band || [20, 30];
    const shadow = status.shadow_validation;
    const retrain = status.retrain_policy;
    const items = [
        ['Backend', status.backend_mode || 'pending'],
        ['Run budget', `${fmt(budget[0], 0)}-${fmt(budget[1], 0)}% active`],
        ['Training time', `${fmt(training?.seconds, 3)}s`],
        ['Total runtime', `${fmt(runtime?.elapsed_seconds, 3)}s`],
        ['Final loss', fmt(training?.final_loss, 6)],
        ['Sensors / steps', `${runtime?.bench_sensors || status.total_sensors} / ${runtime?.bench_steps || 0}`],
        ['Shadow sample', `${fmt((shadow?.sample_rate || 0) * 100, 1)}%`],
        ['Shadow MSE', fmt(shadow?.recent_shadow_mse, 6)],
        ['Retrain due in', `${retrain?.steps_until_forced_retrain ?? status.hybrid_max_timesteps_since_retrain} steps`],
        ['Retrain status', retrain?.required ? 'required' : retrain?.recommended ? 'recommended' : 'healthy'],
        ['Threshold mean', fmt(learned.redundancy_threshold_mean, 6)],
        ['Gate mean', fmt(learned.gate_threshold_mean, 6)],
    ];

    return (
        <div className="grid gap-2 p-4">
            {items.map(([label, value]) => (
                <div key={label} className="flex items-center justify-between gap-4 rounded-md bg-[#0A111C] px-3 py-2">
                    <span className="text-xs uppercase tracking-[0.12em] text-[#5F7288]">{label}</span>
                    <span className="font-mono text-sm tabular-nums text-[#E6EDF5]">{value}</span>
                </div>
            ))}
        </div>
    );
};

const GuidedDemoPanel: FC<{ status: Status; selectedSensorId: number | null; policyCount: number }> = ({ status, selectedSensorId, policyCount }) => {
    let activeStep = 0;
    if (status.learner_status === 'running' || status.current_phase === 'collecting') {
        activeStep = 0;
    } else if (status.current_phase === 'shadow_op') {
        activeStep = 1;
    } else if (selectedSensorId !== null) {
        activeStep = 2;
    } else if (policyCount > 0) {
        activeStep = 3;
    } else if (status.current_phase === 'finished') {
        activeStep = 4;
    }

    return (
        <section className="rounded-lg border border-[#223044] bg-[#0E1520] px-4 py-3">
            <div className="mb-3 flex flex-col gap-1 md:flex-row md:items-end md:justify-between">
                <div>
                    <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[#8EA3B8]">Presentation Mode</h2>
                </div>
                <div className="font-mono text-xs uppercase tracking-[0.14em] text-[#38BDF8]">
                    Step {activeStep + 1} / {demoSteps.length}
                </div>
            </div>
            <div className="grid gap-2 lg:grid-cols-5">
                {demoSteps.map((step, index) => {
                    const active = index === activeStep;
                    const complete = index < activeStep;
                    return (
                        <div
                            key={step.title}
                            className={`rounded-md border px-3 py-3 transition ${
                                active
                                    ? 'border-[#38BDF8] bg-[#38BDF8]/12'
                                    : complete
                                        ? 'border-[#22C55E]/35 bg-[#22C55E]/8'
                                        : 'border-[#223044] bg-[#0A111C]'
                            }`}
                        >
                            <div className="mb-2 flex items-center gap-2">
                                <span className={`grid h-6 w-6 place-items-center rounded-full text-xs font-semibold ${
                                    active
                                        ? 'bg-[#38BDF8] text-[#03111C]'
                                        : complete
                                            ? 'bg-[#22C55E] text-[#03111C]'
                                            : 'bg-[#121C2A] text-[#8EA3B8]'
                                }`}>
                                    {index + 1}
                                </span>
                                <h3 className="text-sm font-semibold text-[#E6EDF5]">{step.title}</h3>
                            </div>
                            <p className="text-xs leading-5 text-[#8EA3B8]">{step.detail}</p>
                        </div>
                    );
                })}
            </div>
        </section>
    );
};

const SensorInspector: FC<{ status: Status; selectedSensorId: number | null }> = ({ status, selectedSensorId }) => {
    const visibleCount = status.sensors.length;
    const selected = selectedSensorId == null
        ? null
        : status.sensors.find(sensor => sensor.id === selectedSensorId) || null;
    const fallback = status.sensors[0] || null;
    const sensor = selected || fallback;

    if (!sensor) {
        return (
            <div className="p-4 text-sm text-[#8EA3B8]">No sensor snapshot.</div>
        );
    }

    const reading = status.current_readings?.[sensor.id];
    const detail = status.sensor_details?.find(item => item.id === sensor.id);
    const activePct = status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : 0;
    const state = detail ? (detail.is_sleeping ? 'SLEEPING' : 'ACTIVE') : sensor.is_off ? 'SLEEPING' : 'ACTIVE';
    const stateColor = state === 'SLEEPING' ? '#64748B' : '#38BDF8';
    const shownReading = detail?.reading ?? reading;
    const estimate = detail?.estimated_reading;
    const absError = detail?.abs_error;
    const anomaly = detail?.is_anomaly ?? false;
    const shadow = detail?.is_shadow ?? sensor.is_shadow ?? false;
    const redundancyGroup = Math.floor(sensor.id / 4) + 1;
    const reason = detail?.decision_reason || (sensor.is_off
        ? 'AURA marked this node as redundant under the learned active-budget policy.'
        : activePct <= (status.active_budget_band?.[1] || 30)
            ? 'Kept active to preserve coverage inside the learned budget band.'
            : 'Active during training/replay initialization.');

    return (
        <div className="grid gap-3 p-4">
            <div className="flex items-center justify-between rounded-md bg-[#0A111C] px-3 py-3">
                <span className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Selected sensor</span>
                <span className="font-mono text-lg font-semibold text-[#E6EDF5]">#{sensor.id + 1}</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">State</div>
                    <div className="mt-1 font-mono text-sm font-semibold" style={{ color: stateColor }}>{state}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Reading</div>
                    <div className="mt-1 font-mono text-sm font-semibold text-[#E6EDF5]">{Number.isFinite(shownReading) ? fmt(shownReading, 4) : 'pending'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Estimate</div>
                    <div className="mt-1 font-mono text-sm font-semibold text-[#E6EDF5]">{Number.isFinite(estimate) ? fmt(estimate, 4) : 'pending'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Abs error</div>
                    <div className="mt-1 font-mono text-sm font-semibold text-[#E6EDF5]">{Number.isFinite(absError) ? fmt(absError, 5) : 'pending'}</div>
                </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Anomaly flag</div>
                    <div className="mt-1 font-mono text-sm font-semibold" style={{ color: anomaly ? '#F59E0B' : '#8EA3B8' }}>{anomaly ? 'YES' : 'NO'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Network load</div>
                    <div className="mt-1 font-mono text-sm font-semibold text-[#38BDF8]">{fmt(activePct)}%</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Redundancy group</div>
                    <div className="mt-1 font-mono text-sm font-semibold text-[#A78BFA]">G{redundancyGroup}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Shadow validation</div>
                    <div className="mt-1 font-mono text-sm font-semibold" style={{ color: shadow ? '#60A5FA' : '#8EA3B8' }}>{shadow ? 'HELD ACTIVE' : 'NO'}</div>
                </div>
            </div>
            <div className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-3">
                <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Decision explanation</div>
                <p className="text-sm leading-6 text-[#B8C7D8]">{reason}</p>
            </div>
            <div className="text-xs text-[#5F7288]">Visible prototype nodes: {visibleCount}</div>
        </div>
    );
};

const HardwarePanel: FC<{ status: Status }> = ({ status }) => {
    const hardware = status.hardware;
    const connected = hardware?.arduino_status === 'connected';
    const modeLabel = connected ? `Connected to ${hardware?.com_port || 'serial device'}` : 'Simulated command preview';
    const items = [
        ['Bridge', hardware?.bridge_status || 'ready'],
        ['Device', hardware?.arduino_status || 'not_connected'],
        ['COM port', hardware?.com_port || 'COM16'],
        ['Baud', hardware?.baud_rate || 115200],
        ['Active nodes', hardware?.active_nodes ?? status.active_sensors],
        ['Sleeping nodes', hardware?.sleeping_nodes ?? Math.max(0, status.sensors.length - status.active_sensors)],
        ['Last sync', hardware?.last_sync || 'pending'],
    ];

    return (
        <div className="grid gap-3 p-4">
            <div className={`rounded-md border px-3 py-3 ${connected ? 'border-[#22C55E]/45 bg-[#22C55E]/10' : 'border-[#F59E0B]/35 bg-[#F59E0B]/10'}`}>
                <div className="text-[11px] uppercase tracking-[0.14em] text-[#8EA3B8]">Hardware status</div>
                <div className={`mt-1 font-mono text-sm font-semibold ${connected ? 'text-[#86EFAC]' : 'text-[#FDE68A]'}`}>{modeLabel}</div>
                <div className="mt-2 text-xs leading-5 text-[#8EA3B8]">Command bits: <span className="font-mono text-[#E6EDF5]">0 = active/on</span>, <span className="font-mono text-[#E6EDF5]">1 = sleeping/off</span>.</div>
            </div>
            <div className="grid grid-cols-2 gap-2">
                {items.map(([label, value]) => (
                    <div key={label} className="rounded-md bg-[#0A111C] px-3 py-3">
                        <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">{label}</div>
                        <div className="mt-1 truncate font-mono text-sm font-semibold text-[#E6EDF5]">{value}</div>
                    </div>
                ))}
            </div>
            <div className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-3">
                <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Last command preview</div>
                <div className="break-all font-mono text-xs leading-5 text-[#22C55E]">{hardware?.last_command || 'pending'}</div>
            </div>
            <div className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-3">
                <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Defence check</div>
                <div className="text-xs leading-5 text-[#8EA3B8]">If no board is connected, the UI still shows the exact serial command AURA would send. Close Arduino Serial Monitor before connecting to real hardware.</div>
            </div>
            {hardware?.last_error && <div className="rounded-md border border-[#EF4444]/45 bg-[#EF4444]/10 px-3 py-2 text-xs text-[#FCA5A5]">{hardware.last_error}</div>}
        </div>
    );
};

const Esp32ExperimentPanel: FC<{
    ldr1Pin: string;
    ldr2Pin: string;
    led1Pin: string;
    led2Pin: string;
    threshold: string;
    onLdr1Pin: (value: string) => void;
    onLdr2Pin: (value: string) => void;
    onLed1Pin: (value: string) => void;
    onLed2Pin: (value: string) => void;
    onThreshold: (value: string) => void;
}> = ({ ldr1Pin, ldr2Pin, led1Pin, led2Pin, threshold, onLdr1Pin, onLdr2Pin, onLed1Pin, onLed2Pin, onThreshold }) => {
    const experimentRows = [
        ['Different light', 'LED1 ON, LED2 ON'],
        ['Similar light', 'LED1 ON, LED2 OFF'],
    ] as const;
    const inputs = [
        ['LDR1 pin', ldr1Pin, onLdr1Pin],
        ['LDR2 pin', ldr2Pin, onLdr2Pin],
        ['LED1 pin', led1Pin, onLed1Pin],
        ['LED2 pin', led2Pin, onLed2Pin],
        ['Threshold', threshold, onThreshold],
    ] as const;

    return (
        <div className="grid gap-3 p-4">
            <div className="grid gap-2 md:grid-cols-5">
                {inputs.map(([label, value, onChange]) => (
                    <label key={label} className="rounded-md bg-[#0A111C] px-3 py-3">
                        <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">{label}</div>
                        <input
                            value={value}
                            onChange={event => onChange(event.target.value)}
                            className="w-full rounded-md border border-[#223044] bg-[#050A12] px-2 py-2 font-mono text-sm text-[#E6EDF5] outline-none focus:border-[#38BDF8]"
                        />
                    </label>
                ))}
            </div>
            <div className="grid gap-2 md:grid-cols-2">
                {experimentRows.map(([label, value]) => (
                    <div key={label} className="flex items-center justify-between gap-4 rounded-md bg-[#0A111C] px-3 py-3">
                        <span className="text-xs uppercase tracking-[0.12em] text-[#5F7288]">{label}</span>
                        <span className="text-right font-mono text-sm text-[#E6EDF5]">{value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

const MegaSimulationPanel: FC<{
    startPin: string;
    nodeCount: string;
    activeBit: string;
    sleepBit: string;
    onStartPin: (value: string) => void;
    onNodeCount: (value: string) => void;
    onActiveBit: (value: string) => void;
    onSleepBit: (value: string) => void;
}> = ({ startPin, nodeCount, activeBit, sleepBit, onStartPin, onNodeCount, onActiveBit, onSleepBit }) => {
    const inputs = [
        ['Start pin', startPin, onStartPin],
        ['Node count', nodeCount, onNodeCount],
        ['Active bit', activeBit, onActiveBit],
        ['Sleep bit', sleepBit, onSleepBit],
    ] as const;
    const firstPin = Number(startPin) || 22;
    const count = Math.max(1, Math.min(Number(nodeCount) || 28, 28));
    const lastPin = firstPin + count - 1;

    return (
        <div className="grid gap-3 p-4">
            <div className="grid gap-2 md:grid-cols-4">
                {inputs.map(([label, value, onChange]) => (
                    <label key={label} className="rounded-md bg-[#0A111C] px-3 py-3">
                        <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">{label}</div>
                        <input
                            value={value}
                            onChange={event => onChange(event.target.value)}
                            className="w-full rounded-md border border-[#223044] bg-[#050A12] px-2 py-2 font-mono text-sm text-[#E6EDF5] outline-none focus:border-[#38BDF8]"
                        />
                    </label>
                ))}
            </div>
            <div className="grid gap-2 md:grid-cols-3">
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Pin range</div>
                    <div className="mt-1 font-mono text-sm text-[#E6EDF5]">{firstPin}-{lastPin}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">LED ON</div>
                    <div className="mt-1 font-mono text-sm text-[#7DD3FC]">bit {activeBit}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">LED OFF</div>
                    <div className="mt-1 font-mono text-sm text-[#CBD5E1]">bit {sleepBit}</div>
                </div>
            </div>
        </div>
    );
};

const KernelProofPanel: FC<{ status: Status }> = ({ status }) => {
    const proof = status.kernel_proof;

    if (!proof) {
        return (
            <EmptyState
                title="Kernel pending"
                detail="Start backend."
            />
        );
    }

    const statusRows = [
        ['Extension loaded', proof.status.extension_loaded],
        ['CUDA enabled', proof.status.cuda_preferred],
        ['Pair cache', proof.status.pair_cache_enabled],
        ['Fused loss', proof.status.fused_cached_training_loss],
        ['Manual backward', proof.status.manual_backward_enabled],
        ['CPU fallback', proof.status.cpu_fallback_available],
    ] as const;

    return (
        <div className="grid gap-4 p-4">
            <div className="grid gap-3 md:grid-cols-4">
                <div className="rounded-md border border-[#223044] bg-[#0A111C] p-4">
                    <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Loaded Backend</div>
                    <div className="font-mono text-lg font-semibold text-[#38BDF8]">{proof.backend_mode}</div>
                    <div className="mt-2 text-xs text-[#5F7288]">Live training: {fmt(proof.status.live_training_seconds, 3)}s</div>
                </div>
                <div className="rounded-md border border-[#22C55E]/35 bg-[#22C55E]/8 p-4">
                    <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#86EFAC]">
                        <CheckCircle2 size={15} /> Forward Parity
                    </div>
                    <div className="font-mono text-lg font-semibold text-[#E6EDF5]">{proof.correctness.forward_loss_parity}</div>
                    <div className="mt-2 text-xs text-[#5F7288]">Loss tolerance {proof.correctness.loss_tolerance}</div>
                </div>
                <div className="rounded-md border border-[#22C55E]/35 bg-[#22C55E]/8 p-4">
                    <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#86EFAC]">
                        <CheckCircle2 size={15} /> Gradient Parity
                    </div>
                    <div className="font-mono text-lg font-semibold text-[#E6EDF5]">{proof.correctness.gradient_parity}</div>
                    <div className="mt-2 text-xs text-[#5F7288]">Max observed error {proof.correctness.max_observed_gradient_error}</div>
                </div>
            </div>

            <div className="grid gap-2 md:grid-cols-3">
                {statusRows.map(([label, value]) => (
                    <div key={label} className="flex items-center justify-between rounded-md bg-[#0A111C] px-3 py-3">
                        <span className="text-xs uppercase tracking-[0.12em] text-[#5F7288]">{label}</span>
                        <span className={`font-mono text-sm font-semibold ${value ? 'text-[#22C55E]' : 'text-[#EF4444]'}`}>
                            {value ? 'ON' : 'OFF'}
                        </span>
                    </div>
                ))}
            </div>

            <div className="overflow-x-auto rounded-md border border-[#223044]">
                <table className="w-full min-w-[720px] border-collapse text-left text-sm">
                    <thead>
                        <tr className="border-b border-[#223044] text-xs uppercase tracking-[0.12em] text-[#8EA3B8]">
                            <th className="px-4 py-3 font-semibold">Backend</th>
                            <th className="px-4 py-3 font-semibold">Training Time</th>
                            <th className="px-4 py-3 font-semibold">Speedup</th>
                            <th className="px-4 py-3 font-semibold">Purpose</th>
                        </tr>
                    </thead>
                    <tbody>
                        {proof.speed.map(row => (
                            <tr key={row.backend} className="border-b border-[#182335] last:border-b-0">
                                <td className="px-4 py-3 font-semibold text-[#E6EDF5]">{row.backend}</td>
                                <td className="px-4 py-3 font-mono tabular-nums text-[#E6EDF5]">{fmt(row.training_seconds, 3)}s</td>
                                <td className="px-4 py-3 font-mono tabular-nums text-[#22C55E]">{fmt(row.speedup_vs_pytorch, 2)}x</td>
                                <td className="px-4 py-3 text-[#8EA3B8]">{row.purpose}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="grid gap-2 lg:grid-cols-6">
                {proof.architecture_steps.map((step, index) => (
                    <div key={step} className="rounded-md border border-[#223044] bg-[#0A111C] p-3">
                        <div className="mb-2 font-mono text-xs font-semibold text-[#38BDF8]">0{index + 1}</div>
                        <div className="text-xs leading-5 text-[#B8C7D8]">{step}</div>
                    </div>
                ))}
            </div>

            <div className="rounded-md border border-[#223044] bg-[#0A111C] px-4 py-3 text-sm leading-6 text-[#B8C7D8]">
                {proof.note}
            </div>
        </div>
    );
};

const DatasetPanel: FC<{
    datasets: DatasetSummary[];
    selectedDatasetId: number | null;
    selectedColumns: string[];
    onSelectDataset: (dataset: DatasetSummary) => void;
    onToggleColumn: (column: string) => void;
    onUpload: (file: File) => Promise<void>;
}> = ({ datasets, selectedDatasetId, selectedColumns, onSelectDataset, onToggleColumn, onUpload }) => {
    const selected = datasets.find(dataset => dataset.id === selectedDatasetId) || null;
    const columns = selected?.numeric_columns || [];

    return (
        <div className="grid gap-4 p-4">
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-[#38BDF8]/45 bg-[#38BDF8]/8 px-4 py-6 text-center transition hover:bg-[#38BDF8]/12">
                <Upload className="mb-2 text-[#38BDF8]" size={22} />
                <span className="text-sm font-semibold text-[#E6EDF5]">Upload CSV sensor dataset</span>
                <span className="mt-1 text-xs text-[#8EA3B8]">CSV only.</span>
                <input
                    type="file"
                    accept=".csv,text/csv"
                    className="hidden"
                    onChange={event => {
                        const file = event.target.files?.[0];
                        if (file) void onUpload(file);
                        event.currentTarget.value = '';
                    }}
                />
            </label>

            <div className="grid gap-2">
                <div className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Uploaded datasets</div>
                {datasets.length ? datasets.map(dataset => (
                    <button
                        key={dataset.id}
                        onClick={() => onSelectDataset(dataset)}
                        className={`rounded-md border px-3 py-3 text-left transition ${
                            selectedDatasetId === dataset.id
                                ? 'border-[#38BDF8] bg-[#38BDF8]/12'
                                : 'border-[#223044] bg-[#0A111C] hover:border-[#64748B]'
                        }`}
                    >
                        <div className="flex items-center justify-between gap-3">
                            <span className="truncate text-sm font-semibold text-[#E6EDF5]">{dataset.filename}</span>
                            <span className="font-mono text-xs text-[#38BDF8]">#{dataset.id}</span>
                        </div>
                        <div className="mt-1 text-xs text-[#8EA3B8]">
                            {dataset.row_count} rows | {dataset.numeric_columns?.length || dataset.numeric_column_count || 0} numeric columns | uploaded {dataset.uploaded_at_iso || 'recently'}
                        </div>
                    </button>
                )) : (
                    <EmptyState title="No datasets" detail="Upload CSV." />
                )}
            </div>

            {selected && (
                <div className="rounded-md border border-[#223044] bg-[#0A111C] p-3">
                    <div className="mb-3 flex items-center justify-between gap-3">
                        <div>
                            <div className="text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Sensor column selection</div>
                            <div className="mt-1 text-xs text-[#5F7288]">{selectedColumns.length} selected for the next run</div>
                        </div>
                    </div>
                    <div className="grid max-h-[360px] gap-2 overflow-y-auto pr-1 sm:grid-cols-2 xl:grid-cols-3">
                        {columns.map(column => {
                            const checked = selectedColumns.includes(column);
                            return (
                                <label key={column} className={`flex items-center gap-2 rounded-md border px-2 py-2 text-xs ${checked ? 'border-[#22C55E]/45 bg-[#22C55E]/8 text-[#D1FAE5]' : 'border-[#223044] bg-[#121C2A] text-[#8EA3B8]'}`}>
                                    <input type="checkbox" checked={checked} onChange={() => onToggleColumn(column)} />
                                    <span className="truncate">{column}</span>
                                </label>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

const HistoryPanel: FC<{ runs?: BenchmarkRun[] }> = ({ runs = [] }) => (
    <div className="overflow-x-auto">
        {runs.length ? (
            <table className="w-full min-w-[720px] border-collapse text-left text-sm">
                <thead>
                    <tr className="border-b border-[#223044] text-xs uppercase tracking-[0.12em] text-[#8EA3B8]">
                        <th className="px-4 py-3 font-semibold">Run</th>
                        <th className="px-4 py-3 font-semibold">Started</th>
                        <th className="px-4 py-3 font-semibold">Phase</th>
                        <th className="px-4 py-3 font-semibold">Backend</th>
                        <th className="px-4 py-3 font-semibold">Dataset</th>
                        <th className="px-4 py-3 font-semibold">Error</th>
                    </tr>
                </thead>
                <tbody>
                    {runs.map(run => (
                        <tr key={run.id} className="border-b border-[#182335] last:border-b-0">
                            <td className="px-4 py-3 font-mono text-[#38BDF8]">#{run.id}</td>
                            <td className="px-4 py-3 text-[#B8C7D8]">{run.started_at_iso || 'pending'}</td>
                            <td className="px-4 py-3 font-mono text-[#E6EDF5]">{run.phase}</td>
                            <td className="px-4 py-3 text-[#B8C7D8]">{run.backend_mode || 'pending'}</td>
                            <td className="px-4 py-3 font-mono text-[#8EA3B8]">{run.dataset_id || 'synthetic'}</td>
                            <td className="px-4 py-3 text-[#FCA5A5]">{run.error || '-'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        ) : (
            <EmptyState title="No history" detail="Run AURA." />
        )}
    </div>
);

const DiagnosticsPanel: FC<{ diagnostics?: DiagnosticEntry[]; status: Status; transport: string; onRetrain: () => void }> = ({ diagnostics = [], status, transport, onRetrain }) => {
    const fallbackReason = status.kernel_proof?.status.fallback_reason;
    const hardwareError = status.hardware?.last_error;
    const shadow = status.shadow_validation;
    const retrain = status.retrain_policy;
    const rows = [
        ...(retrain?.reason ? [{ time: '', severity: retrain.required ? 'warning' : 'info', source: 'retrain', message: retrain.reason }] : []),
        ...(fallbackReason ? [{ time: '', severity: 'warning', source: 'kernel', message: fallbackReason }] : []),
        ...(hardwareError ? [{ time: '', severity: 'warning', source: 'hardware', message: hardwareError }] : []),
        ...diagnostics,
    ];

    return (
        <div className="grid gap-4 p-4">
            <div className="grid gap-3 md:grid-cols-3">
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Status stream</div>
                    <div className="mt-1 font-mono text-lg font-semibold text-[#38BDF8]">{transport}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Backend mode</div>
                    <div className="mt-1 font-mono text-lg font-semibold text-[#E6EDF5]">{status.backend_mode || 'pending'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Dataset</div>
                    <div className="mt-1 truncate font-mono text-lg font-semibold text-[#E6EDF5]">{status.dataset?.filename || 'synthetic/default'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Storage</div>
                    <div className="mt-1 font-mono text-lg font-semibold text-[#22C55E]">{status.storage?.backend || 'sqlite'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Shadow samples</div>
                    <div className="mt-1 font-mono text-lg font-semibold text-[#A78BFA]">{shadow?.sample_count ?? 0}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Shadow MSE</div>
                    <div className="mt-1 font-mono text-lg font-semibold text-[#E6EDF5]">{fmt(shadow?.recent_shadow_mse, 6)}</div>
                </div>
                <div className={`rounded-md p-4 ${retrain?.required ? 'border border-[#F59E0B]/45 bg-[#F59E0B]/10' : 'bg-[#0A111C]'}`}>
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Retrain policy</div>
                    <div className={`mt-1 font-mono text-lg font-semibold ${retrain?.required ? 'text-[#FDE68A]' : 'text-[#22C55E]'}`}>{retrain?.required ? 'required' : retrain?.recommended ? 'recommended' : 'healthy'}</div>
                </div>
                <div className="rounded-md bg-[#0A111C] p-4">
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Forced retrain in</div>
                    <div className="mt-1 font-mono text-lg font-semibold text-[#38BDF8]">{retrain?.steps_until_forced_retrain ?? status.hybrid_max_timesteps_since_retrain}</div>
                </div>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-[#223044] bg-[#0A111C] px-3 py-3">
                <div>
                    <div className="text-xs uppercase tracking-[0.14em] text-[#5F7288]">Shadow validation</div>
                    <div className="mt-1 text-sm leading-6 text-[#B8C7D8]">
                        {fmt((shadow?.sample_rate || status.shadow_mode_probability) * 100, 1)}% of policy-sleeping sensors are silently kept active and checked against their predicted value.
                    </div>
                </div>
                <button
                    onClick={onRetrain}
                    className="rounded-md border border-[#A78BFA]/40 bg-[#A78BFA]/10 px-3 py-2 text-sm font-semibold text-[#DDD6FE] transition hover:bg-[#A78BFA]/20"
                >
                    Retrain now
                </button>
            </div>
            {rows.length ? rows.slice().reverse().map((item, index) => (
                <div key={`${item.time}-${item.source}-${index}`} className={`rounded-md border px-3 py-3 ${item.severity === 'error' ? 'border-[#EF4444]/45 bg-[#EF4444]/10' : item.severity === 'warning' ? 'border-[#F59E0B]/45 bg-[#F59E0B]/10' : 'border-[#223044] bg-[#0A111C]'}`}>
                    <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-[#8EA3B8]">
                        <AlertTriangle size={14} /> {item.source} {item.time && <span className="font-mono text-[#5F7288]">{item.time}</span>}
                    </div>
                    <div className="text-sm leading-6 text-[#E6EDF5]">{item.message}</div>
                </div>
            )) : (
                <EmptyState title="No diagnostics" detail="All clear." />
            )}
        </div>
    );
};

const ClientOnlyApp: FC = () => {
    const { status, sendCommand, setChartData, transport, fetchStatus } = useStatus();
    const [selectedSensorId, setSelectedSensorId] = useState<number | null>(null);
    const [selectedPreset, setSelectedPreset] = useState<DemoPresetKey>('fastCuda');
    const [selectedScenario, setSelectedScenario] = useState<ScenarioPresetKey>('balanced');
    const [selectedQuestion, setSelectedQuestion] = useState<PanelistQuestionKey>('balanced');
    const [playgroundAdvancedOpen, setPlaygroundAdvancedOpen] = useState(false);
    const [whatIfControls, setWhatIfControls] = useState<WhatIfControls>(defaultWhatIfControls);
    const [challengeSettings, setChallengeSettings] = useState<ChallengeSettings>(defaultChallengeSettings);
    const [selectedBackendMode, setSelectedBackendMode] = useState<BackendModeKey>('autoCuda');
    const [presentationMode, setPresentationMode] = useState(true);
    const [compactMode, setCompactMode] = useState(false);
    const [farmView, setFarmView] = useState<'aura' | 'allActive'>('aura');
    const [replaySpeed, setReplaySpeed] = useState(1);
    const [runtimeComparisonRows, setRuntimeComparisonRows] = useState<RuntimeComparisonRow[]>([]);
    const [runtimeComparisonRunning, setRuntimeComparisonRunning] = useState(false);
    const [activeTab, setActiveTab] = useState<AppTab>(getInitialTab);
    const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
    const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
    const [hardwareTarget, setHardwareTarget] = useState<HardwareTarget>('esp32_experiment');
    const [hardwarePort, setHardwarePort] = useState('COM16');
    const [hardwarePorts, setHardwarePorts] = useState<SerialPortOption[]>([]);
    const [hardwarePortsError, setHardwarePortsError] = useState('');
    const [hardwarePortsLoading, setHardwarePortsLoading] = useState(false);
    const [hardwareBaudRate, setHardwareBaudRate] = useState('115200');
    const [ldr1Pin, setLdr1Pin] = useState('34');
    const [ldr2Pin, setLdr2Pin] = useState('35');
    const [led1Pin, setLed1Pin] = useState('25');
    const [led2Pin, setLed2Pin] = useState('26');
    const [redundancyThreshold, setRedundancyThreshold] = useState('280');
    const [megaStartPin, setMegaStartPin] = useState('22');
    const [megaNodeCount, setMegaNodeCount] = useState('28');
    const [megaActiveBit, setMegaActiveBit] = useState('0');
    const [megaSleepBit, setMegaSleepBit] = useState('1');

    const refreshHardwarePorts = async () => {
        setHardwarePortsLoading(true);
        setHardwarePortsError('');
        try {
            const response = await fetch(`${API_BASE_URL}/hardware/ports`);
            if (!response.ok) throw new Error('port scan failed');
            const body: { ports?: SerialPortOption[]; error?: string } = await response.json();
            const ports = body.ports || [];
            setHardwarePorts(ports);
            setHardwarePortsError(body.error || '');
            if (ports.length && !ports.some(port => port.device === hardwarePort)) {
                setHardwarePort(ports[0].device);
            }
        } catch (error) {
            setHardwarePorts([]);
            setHardwarePortsError(error instanceof Error ? error.message : 'port scan failed');
        } finally {
            setHardwarePortsLoading(false);
        }
    };

    useEffect(() => {
        let cancelled = false;
        const loadDatasets = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/datasets`);
                if (!response.ok) return;
                const body: { datasets?: DatasetSummary[] } = await response.json();
                if (!cancelled) setDatasets(body.datasets || []);
            } catch (error) {
                console.error('Failed to load datasets:', error);
            }
        };
        void loadDatasets();
        void refreshHardwarePorts();
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (!selectedDatasetId && datasets.length) {
            const dataset = datasets[0];
            setSelectedDatasetId(dataset.id);
            setSelectedColumns(dataset.selected_columns?.length ? dataset.selected_columns : (dataset.numeric_columns || []).slice(0, 64));
        }
    }, [datasets, selectedDatasetId]);

    useEffect(() => {
        if (presentationMode && appTabs.find(tab => tab.key === activeTab)?.technicalOnly) {
            setActiveTab('live');
        }
    }, [activeTab, presentationMode]);

    if (!status) {
        return <main className="grid min-h-screen place-items-center bg-[#070B10] text-[#E6EDF5]">Connecting to AURA gateway</main>;
    }

    const policies = status.policy_metrics || {};
    const policyRows = Object.entries(policies).map(([name, metrics]) => ({
        name,
        power: metrics.power_saved_pct,
        bandwidth: metrics.power_saved_pct,
        recall: metrics.anomaly_recall_pct,
        mse: metrics.global_reconstruction_mse,
    }));
    const trainingLoss = (status.training?.losses || []).map((loss, epoch) => ({ epoch: epoch + 1, loss }));
    const activePct = status.total_sensors ? (status.active_sensors / status.total_sensors) * 100 : 0;
    const selectedSensorExists = selectedSensorId == null || status.sensors.some(sensor => sensor.id === selectedSensorId);
    const effectiveSelectedSensorId = selectedSensorExists ? selectedSensorId : null;
    const displayedSensors = farmView === 'allActive'
        ? status.sensors.map(sensor => ({ ...sensor, is_off: false }))
        : status.sensors;
    const farmSensors = displayedSensors.slice(0, 28);
    const farmSensorDetails = (status.sensor_details || []).slice(0, 28);
    const effectiveReplaySpeed = status.replay_speed || replaySpeed;

    const refreshDatasets = async () => {
        const response = await fetch(`${API_BASE_URL}/datasets`);
        if (!response.ok) return;
        const body: { datasets?: DatasetSummary[] } = await response.json();
        setDatasets(body.datasets || []);
    };

    const persistDatasetSelection = async (datasetId: number, columns: string[]) => {
        await fetch(`${API_BASE_URL}/datasets/${datasetId}/selection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selected_columns: columns }),
        });
    };

    const handleUploadDataset = async (file: File) => {
        const response = await fetch(`${API_BASE_URL}/datasets/upload`, {
            method: 'POST',
            headers: {
                'Content-Type': file.type || 'text/csv',
                'x-filename': file.name,
            },
            body: await file.arrayBuffer(),
        });
        if (!response.ok) return;
        const dataset: DatasetSummary = await response.json();
        await refreshDatasets();
        setSelectedDatasetId(dataset.id);
        setSelectedColumns(dataset.selected_columns?.length ? dataset.selected_columns : (dataset.numeric_columns || []).slice(0, 64));
        await fetchStatus();
    };

    const handleSelectDataset = (dataset: DatasetSummary) => {
        setSelectedDatasetId(dataset.id);
        setSelectedColumns(dataset.selected_columns?.length ? dataset.selected_columns : (dataset.numeric_columns || []).slice(0, 64));
    };

    const handleToggleColumn = (column: string) => {
        if (!selectedDatasetId) return;
        const next = selectedColumns.includes(column)
            ? selectedColumns.filter(item => item !== column)
            : [...selectedColumns, column];
        setSelectedColumns(next);
        void persistDatasetSelection(selectedDatasetId, next).then(refreshDatasets).catch(error => {
            console.error('Failed to persist dataset column selection:', error);
        });
    };

    const handleScenarioSelect = (scenario: ScenarioPresetKey) => {
        const preset = scenarioPresets[scenario];
        setSelectedScenario(scenario);
        setWhatIfControls(preset.controls);
        setChallengeSettings(preset.challenge);
        setSelectedPreset(preset.preset);
    };

    const handleQuestionSelect = (question: PanelistQuestionKey) => {
        const next = panelistQuestions[question];
        setSelectedQuestion(question);
        setWhatIfControls({ ...whatIfControls, ...next.controls });
        setChallengeSettings({ ...challengeSettings, ...next.challenge });
        if (next.backend) setSelectedBackendMode(next.backend);
    };

    const buildPlaygroundPayload = () => {
        const baseSensors = demoPresets[selectedPreset].payload.BENCH_SENSORS;
        const payload = {
            ...demoPresets[selectedPreset].payload,
            ...backendModes[selectedBackendMode].payload,
            ...controlsToPayload(whatIfControls, challengeSettings, baseSensors),
            REPLAY_SPEED: replaySpeed,
        };
        if (!presentationMode && selectedDatasetId && selectedColumns.length >= 2) {
            Object.assign(payload, {
                DATASET_ID: selectedDatasetId,
                DATASET_COLUMNS: selectedColumns,
            });
        }
        return payload;
    };

    const handleStartPause = () => {
        if (status.is_running) {
            sendCommand('pause');
            return;
        }
        setChartData([]);
        sendCommand('start', buildPlaygroundPayload());
    };

    const handlePlaygroundRerun = async () => {
        setChartData([]);
        await sendCommand('reset');
        await sleep(350);
        await sendCommand('start', buildPlaygroundPayload());
    };

    const handlePresentationModeStart = async () => {
        setPresentationMode(true);
        setActiveTab('playground');
        setSelectedPreset('fastCuda');
        setSelectedScenario('balanced');
        setSelectedQuestion('balanced');
        setSelectedBackendMode('autoCuda');
        setWhatIfControls({ ...defaultWhatIfControls });
        setChallengeSettings({ ...defaultChallengeSettings });
        setChartData([]);
        await sendCommand('reset');
        await sleep(350);
        await sendCommand('diagnostics/clear');
        await sendCommand('presentation');
    };

    const handleResetDemo = async () => {
        setChartData([]);
        await sendCommand('reset');
        await sendCommand('diagnostics/clear');
        await fetchStatus();
    };

    const handleReplaySpeedChange = (speed: number) => {
        setReplaySpeed(speed);
        void sendCommand('replay/speed', { speed });
    };

    const handleHardwareConnect = () => {
        sendCommand('hardware/connect', {
            port: hardwarePort,
            baud_rate: Number(hardwareBaudRate) || 115200,
            target: hardwareTarget,
            experiment: {
                ldr1_pin: ldr1Pin,
                ldr2_pin: ldr2Pin,
                led1_pin: led1Pin,
                led2_pin: led2Pin,
                redundancy_threshold: Number(redundancyThreshold) || 280,
            },
            simulation: {
                start_pin: Number(megaStartPin) || 22,
                node_count: Number(megaNodeCount) || 28,
                active_bit: megaActiveBit,
                sleep_bit: megaSleepBit,
            },
        });
    };

    const fetchStatusSnapshot = async () => {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (!response.ok) throw new Error('status request failed');
        return response.json() as Promise<Status>;
    };

    const postCommand = async (command: string, body: object = {}) => {
        const response = await fetch(`${API_BASE_URL}/${command}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(`${command} request failed`);
    };

    const waitForFinishedRun = async () => {
        for (let attempt = 0; attempt < 360; attempt++) {
            await sleep(1000);
            const snapshot = await fetchStatusSnapshot();
            if (snapshot.current_phase === 'finished' || snapshot.current_phase === 'error') {
                return snapshot;
            }
        }
        throw new Error('runtime comparison timed out');
    };

    const handleRuntimeComparison = async () => {
        if (runtimeComparisonRunning || status.is_running) return;
        setRuntimeComparisonRunning(true);
        setRuntimeComparisonRows([]);
        try {
            const response = await fetch(`${API_BASE_URL}/runtime/acceleration-comparison`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sensor_count: 128, step_count: 2000, epochs: 20, max_pairs: 10000 }),
            });
            if (!response.ok) throw new Error('acceleration comparison request failed');
            const data = await response.json() as {
                rows?: Array<{
                    backend: string;
                    training_seconds: number;
                    speedup_vs_pytorch: number;
                    purpose?: string;
                    final_loss?: number;
                    device?: string;
                }>;
            };
            if (data.rows?.length) {
                setRuntimeComparisonRows(data.rows.map(row => ({
                    mode: row.backend,
                    backend: row.device || row.backend,
                    runtimeSeconds: row.training_seconds,
                    trainingSeconds: row.training_seconds,
                    speedup: row.speedup_vs_pytorch,
                    powerSaved: 0,
                    recall: 0,
                    mse: row.final_loss || 0,
                    purpose: row.purpose,
                })));
                return;
            }
            const rows: RuntimeComparisonRow[] = [];
            const modes: BackendModeKey[] = ['pythonReference', 'cpuCpp', 'autoCuda'];
            const basePayload = {
                BENCH_SENSORS: 512,
                BENCH_STEPS: 160,
                BENCH_EPOCHS: 16,
                BENCH_MAX_PAIRS: 18000,
                CELL8_SHOW_PLOTS: false,
                SAFETY_EPOCHS: 6,
                AURA_MIN_ACTIVE_FRACTION: 0.18,
                AURA_MAX_ACTIVE_FRACTION: 0.26,
                AURA_SHADOW_SAMPLE_RATE: 0.05,
                AURA_GLOBAL_RETRAIN_PERIOD_FRACTION: 0.50,
                REPLAY_SPEED: 8,
            };

            for (const mode of modes) {
                await postCommand('reset');
                await sleep(600);
                await postCommand('start', {
                    ...basePayload,
                    ...backendModes[mode].payload,
                });
                const snapshot = await waitForFinishedRun();
                const aura = snapshot.policy_metrics?.['Intelligent AURA'];
                const runtimeSeconds = snapshot.runtime?.elapsed_seconds || 0;
                const trainingSeconds = snapshot.training?.seconds || 0;
                const pythonTraining = rows[0]?.trainingSeconds || trainingSeconds || 1;
                rows.push({
                    mode: backendModes[mode].label,
                    backend: snapshot.backend_mode || 'pending',
                    runtimeSeconds,
                    trainingSeconds,
                    speedup: trainingSeconds > 0 ? pythonTraining / trainingSeconds : 0,
                    powerSaved: aura?.power_saved_pct ?? snapshot.power_saved_percent,
                    recall: aura?.anomaly_recall_pct ?? snapshot.metrics?.anomaly_recall_pct ?? 0,
                    mse: aura?.global_reconstruction_mse ?? snapshot.metrics?.global_reconstruction_mse ?? 0,
                });
                setRuntimeComparisonRows([...rows]);
            }
            await fetchStatus();
        } catch (error) {
            console.error('Runtime comparison failed:', error);
        } finally {
            setRuntimeComparisonRunning(false);
        }
    };

    return (
        <main className="min-h-screen bg-[#070B10] px-4 py-4 text-[#E6EDF5] md:px-6">
            <div className="mx-auto flex max-w-[1600px] flex-col gap-4">
                <header className="flex flex-col gap-3 rounded-lg border border-[#223044] bg-[#0E1520] px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
                    <div className="flex items-center gap-3">
                        <div className="grid h-11 w-11 place-items-center rounded-lg border border-[#14B8A6]/40 bg-[#14B8A6]/10 text-[#14B8A6]">
                            <Satellite size={24} />
                        </div>
                        <div>
                            <h1 className="text-xl font-semibold tracking-tight text-white md:text-2xl">AURA Gateway</h1>
                            <p className="text-sm text-[#8EA3B8]">Sensor-network control</p>
                        </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-2 font-mono text-xs uppercase tracking-[0.14em] text-[#38BDF8]">
                            {displayBackendMode(status.backend_mode)}
                        </span>
                        {!presentationMode && <span className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-2 font-mono text-xs uppercase tracking-[0.14em] text-[#8EA3B8]">
                            {transport}
                        </span>}
                        <ReplaySpeedControl speed={effectiveReplaySpeed} disabled={false} onChange={handleReplaySpeedChange} />
                        <button
                            onClick={() => setPresentationMode(value => !value)}
                            className={`rounded-md border px-3 py-2 text-sm font-semibold transition ${
                                presentationMode
                                    ? 'border-[#22C55E]/50 bg-[#22C55E]/10 text-[#86EFAC]'
                                    : 'border-[#223044] bg-[#121C2A] text-[#8EA3B8] hover:border-[#64748B] hover:text-[#E6EDF5]'
                            }`}
                        >
                            {presentationMode ? 'Presentation' : 'Technical'}
                        </button>
                        <button
                            onClick={handlePresentationModeStart}
                            disabled={status.is_running}
                            className="rounded-md border border-[#22C55E]/50 bg-[#22C55E]/10 px-3 py-2 text-sm font-semibold text-[#86EFAC] transition hover:bg-[#22C55E]/18 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Presentation Mode
                        </button>
                        <button
                            onClick={handleResetDemo}
                            className="rounded-md border border-[#223044] bg-[#121C2A] px-3 py-2 text-sm font-semibold text-[#8EA3B8] transition hover:border-[#64748B] hover:text-[#E6EDF5]"
                        >
                            Reset Demo
                        </button>
                        <button onClick={handleStartPause} className="inline-flex items-center gap-2 rounded-md bg-[#38BDF8] px-4 py-2 text-sm font-semibold text-[#03111C] transition hover:bg-[#7DD3FC]">
                            {status.is_running ? <Pause size={16} /> : <Play size={16} />}
                            {status.is_running ? 'Pause' : 'Start'}
                        </button>
                    </div>
                </header>

                {status.error && (
                    <div className="rounded-lg border border-[#EF4444]/50 bg-[#EF4444]/10 px-4 py-3 text-sm text-[#FCA5A5]">{status.error}</div>
                )}

                <nav className="flex flex-wrap items-center gap-2 rounded-lg border border-[#223044] bg-[#0E1520] p-2">
                    <div className="flex flex-wrap gap-2">
                        {appTabs.filter(tab => !tab.technicalOnly || !presentationMode).map(tab => {
                            const active = activeTab === tab.key;
                            return (
                                <button
                                    key={tab.key}
                                    onClick={() => setActiveTab(tab.key)}
                                    className={`rounded-md px-4 py-2 text-sm font-semibold transition ${
                                        active
                                            ? 'bg-[#38BDF8] text-[#03111C]'
                                            : 'bg-[#0A111C] text-[#8EA3B8] hover:bg-[#121C2A] hover:text-[#E6EDF5]'
                                    }`}
                                >
                                    {tab.label}
                                </button>
                            );
                        })}
                    </div>
                    <div className="ml-auto flex flex-wrap gap-1 rounded-md border border-[#223044] bg-[#0A111C] p-1">
                        {(Object.keys(backendModes) as BackendModeKey[]).map(key => {
                            const active = selectedBackendMode === key;
                            return (
                                <button
                                    key={key}
                                    onClick={() => setSelectedBackendMode(key)}
                                    disabled={status.is_running}
                                    title={backendModes[key].description}
                                    className={`rounded px-2.5 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                                        active
                                            ? 'bg-[#38BDF8] text-[#03111C]'
                                            : 'text-[#8EA3B8] hover:bg-[#121C2A] hover:text-[#E6EDF5]'
                                    }`}
                                >
                                    {backendModes[key].shortLabel}
                                </button>
                            );
                        })}
                    </div>
                </nav>

                {activeTab === 'live' && presentationMode && <PresentationImpactStrip status={status} policies={policies} />}

                {activeTab === 'live' && !presentationMode && <GuidedDemoPanel status={status} selectedSensorId={effectiveSelectedSensorId} policyCount={Object.keys(policies).length} />}

                {activeTab === 'live' && !presentationMode && <section className="rounded-lg border border-[#223044] bg-[#0E1520] px-4 py-3">
                        <div className="mb-3 flex flex-col gap-1 md:flex-row md:items-end md:justify-between">
                            <div>
                                <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[#8EA3B8]">Presentation Preset</h2>
                                <p className="mt-1 text-sm text-[#8EA3B8]">{demoPresets[selectedPreset].description}</p>
                            </div>
                            <div className="font-mono text-xs text-[#5F7288]">
                                Sensors {demoPresets[selectedPreset].payload.BENCH_SENSORS} | Steps {demoPresets[selectedPreset].payload.BENCH_STEPS} | Epochs {demoPresets[selectedPreset].payload.BENCH_EPOCHS} | Backend {backendModes[selectedBackendMode].label}
                            </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {(Object.keys(demoPresets) as DemoPresetKey[]).map(key => {
                                const active = selectedPreset === key;
                                return (
                                    <button
                                        key={key}
                                        onClick={() => setSelectedPreset(key)}
                                        disabled={status.is_running}
                                        className={`rounded-md border px-3 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                                            active
                                                ? 'border-[#38BDF8] bg-[#38BDF8]/15 text-[#7DD3FC]'
                                                : 'border-[#223044] bg-[#0A111C] text-[#8EA3B8] hover:border-[#64748B] hover:text-[#E6EDF5]'
                                        }`}
                                    >
                                        {demoPresets[key].label}
                                    </button>
                                );
                            })}
                        </div>
                </section>}

                {activeTab === 'playground' && <PlaygroundStudio
                    status={status}
                    controls={whatIfControls}
                    challenge={challengeSettings}
                    selectedScenario={selectedScenario}
                    selectedQuestion={selectedQuestion}
                    selectedBackendMode={selectedBackendMode}
                    selectedPreset={selectedPreset}
                    runtimeRows={runtimeComparisonRows}
                    runtimeComparisonRunning={runtimeComparisonRunning}
                    advancedOpen={playgroundAdvancedOpen}
                    onScenario={handleScenarioSelect}
                    onQuestion={handleQuestionSelect}
                    onControls={setWhatIfControls}
                    onChallenge={setChallengeSettings}
                    onBackendMode={setSelectedBackendMode}
                    onAdvancedOpen={setPlaygroundAdvancedOpen}
                    onRun={handleStartPause}
                    onRerun={handlePlaygroundRerun}
                    onReset={handleResetDemo}
                    onCompare={handleRuntimeComparison}
                />}

                {activeTab === 'live' && <section className={`grid grid-cols-1 gap-4 ${presentationMode ? 'xl:grid-cols-[minmax(0,1.35fr)_minmax(420px,0.75fr)]' : 'xl:grid-cols-[minmax(0,1.55fr)_minmax(360px,0.75fr)]'}`}>
                    <Panel
                        title="Live Prototype Digital Twin"
                        className="overflow-hidden"
                        action={
                            <div className="flex rounded-md border border-[#223044] bg-[#0A111C] p-1">
                                {[
                                    ['aura', 'AURA Optimized'],
                                    ['allActive', 'All Active'],
                                ].map(([key, label]) => (
                                    <button
                                        key={key}
                                        onClick={() => setFarmView(key as 'aura' | 'allActive')}
                                        className={`rounded px-2.5 py-1.5 text-xs font-semibold transition ${
                                            farmView === key
                                                ? 'bg-[#38BDF8] text-[#03111C]'
                                                : 'text-[#8EA3B8] hover:bg-[#121C2A] hover:text-[#E6EDF5]'
                                        }`}
                                    >
                                        {label}
                                    </button>
                                ))}
                            </div>
                        }
                    >
                        <div className={`${compactMode ? 'h-[340px] md:h-[420px]' : presentationMode ? 'h-[520px] md:h-[660px]' : 'h-[420px] md:h-[520px]'} relative cursor-grab bg-[#030712] active:cursor-grabbing`}>
                            <AnomalyAlert status={status} />
                            <button
                                onClick={() => setCompactMode(value => !value)}
                                className={`absolute right-4 top-4 z-10 rounded-md border px-3 py-2 text-xs font-semibold shadow-[0_12px_30px_rgba(0,0,0,0.35)] transition ${
                                    compactMode
                                        ? 'border-[#FACC15]/60 bg-[#FACC15]/15 text-[#FDE68A]'
                                        : 'border-[#223044] bg-[#0A111C]/90 text-[#8EA3B8] hover:border-[#64748B] hover:text-[#E6EDF5]'
                                }`}
                            >
                                {compactMode ? 'Compact On' : 'Compact'}
                            </button>
                            <Suspense fallback={<div className="grid h-full place-items-center text-[#8EA3B8]">Loading scene</div>}>
                                <FarmScene
                                    sensors={farmSensors}
                                    sensorDetails={farmSensorDetails}
                                    selectedSensorId={effectiveSelectedSensorId}
                                    retrainActive={Boolean(status.retrain_policy?.required || status.retrain_policy?.recommended)}
                                    onSelectSensor={setSelectedSensorId}
                                />
                            </Suspense>
                        </div>
                        <div className="flex flex-wrap items-center gap-4 border-t border-[#223044] px-4 py-3 text-xs text-[#8EA3B8]">
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#38BDF8]" /> Active</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#64748B]" /> Sleeping</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#EF4444]" /> Anomaly</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full border border-[#60A5FA]" /> Shadow</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#FACC15]" /> Gateway</span>
                            <span className="ml-auto hidden text-[#5F7288] md:inline">Click a sensor.</span>
                        </div>
                    </Panel>

                    <div className="grid gap-4">
                        {!presentationMode && <div className="grid grid-cols-2 gap-3">
                            <TelemetryCard label="Power Saved" value={`${fmt(status.power_saved_percent)}%`} icon={<Zap size={18} />} accent="#22C55E" sub="replayed active mask" />
                            <TelemetryCard label="Active Sensors" value={`${fmt(activePct)}%`} icon={<Activity size={18} />} accent="#38BDF8" sub={`${status.active_sensors} / ${status.total_sensors}`} />
                            <TelemetryCard label="Anomaly Recall" value={`${fmt(status.metrics?.anomaly_recall_pct)}%`} icon={<Radar size={18} />} accent="#F59E0B" sub="Intelligent AURA" />
                            <TelemetryCard label="Final Loss" value={fmt(status.training?.final_loss, 5)} icon={<BrainCircuit size={18} />} accent="#A78BFA" sub={`${status.training?.epochs || 0} epochs`} />
                        </div>}
                        <Panel title={presentationMode ? 'AURA vs Baselines' : 'Algorithm State'}>
                            {presentationMode ? <PolicyFaceoff policies={policies} status={status} /> : <AlgorithmState status={status} />}
                        </Panel>
                        {presentationMode && <Panel title="Quality Guardrails">
                            <QualityGuardrails status={status} policies={policies} />
                        </Panel>}
                        {presentationMode && <Panel title="Latest Sensor Stream">
                            <LiveSensorStream status={status} />
                        </Panel>}
                        {presentationMode && <Panel title="Live Event Log">
                            <LiveEventLog status={status} />
                        </Panel>}
                        <Panel title="Sensor Inspector">
                            <SensorInspector status={status} selectedSensorId={effectiveSelectedSensorId} />
                        </Panel>
                        <Panel title={presentationMode ? 'Live Arduino Command' : 'Hardware Bridge'}>
                            {presentationMode ? <ArduinoCommandStream status={status} /> : <HardwarePanel status={status} />}
                        </Panel>
                    </div>
                </section>}

                {activeTab === 'benchmarks' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                    <Panel title="Benchmark Evidence">
                        <BenchmarkTable policies={policies} />
                    </Panel>

                    <Panel title="Power Saving vs Anomaly Recall">
                        <div className="flex flex-wrap gap-3 px-4 pt-4 text-xs text-[#8EA3B8]">
                            {policyRows.map(row => (
                                <span key={row.name} className="inline-flex items-center gap-2">
                                    <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: policyColors[row.name] || '#38BDF8' }} />
                                    {row.name}
                                </span>
                            ))}
                        </div>
                        <div className="h-[300px] p-4">
                            {policyRows.length ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 10, right: 18, bottom: 18, left: 0 }}>
                                        <CartesianGrid stroke="#223044" strokeDasharray="3 3" />
                                        <XAxis
                                            dataKey="power"
                                            name="Power saved"
                                            domain={[0, 100]}
                                            ticks={[0, 25, 50, 75, 100]}
                                            tickFormatter={value => `${fmt(Number(value), 0)}%`}
                                            stroke="#8EA3B8"
                                            tick={{ fill: '#8EA3B8', fontSize: 12 }}
                                        />
                                        <YAxis
                                            dataKey="recall"
                                            name="Anomaly recall"
                                            domain={[0, 100]}
                                            ticks={[0, 25, 50, 75, 100]}
                                            tickFormatter={value => `${fmt(Number(value), 0)}%`}
                                            stroke="#8EA3B8"
                                            tick={{ fill: '#8EA3B8', fontSize: 12 }}
                                        />
                                        <Tooltip
                                            cursor={{ strokeDasharray: '3 3' }}
                                            formatter={(value, name) => [`${fmt(Number(value))}%`, name === 'power' ? 'Power saved' : 'Anomaly recall']}
                                            labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ''}
                                            contentStyle={{ background: '#0E1520', border: '1px solid #223044', color: '#E6EDF5' }}
                                        />
                                        <Scatter data={policyRows} isAnimationActive={false}>
                                            {policyRows.map(row => <Cell key={row.name} fill={policyColors[row.name] || '#38BDF8'} />)}
                                        </Scatter>
                                    </ScatterChart>
                                </ResponsiveContainer>
                            ) : (
                                <EmptyState title="No points" detail="Run preset." />
                            )}
                        </div>
                    </Panel>
                </section>}

                {activeTab === 'benchmarks' && <Panel title="Python vs C++ vs CUDA/C++" action={<Cpu size={16} className="text-[#8EA3B8]" />}>
                    <RuntimeComparisonPanel rows={runtimeComparisonRows} isRunning={runtimeComparisonRunning || status.is_running} onRun={handleRuntimeComparison} />
                </Panel>}

                {activeTab === 'benchmarks' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                    <Panel title="Training Loss">
                        <div className="h-[280px] p-4">
                            {trainingLoss.length ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={trainingLoss}>
                                        <CartesianGrid stroke="#223044" strokeDasharray="3 3" />
                                        <XAxis dataKey="epoch" stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <YAxis stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <Tooltip contentStyle={{ background: '#0E1520', border: '1px solid #223044', color: '#E6EDF5' }} />
                                        <Line type="monotone" dataKey="loss" stroke="#A78BFA" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <EmptyState title="No loss curve" detail="Run AURA." />
                            )}
                        </div>
                    </Panel>

                    <Panel title="Policy Power And Recall" action={<AreaChart size={16} className="text-[#8EA3B8]" />}>
                        <div className="flex flex-wrap gap-4 px-4 pt-4 text-xs text-[#8EA3B8]">
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm bg-[#22C55E]" /> Power saved</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm bg-[#F59E0B]" /> Anomaly recall</span>
                        </div>
                        <div className="h-[280px] p-4">
                            {policyRows.length ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={policyRows}>
                                        <CartesianGrid stroke="#223044" strokeDasharray="3 3" />
                                        <XAxis dataKey="name" stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 11 }} interval={0} />
                                        <YAxis stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <Tooltip contentStyle={{ background: '#0E1520', border: '1px solid #223044', color: '#E6EDF5' }} />
                                        <Bar dataKey="power" name="Power saved %" fill="#22C55E" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                                        <Bar dataKey="recall" name="Anomaly recall %" fill="#F59E0B" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <EmptyState title="No bars" detail="Run benchmark." />
                            )}
                        </div>
                    </Panel>
                </section>}

                {activeTab === 'data' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.85fr)]">
                    <Panel title="Dataset Upload And Column Selection" action={<Database size={16} className="text-[#8EA3B8]" />}>
                        <DatasetPanel
                            datasets={datasets}
                            selectedDatasetId={selectedDatasetId}
                            selectedColumns={selectedColumns}
                            onSelectDataset={handleSelectDataset}
                            onToggleColumn={handleToggleColumn}
                            onUpload={handleUploadDataset}
                        />
                    </Panel>
                    <Panel title="Persistent Benchmark History">
                        <HistoryPanel runs={status.history} />
                    </Panel>
                    <Panel title="Selected Dataset Runtime Mapping" className="xl:col-span-2">
                        <div className="grid gap-3 p-4 text-sm leading-6 text-[#B8C7D8] md:grid-cols-3">
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Dataset</div>
                                {selectedDatasetId ? `Dataset #${selectedDatasetId}` : 'Synthetic/default'}
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Sensor Columns</div>
                                {selectedColumns.length} selected
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Storage</div>
                                {status.storage?.backend || 'local'}
                            </div>
                        </div>
                    </Panel>
                </section>}

                {activeTab === 'algorithm' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(360px,0.75fr)_minmax(0,1.25fr)]">
                    <Panel title="Algorithm State">
                        <AlgorithmState status={status} />
                    </Panel>
                    <Panel title="Training Loss">
                        <div className="h-[360px] p-4">
                            {trainingLoss.length ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={trainingLoss}>
                                        <CartesianGrid stroke="#223044" strokeDasharray="3 3" />
                                        <XAxis dataKey="epoch" stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <YAxis stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <Tooltip contentStyle={{ background: '#0E1520', border: '1px solid #223044', color: '#E6EDF5' }} />
                                        <Line type="monotone" dataKey="loss" stroke="#A78BFA" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <EmptyState title="No loss curve" detail="Run AURA." />
                            )}
                        </div>
                    </Panel>
                    <Panel title="Algorithm Design Evidence" className="xl:col-span-2">
                        <div className="grid gap-3 p-4 text-sm leading-6 text-[#B8C7D8] md:grid-cols-3">
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Learning</div>
                                Gradient descent learns redundancy thresholds, anomaly gates, and sleep behavior under an active-sensor budget.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Redundancy</div>
                                Pair-cache comparisons identify sensors that can sleep while preserving reconstruction and anomaly visibility.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Acceleration</div>
                                The same policy can run through Python, C++ CPU, or fused CUDA/C++ for runtime comparison.
                            </div>
                        </div>
                    </Panel>
                </section>}

                {activeTab === 'kernel' && <section className="grid grid-cols-1 gap-4">
                    <Panel title="Custom Kernel Proof" action={<Cpu size={16} className="text-[#8EA3B8]" />}>
                        <KernelProofPanel status={status} />
                    </Panel>
                    <Panel title="Python vs C++ vs CUDA/C++" action={<Cpu size={16} className="text-[#8EA3B8]" />}>
                        <RuntimeComparisonPanel rows={runtimeComparisonRows} isRunning={runtimeComparisonRunning || status.is_running} onRun={handleRuntimeComparison} />
                    </Panel>
                    <Panel title="How To Explain This To Panelists">
                        <div className="grid gap-3 p-4 text-sm leading-6 text-[#B8C7D8] md:grid-cols-3">
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Isolation</div>
                                Training operator only.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Correctness</div>
                                Checked against PyTorch.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Impact</div>
                                Speedup uses training time.
                            </div>
                        </div>
                    </Panel>
                </section>}

                {activeTab === 'hardware' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(420px,0.85fr)_minmax(0,1.15fr)]">
                    <Panel title="Hardware Configuration">
                        <div className="grid gap-3 p-4">
                            <div className="grid gap-3 md:grid-cols-3">
                                <label className="rounded-md bg-[#0A111C] px-3 py-3">
                                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Target</div>
                                    <select
                                        value={hardwareTarget}
                                        onChange={event => setHardwareTarget(event.target.value as HardwareTarget)}
                                        className="w-full rounded-md border border-[#223044] bg-[#050A12] px-2 py-2 font-mono text-sm text-[#E6EDF5] outline-none focus:border-[#38BDF8]"
                                    >
                                        <option value="esp32_experiment">ESP32 experiment</option>
                                        <option value="mega_simulation">Mega simulation</option>
                                    </select>
                                </label>
                                <label className="rounded-md bg-[#0A111C] px-3 py-3">
                                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">COM port</div>
                                    <select
                                        value={hardwarePort}
                                        onChange={event => setHardwarePort(event.target.value)}
                                        className="w-full rounded-md border border-[#223044] bg-[#050A12] px-2 py-2 font-mono text-sm text-[#E6EDF5] outline-none focus:border-[#38BDF8]"
                                    >
                                        {hardwarePorts.length ? (
                                            hardwarePorts.map(port => (
                                                <option key={port.device} value={port.device}>
                                                    {port.device}{port.description ? ` - ${port.description}` : ''}
                                                </option>
                                            ))
                                        ) : (
                                            <option value={hardwarePort}>No ports found</option>
                                        )}
                                    </select>
                                </label>
                                <label className="rounded-md bg-[#0A111C] px-3 py-3">
                                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#5F7288]">Baud</div>
                                    <select
                                        value={hardwareBaudRate}
                                        onChange={event => setHardwareBaudRate(event.target.value)}
                                        className="w-full rounded-md border border-[#223044] bg-[#050A12] px-2 py-2 font-mono text-sm text-[#E6EDF5] outline-none focus:border-[#38BDF8]"
                                    >
                                        <option value="9600">9600</option>
                                        <option value="57600">57600</option>
                                        <option value="115200">115200</option>
                                        <option value="230400">230400</option>
                                    </select>
                                </label>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                <button
                                    onClick={refreshHardwarePorts}
                                    disabled={hardwarePortsLoading}
                                    className="rounded-md border border-[#64748B]/40 bg-[#64748B]/10 px-3 py-2 text-sm font-semibold text-[#CBD5E1] transition hover:bg-[#64748B]/20 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    {hardwarePortsLoading ? 'Scanning...' : 'Refresh Ports'}
                                </button>
                                <button
                                    onClick={handleHardwareConnect}
                                    disabled={!hardwarePorts.length}
                                    className="rounded-md border border-[#38BDF8]/40 bg-[#38BDF8]/10 px-3 py-2 text-sm font-semibold text-[#7DD3FC] transition hover:bg-[#38BDF8]/20 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    Connect
                                </button>
                                <button
                                    onClick={() => sendCommand('hardware/sync')}
                                    className="rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-3 py-2 text-sm font-semibold text-[#86EFAC] transition hover:bg-[#22C55E]/20"
                                >
                                    Sync
                                </button>
                                <button
                                    onClick={() => sendCommand('hardware/disconnect')}
                                    className="rounded-md border border-[#EF4444]/40 bg-[#EF4444]/10 px-3 py-2 text-sm font-semibold text-[#FCA5A5] transition hover:bg-[#EF4444]/20"
                                >
                                    Disconnect
                                </button>
                            </div>
                            {hardwarePortsError && <div className="rounded-md border border-[#F59E0B]/45 bg-[#F59E0B]/10 px-3 py-2 text-xs text-[#FDE68A]">{hardwarePortsError}</div>}
                        </div>
                    </Panel>
                    <Panel title="Connection Status">
                        <HardwarePanel status={status} />
                    </Panel>
                    <Panel title={hardwareTarget === 'esp32_experiment' ? 'ESP32 Experiment Pins' : 'Mega Simulation Pins'}>
                        {hardwareTarget === 'esp32_experiment' ? (
                            <Esp32ExperimentPanel
                                ldr1Pin={ldr1Pin}
                                ldr2Pin={ldr2Pin}
                                led1Pin={led1Pin}
                                led2Pin={led2Pin}
                                threshold={redundancyThreshold}
                                onLdr1Pin={setLdr1Pin}
                                onLdr2Pin={setLdr2Pin}
                                onLed1Pin={setLed1Pin}
                                onLed2Pin={setLed2Pin}
                                onThreshold={setRedundancyThreshold}
                            />
                        ) : (
                            <MegaSimulationPanel
                                startPin={megaStartPin}
                                nodeCount={megaNodeCount}
                                activeBit={megaActiveBit}
                                sleepBit={megaSleepBit}
                                onStartPin={setMegaStartPin}
                                onNodeCount={setMegaNodeCount}
                                onActiveBit={setMegaActiveBit}
                                onSleepBit={setMegaSleepBit}
                            />
                        )}
                    </Panel>
                    <Panel title="Command Preview">
                        <ArduinoCommandStream
                            status={status}
                            startPin={Number(megaStartPin) || 22}
                            nodeCount={Math.max(1, Math.min(Number(megaNodeCount) || 28, 28))}
                        />
                    </Panel>
                </section>}

                {activeTab === 'diagnostics' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.85fr)]">
                    <Panel title="Diagnostics And Fallbacks" action={<AlertTriangle size={16} className="text-[#F59E0B]" />}>
                        <DiagnosticsPanel diagnostics={status.diagnostics} status={status} transport={transport} onRetrain={() => sendCommand('retrain')} />
                    </Panel>
                    <Panel title="Runtime Fallback Model">
                        <div className="grid gap-3 p-4 text-sm leading-6 text-[#B8C7D8]">
                            <div className="rounded-md bg-[#0A111C] p-4">
                                WebSocket, then polling.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                Native path falls back safely.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                Preview mode until connected.
                            </div>
                        </div>
                    </Panel>
                </section>}

                <footer className="grid grid-cols-1 gap-3 rounded-lg border border-[#223044] bg-[#0E1520] p-4 text-xs text-[#8EA3B8] md:grid-cols-3">
                    <div className="flex items-center gap-2"><Cpu size={15} className="text-[#22D3EE]" /> Backend: {status.backend_mode || 'pending'}</div>
                    <div className="flex items-center gap-2"><BrainCircuit size={15} className="text-[#A78BFA]" /> Stream: {transport} / {status.algorithm || 'refined_optimized_aura'}</div>
                    <div className="flex items-center gap-2"><Zap size={15} className="text-[#22C55E]" /> Hardware bridge: {status.hardware?.bridge_status || 'ready'} / {status.hardware?.arduino_status || 'not_connected'}</div>
                </footer>
            </div>
        </main>
    );
};

const Home: FC = () => (
    <Suspense fallback={<main className="grid min-h-screen place-items-center bg-[#070B10] text-[#E6EDF5]">Loading AURA</main>}>
        <ClientOnlyApp />
    </Suspense>
);

export default Home;
