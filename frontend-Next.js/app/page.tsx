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
    Download,
    Pause,
    Play,
    Radar,
    RotateCcw,
    Satellite,
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
import type { BenchmarkRun, DatasetSummary, DiagnosticEntry, PolicyMetrics, Status } from './lib/types';

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
type AppTab = 'live' | 'benchmarks' | 'data' | 'algorithm' | 'kernel' | 'hardware' | 'diagnostics' | 'export';

const appTabs: { key: AppTab; label: string }[] = [
    { key: 'live', label: 'Live Prototype' },
    { key: 'benchmarks', label: 'Benchmarks' },
    { key: 'data', label: 'Data' },
    { key: 'algorithm', label: 'Algorithm' },
    { key: 'kernel', label: 'Kernel' },
    { key: 'hardware', label: 'Hardware' },
    { key: 'diagnostics', label: 'Diagnostics' },
    { key: 'export', label: 'Export' },
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
        title: 'Export Evidence',
        detail: 'Export results.',
    },
] as const;

const fmt = (value?: number, digits = 2) => Number.isFinite(value) ? Number(value).toFixed(digits) : '0.00';

const downloadText = (filename: string, content: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};

const csvEscape = (value: string | number) => `"${String(value).replace(/"/g, '""')}"`;
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const phaseLabel = (status: Status) => {
    if (status.current_phase === 'error') return 'ERROR';
    if (status.learner_status === 'running') return 'TRAINING';
    if (status.current_phase === 'shadow_op') return 'REPLAYING';
    if (status.current_phase === 'finished') return 'FINISHED';
    return status.is_running ? status.current_phase.toUpperCase() : 'IDLE';
};

const phaseColorClass = (status: Status) => {
    if (status.current_phase === 'error') return 'border-[#EF4444]/50 bg-[#EF4444]/10 text-[#FCA5A5]';
    if (status.learner_status === 'running' || status.current_phase === 'collecting') return 'border-[#A78BFA]/50 bg-[#A78BFA]/10 text-[#C4B5FD]';
    if (status.current_phase === 'shadow_op') return 'border-[#38BDF8]/50 bg-[#38BDF8]/10 text-[#7DD3FC]';
    if (status.current_phase === 'finished') return 'border-[#22C55E]/50 bg-[#22C55E]/10 text-[#86EFAC]';
    return 'border-[#223044] bg-[#0A111C] text-[#8EA3B8]';
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
        ['Global MSE', 'global_reconstruction_mse', ''],
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
    const bandwidthSaved = aura?.power_saved_pct ?? status.power_saved_percent;
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

const PolicyFaceoff: FC<{ policies: Record<string, PolicyMetrics> }> = ({ policies }) => {
    const aura = policies['Intelligent AURA'];
    const baselines = [
        ['LoRaWAN', policies['LoRaWAN-style'], '#FDBA74'],
        ['NB-IoT', policies['NB-IoT-style'], '#2DD4BF'],
    ].filter(([, metrics]) => metrics) as [string, PolicyMetrics, string][];
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

const ArduinoCommandStream: FC<{ status: Status }> = ({ status }) => {
    const hardware = status.hardware;
    const command = hardware?.last_command || 'pending';
    const ack = hardware?.last_ack || (hardware?.arduino_status === 'connected' ? 'waiting for acknowledgement' : 'preview mode');

    return (
        <div className="grid gap-3 p-4">
            <div className="rounded-md border border-[#223044] bg-[#050A12] p-3 font-mono text-xs leading-6">
                <div className="text-[#38BDF8]">AURA -&gt; Mega: <span className="break-all text-[#22C55E]">{command}</span></div>
                <div className="text-[#A78BFA]">Mega -&gt; AURA: <span className="text-[#E6EDF5]">{ack}</span></div>
            </div>
            <div className="grid grid-cols-7 gap-1.5">
                {(command || '').split(',').slice(0, 28).map((bit, index) => {
                    const active = bit.trim() === '0';
                    return (
                        <div key={`${index}-${bit}`} className={`rounded border px-1.5 py-1 text-center font-mono text-[10px] ${active ? 'border-[#38BDF8]/50 bg-[#38BDF8]/15 text-[#7DD3FC]' : 'border-[#64748B]/35 bg-[#64748B]/10 text-[#94A3B8]'}`}>
                            <div>{22 + index}</div>
                            <div className="text-[9px]">{active ? 'ON' : 'OFF'}</div>
                        </div>
                    );
                })}
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
};

const RuntimeComparisonPanel: FC<{
    rows: RuntimeComparisonRow[];
    isRunning: boolean;
    onRun: () => void;
}> = ({ rows, isRunning, onRun }) => (
    <div className="grid gap-4 p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <p className="text-sm leading-6 text-[#8EA3B8]">Controlled quick benchmark across backends.</p>
            <button
                onClick={onRun}
                disabled={isRunning}
                className="rounded-md border border-[#38BDF8]/45 bg-[#38BDF8]/10 px-4 py-2 text-sm font-semibold text-[#7DD3FC] transition hover:bg-[#38BDF8]/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
                {isRunning ? 'Comparing...' : 'Run Python vs C++ vs CUDA/C++'}
            </button>
        </div>
        {rows.length ? (
            <div className="overflow-x-auto rounded-md border border-[#223044]">
                <table className="w-full min-w-[760px] border-collapse text-left text-sm">
                    <thead>
                        <tr className="border-b border-[#223044] text-xs uppercase tracking-[0.12em] text-[#8EA3B8]">
                            <th className="px-4 py-3 font-semibold">Requested Mode</th>
                            <th className="px-4 py-3 font-semibold">Actual Backend</th>
                            <th className="px-4 py-3 font-semibold">Runtime</th>
                            <th className="px-4 py-3 font-semibold">Training</th>
                            <th className="px-4 py-3 font-semibold">Speedup</th>
                            <th className="px-4 py-3 font-semibold">Power</th>
                            <th className="px-4 py-3 font-semibold">Recall</th>
                            <th className="px-4 py-3 font-semibold">MSE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map(row => (
                            <tr key={row.mode} className="border-b border-[#182335] last:border-b-0">
                                <td className="px-4 py-3 font-semibold text-[#E6EDF5]">{row.mode}</td>
                                <td className="px-4 py-3 text-[#B8C7D8]">{row.backend}</td>
                                <td className="px-4 py-3 font-mono text-[#E6EDF5]">{fmt(row.runtimeSeconds, 3)}s</td>
                                <td className="px-4 py-3 font-mono text-[#E6EDF5]">{fmt(row.trainingSeconds, 3)}s</td>
                                <td className="px-4 py-3 font-mono text-[#22C55E]">{fmt(row.speedup, 2)}x</td>
                                <td className="px-4 py-3 font-mono text-[#22C55E]">{fmt(row.powerSaved)}%</td>
                                <td className="px-4 py-3 font-mono text-[#F59E0B]">{fmt(row.recall)}%</td>
                                <td className="px-4 py-3 font-mono text-[#A78BFA]">{fmt(row.mse, 5)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        ) : (
            <EmptyState title="No runtime comparison" detail="Run comparison." />
        )}
    </div>
);

const AlgorithmState: FC<{ status: Status }> = ({ status }) => {
    const learned = status.learned_parameters || {};
    const runtime = status.runtime;
    const training = status.training;
    const budget = status.active_budget_band || [20, 30];
    const items = [
        ['Backend', status.backend_mode || 'pending'],
        ['Run budget', `${fmt(budget[0], 0)}-${fmt(budget[1], 0)}% active`],
        ['Training time', `${fmt(training?.seconds, 3)}s`],
        ['Total runtime', `${fmt(runtime?.elapsed_seconds, 3)}s`],
        ['Final loss', fmt(training?.final_loss, 6)],
        ['Sensors / steps', `${runtime?.bench_sensors || status.total_sensors} / ${runtime?.bench_steps || 0}`],
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
    const items = [
        ['Bridge', hardware?.bridge_status || 'ready'],
        ['Arduino', hardware?.arduino_status || 'not_connected'],
        ['COM port', hardware?.com_port || 'COM16'],
        ['Baud', hardware?.baud_rate || 115200],
        ['Active nodes', hardware?.active_nodes ?? status.active_sensors],
        ['Sleeping nodes', hardware?.sleeping_nodes ?? Math.max(0, status.sensors.length - status.active_sensors)],
        ['Last sync', hardware?.last_sync || 'pending'],
    ];

    return (
        <div className="grid gap-3 p-4">
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
            <p className="text-xs leading-5 text-[#5F7288]">{hardware?.note || 'Preview of the Arduino sleep/wake command stream.'}</p>
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
        ['CUDA preferred', proof.status.cuda_preferred],
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

const DiagnosticsPanel: FC<{ diagnostics?: DiagnosticEntry[]; status: Status; transport: string }> = ({ diagnostics = [], status, transport }) => {
    const fallbackReason = status.kernel_proof?.status.fallback_reason;
    const hardwareError = status.hardware?.last_error;
    const rows = [
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
    const [selectedBackendMode, setSelectedBackendMode] = useState<BackendModeKey>('autoCuda');
    const [presentationMode, setPresentationMode] = useState(true);
    const [compactMode, setCompactMode] = useState(false);
    const [farmView, setFarmView] = useState<'aura' | 'allActive'>('aura');
    const [runtimeComparisonRows, setRuntimeComparisonRows] = useState<RuntimeComparisonRow[]>([]);
    const [runtimeComparisonRunning, setRuntimeComparisonRunning] = useState(false);
    const [activeTab, setActiveTab] = useState<AppTab>(getInitialTab);
    const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
    const [selectedColumns, setSelectedColumns] = useState<string[]>([]);

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
        return () => {
            cancelled = true;
        };
    }, []);

    useEffect(() => {
        if (!selectedDatasetId && datasets.length) {
            const dataset = datasets[0];
            setSelectedDatasetId(dataset.id);
            setSelectedColumns(dataset.selected_columns?.length ? dataset.selected_columns : (dataset.numeric_columns || []).slice(0, 64));
        }
    }, [datasets, selectedDatasetId]);

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

    const handleStartPause = () => {
        if (status.is_running) {
            sendCommand('pause');
            return;
        }
        setChartData([]);
        const payload = {
            ...demoPresets[selectedPreset].payload,
            ...backendModes[selectedBackendMode].payload,
        };
        if (selectedDatasetId && selectedColumns.length >= 2) {
            Object.assign(payload, {
                DATASET_ID: selectedDatasetId,
                DATASET_COLUMNS: selectedColumns,
            });
        }
        sendCommand('start', payload);
    };

    const handleReset = () => {
        sendCommand('reset');
        setChartData([]);
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
            const rows: RuntimeComparisonRow[] = [];
            const modes: BackendModeKey[] = ['pythonReference', 'cpuCpp', 'autoCuda'];
            const basePayload = {
                ...demoPresets.quick.payload,
                BENCH_EPOCHS: Math.min(4, demoPresets.quick.payload.BENCH_EPOCHS),
                SAFETY_EPOCHS: 4,
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
                const pythonRuntime = rows[0]?.runtimeSeconds || runtimeSeconds || 1;
                rows.push({
                    mode: backendModes[mode].label,
                    backend: snapshot.backend_mode || 'pending',
                    runtimeSeconds,
                    trainingSeconds: snapshot.training?.seconds || 0,
                    speedup: runtimeSeconds > 0 ? pythonRuntime / runtimeSeconds : 0,
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

    const exportJson = () => {
        downloadText('aura-dashboard-status.json', JSON.stringify(status, null, 2), 'application/json');
    };

    const exportCsv = () => {
        const headers = ['policy', 'power_saved_pct', 'bandwidth_saved_pct', 'active_sensor_pct', 'anomaly_recall_pct', 'global_reconstruction_mse'];
        const rows = Object.entries(policies).map(([name, metrics]) => [
            name,
            fmt(metrics.power_saved_pct, 6),
            fmt(metrics.power_saved_pct, 6),
            fmt(metrics.active_sensor_pct, 6),
            fmt(metrics.anomaly_recall_pct, 6),
            fmt(metrics.global_reconstruction_mse, 8),
        ]);
        const csv = [headers, ...rows].map(row => row.map(csvEscape).join(',')).join('\n');
        downloadText('aura-benchmark-comparison.csv', csv, 'text/csv');
    };

    const exportSummary = () => {
        const aura = policies['Intelligent AURA'];
        const lorawan = policies['LoRaWAN-style'];
        const fair = policies['Budget-matched LoRaWAN'];
        const nbiot = policies['NB-IoT-style'];
        const summary = [
            '# AURA Prototype Benchmark Summary',
            '',
            `Backend: ${status.backend_mode || 'pending'}`,
            `Algorithm: ${status.algorithm || 'refined_optimized_aura'}`,
            `Preset: ${demoPresets[selectedPreset].label}`,
            `Requested backend mode: ${backendModes[selectedBackendMode].label}`,
            `Sensors: ${status.runtime?.bench_sensors || status.total_sensors}`,
            `Timesteps: ${status.runtime?.bench_steps || 0}`,
            `Epochs: ${status.training?.epochs || 0}`,
            `Training time: ${fmt(status.training?.seconds, 3)} s`,
            `Total runtime: ${fmt(status.runtime?.elapsed_seconds, 3)} s`,
            `Final loss: ${fmt(status.training?.final_loss, 6)}`,
            '',
            '## Intelligent AURA',
            `Power saved: ${fmt(aura?.power_saved_pct)}%`,
            `Bandwidth saved: ${fmt(aura?.power_saved_pct)}%`,
            `Active sensors: ${fmt(aura?.active_sensor_pct)}%`,
            `Anomaly recall: ${fmt(aura?.anomaly_recall_pct)}%`,
            `Global reconstruction MSE: ${fmt(aura?.global_reconstruction_mse, 8)}`,
            '',
            '## Comparison',
            `LoRaWAN-style anomaly recall: ${fmt(lorawan?.anomaly_recall_pct)}%`,
            `LoRaWAN-style power saved: ${fmt(lorawan?.power_saved_pct)}%`,
            `Budget-matched LoRaWAN anomaly recall: ${fmt(fair?.anomaly_recall_pct)}%`,
            `Budget-matched LoRaWAN power saved: ${fmt(fair?.power_saved_pct)}%`,
            `NB-IoT-style anomaly recall: ${fmt(nbiot?.anomaly_recall_pct)}%`,
            `NB-IoT-style power saved: ${fmt(nbiot?.power_saved_pct)}%`,
            '',
            '## Presentation Claim',
            'AURA reduces sensor activity while preserving anomaly visibility and reconstruction quality.',
            '',
        ].join('\n');
        downloadText('aura-presentation-summary.md', summary, 'text/markdown');
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
                            <h1 className="text-xl font-semibold tracking-tight text-white md:text-2xl">AURA Gateway Telemetry</h1>
                            <p className="text-sm text-[#8EA3B8]">Sensor-network control</p>
                        </div>
                        <span className="hidden rounded-md border border-[#22C55E]/35 bg-[#22C55E]/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-[#86EFAC] md:inline-flex">
                            Live Prototype
                        </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-2 font-mono text-xs uppercase tracking-[0.14em] text-[#38BDF8]">
                            {status.backend_mode || 'Pending'}
                        </span>
                        {!presentationMode && <span className="rounded-md border border-[#223044] bg-[#0A111C] px-3 py-2 font-mono text-xs uppercase tracking-[0.14em] text-[#8EA3B8]">
                            {transport}
                        </span>}
                        <span className={`rounded-md border px-3 py-2 font-mono text-xs uppercase tracking-[0.14em] ${phaseColorClass(status)}`}>
                            {phaseLabel(status)}
                        </span>
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
                            onClick={() => setCompactMode(value => !value)}
                            className={`rounded-md border px-3 py-2 text-sm font-semibold transition ${
                                compactMode
                                    ? 'border-[#FACC15]/50 bg-[#FACC15]/10 text-[#FDE68A]'
                                    : 'border-[#223044] bg-[#121C2A] text-[#8EA3B8] hover:border-[#64748B] hover:text-[#E6EDF5]'
                            }`}
                        >
                            {compactMode ? 'Compact On' : 'Compact'}
                        </button>
                        <button onClick={handleStartPause} className="inline-flex items-center gap-2 rounded-md bg-[#38BDF8] px-4 py-2 text-sm font-semibold text-[#03111C] transition hover:bg-[#7DD3FC]">
                            {status.is_running ? <Pause size={16} /> : <Play size={16} />}
                            {status.is_running ? 'Pause' : 'Start'}
                        </button>
                        <button onClick={handleReset} className="inline-flex items-center gap-2 rounded-md border border-[#223044] bg-[#121C2A] px-4 py-2 text-sm font-semibold text-[#E6EDF5] transition hover:border-[#64748B]">
                            <RotateCcw size={16} /> Reset
                        </button>
                        {!presentationMode && <button onClick={exportJson} className="inline-flex items-center gap-2 rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-3 py-2 text-sm font-semibold text-[#22C55E] transition hover:bg-[#22C55E]/20">
                            <Download size={16} /> JSON
                        </button>}
                        {!presentationMode && <button onClick={exportCsv} className="inline-flex items-center gap-2 rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-3 py-2 text-sm font-semibold text-[#22C55E] transition hover:bg-[#22C55E]/20">
                            CSV
                        </button>}
                        <button onClick={exportSummary} className="inline-flex items-center gap-2 rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-3 py-2 text-sm font-semibold text-[#22C55E] transition hover:bg-[#22C55E]/20">
                            Summary
                        </button>
                    </div>
                </header>

                {status.error && (
                    <div className="rounded-lg border border-[#EF4444]/50 bg-[#EF4444]/10 px-4 py-3 text-sm text-[#FCA5A5]">{status.error}</div>
                )}

                <nav className="flex flex-wrap items-center gap-2 rounded-lg border border-[#223044] bg-[#0E1520] p-2">
                    <div className="flex flex-wrap gap-2">
                        {appTabs.map(tab => {
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
                        <div className={`${compactMode ? 'h-[340px] md:h-[420px]' : presentationMode ? 'h-[520px] md:h-[660px]' : 'h-[420px] md:h-[520px]'} cursor-grab bg-[#030712] active:cursor-grabbing`}>
                            <Suspense fallback={<div className="grid h-full place-items-center text-[#8EA3B8]">Loading scene</div>}>
                                <FarmScene
                                    sensors={displayedSensors}
                                    sensorDetails={status.sensor_details}
                                    selectedSensorId={effectiveSelectedSensorId}
                                    onSelectSensor={setSelectedSensorId}
                                />
                            </Suspense>
                        </div>
                        <div className="flex flex-wrap items-center gap-4 border-t border-[#223044] px-4 py-3 text-xs text-[#8EA3B8]">
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#38BDF8]" /> Active</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#64748B]" /> Sleeping</span>
                            <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#EF4444]" /> Anomaly</span>
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
                            {presentationMode ? <PolicyFaceoff policies={policies} /> : <AlgorithmState status={status} />}
                        </Panel>
                        {presentationMode && <Panel title="Quality Guardrails">
                            <QualityGuardrails status={status} policies={policies} />
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

                    <Panel title="Power vs Anomaly Recall">
                        <div className="h-[320px] p-4">
                            {policyRows.length ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 10, right: 18, bottom: 10, left: 0 }}>
                                        <CartesianGrid stroke="#223044" strokeDasharray="3 3" />
                                        <XAxis dataKey="power" name="Power saved" unit="%" stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <YAxis dataKey="recall" name="Anomaly recall" unit="%" stroke="#8EA3B8" tick={{ fill: '#8EA3B8', fontSize: 12 }} />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#0E1520', border: '1px solid #223044', color: '#E6EDF5' }} />
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

                    <Panel title="Runtime And Policy Comparison" action={<AreaChart size={16} className="text-[#8EA3B8]" />}>
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
                    <Panel title="Method Notes" className="xl:col-span-2">
                        <div className="grid gap-3 p-4 text-sm leading-6 text-[#B8C7D8] md:grid-cols-3">
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Learning</div>
                                Gradient descent trains AURA&apos;s thresholds, gates, and sleep behavior under an active-sensor budget.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Redundancy</div>
                                N=2 redundancy specialization.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8EA3B8]">Acceleration</div>
                                CPU/CUDA acceleration.
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

                {activeTab === 'hardware' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(360px,0.8fr)_minmax(0,1.2fr)]">
                    <Panel title="Hardware Bridge">
                        <div className="flex flex-wrap gap-2 border-b border-[#223044] px-4 py-3">
                            <button
                                onClick={() => sendCommand('hardware/connect', {
                                    port: status.hardware?.com_port || 'COM16',
                                    baud_rate: status.hardware?.baud_rate || 115200,
                                })}
                                className="rounded-md border border-[#38BDF8]/40 bg-[#38BDF8]/10 px-3 py-2 text-sm font-semibold text-[#7DD3FC] transition hover:bg-[#38BDF8]/20"
                            >
                                Connect
                            </button>
                            <button
                                onClick={() => sendCommand('hardware/sync')}
                                className="rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-3 py-2 text-sm font-semibold text-[#86EFAC] transition hover:bg-[#22C55E]/20"
                            >
                                Sync Command
                            </button>
                            <button
                                onClick={() => sendCommand('hardware/disconnect')}
                                className="rounded-md border border-[#EF4444]/40 bg-[#EF4444]/10 px-3 py-2 text-sm font-semibold text-[#FCA5A5] transition hover:bg-[#EF4444]/20"
                            >
                                Disconnect
                            </button>
                        </div>
                        <HardwarePanel status={status} />
                    </Panel>
                    <Panel title="Deployment Link">
                        <div className="grid gap-3 p-4 text-sm leading-6 text-[#B8C7D8]">
                            <div className="rounded-md bg-[#0A111C] p-4">
                                Gateway command stream.
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4 font-mono text-xs text-[#22C55E]">
                                {status.hardware?.last_command || 'pending'}
                            </div>
                            <div className="rounded-md bg-[#0A111C] p-4">
                                <span className="font-mono text-[#E6EDF5]">0</span> active/on, <span className="font-mono text-[#E6EDF5]">1</span> sleep/off.
                            </div>
                        </div>
                    </Panel>
                </section>}

                {activeTab === 'diagnostics' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.85fr)]">
                    <Panel title="Diagnostics And Fallbacks" action={<AlertTriangle size={16} className="text-[#F59E0B]" />}>
                        <DiagnosticsPanel diagnostics={status.diagnostics} status={status} transport={transport} />
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

                {activeTab === 'export' && <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(360px,0.8fr)_minmax(0,1.2fr)]">
                    <Panel title="Export Evidence">
                        <div className="grid gap-3 p-4">
                            <button onClick={exportJson} className="rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-4 py-3 text-left text-sm font-semibold text-[#22C55E] transition hover:bg-[#22C55E]/20">
                                Export raw dashboard JSON
                            </button>
                            <button onClick={exportCsv} className="rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-4 py-3 text-left text-sm font-semibold text-[#22C55E] transition hover:bg-[#22C55E]/20">
                                Export benchmark CSV
                            </button>
                            <button onClick={exportSummary} className="rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-4 py-3 text-left text-sm font-semibold text-[#22C55E] transition hover:bg-[#22C55E]/20">
                                Export presentation summary
                            </button>
                        </div>
                    </Panel>
                    <Panel title="Panel Claim">
                        <div className="p-4 text-sm leading-7 text-[#B8C7D8]">
                            AURA edge-gateway benchmark summary.
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
