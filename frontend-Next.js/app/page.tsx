'use client';

import { useState, Suspense, FC } from 'react';
import dynamic from 'next/dynamic';
import { Play, Pause, RotateCcw, BrainCircuit, AreaChart, ShieldCheck, History } from 'lucide-react';
import { useStatus } from './hooks/useStatus';
import MetricCard from './components/MetricCard';
import ChartsWindow from './components/ChartsWindow';
import type { Status } from './lib/types';

// Dynamically import the 3D scene to ensure it's client-side only
const FarmScene = dynamic(() => import('./components/FarmScene'), {
    ssr: false,
    loading: () => <div className="flex items-center justify-center h-full bg-gray-900">Loading 3D Scene...</div>
});

const getPhaseInfo = (phase: Status['current_phase'], isRunning: boolean) => {
    if (!isRunning) {
        return { text: 'IDLE', color: 'text-white', size: 'text-xl lg:text-2xl' };
    }
    switch(phase) {
        case 'collecting': 
            return { text: 'COLLECTING', color: 'text-yellow-400', size: 'text-xl lg:text-2xl' };
        case 'shadow_op': 
            return { text: 'SHADOW POWERSAVING', color: 'text-cyan-400', size: 'text-base sm:text-lg lg:text-[12px]' };
        default: 
            return { text: phase.toUpperCase().replace('_', ' '), color: 'text-white', size: 'text-xl lg:text-2xl' };
    }
}

const ClientOnlyApp: FC = () => {
    const { status, chartData, sendCommand, setChartData } = useStatus();
    
    // UI Control states
    const [isChartsVisible, setIsChartsVisible] = useState(false);
    
    // Simulation parameter states
    const [threshold, setThreshold] = useState(0.98);
    const [duration, setDuration] = useState(40);
    const [nWayComparison, setNWayComparison] = useState(2);
    const [shadowProb, setShadowProb] = useState(0.05);
    
    // Hybrid model parameter states
    const [hybridFidelityThreshold, setHybridFidelityThreshold] = useState(0.97);
    const [hybridMaxTimesteps, setHybridMaxTimesteps] = useState(2880);
    const [collectionPeriod, setCollectionPeriod] = useState(200);

    const handleStartPause = () => {
        if (status?.is_running) {
            sendCommand('pause');
        } else {
            setChartData([]); // Clear previous data on new start
            const payload = {
                threshold,
                duration,
                n_way_comparison: nWayComparison,
                shadow_mode_probability: shadowProb,
                hybrid_fidelity_threshold: hybridFidelityThreshold,
                hybrid_max_timesteps_since_retrain: hybridMaxTimesteps,
                collection_period: collectionPeriod
            };
            sendCommand('start', payload);
        }
    };

    const handleReset = () => {
        sendCommand('reset');
        setChartData([]);
    };

    if (!status) {
        return <div className="flex items-center justify-center h-screen bg-gray-900 text-white">Connecting to Simulation Server...</div>;
    }

    const phaseInfo = getPhaseInfo(status.current_phase, status.is_running);

    return (
        <main className="bg-gray-900 min-h-screen text-gray-200 font-sans relative">
            {isChartsVisible && <ChartsWindow data={chartData} onClose={() => setIsChartsVisible(false)} />}
            
            <div className="container mx-auto p-4 md:p-8 max-w-7xl">
                <header className="text-center mb-8">
                    <h1 className="text-4xl md:text-5xl font-bold text-white">AURA Intelligent Sensor Network</h1>
                    <p className="text-lg text-gray-400 mt-2">Live System with Autonomous Retraining</p>
                </header>

                <div className="w-full aspect-[16/7] rounded-2xl mb-8 cursor-grab active:cursor-grabbing overflow-hidden relative bg-black">
                    <Suspense fallback={<div className="flex items-center justify-center h-full bg-gray-900">Loading 3D Scene...</div>}>
                        <FarmScene sensors={status.sensors} />
                    </Suspense>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8 text-center">
                    <MetricCard label="TIMESTEP" value={status.timestep || 0} />
                    <MetricCard label="CURRENT PHASE" value={phaseInfo.text} colorClass={phaseInfo.color} valueClass={phaseInfo.size} />
                    <MetricCard label="ACTIVE SENSORS" value={`${status.active_sensors} / ${status.total_sensors}`} />
                    <MetricCard label="POWER SAVED (%)" value={(status.power_saved_percent || 0).toFixed(2)} unit="%" colorClass="text-green-400" />
                    <MetricCard label="LEARNER STATUS" value={(status.learner_status || 'idle').toUpperCase()} icon={<BrainCircuit size={16}/>} colorClass={status.learner_status === 'running' ? 'text-yellow-400 animate-pulse' : 'text-white'} />
                    <MetricCard label="FIDELITY" value={((status.fidelity || 1) * 100).toFixed(2)} unit="%" colorClass="text-blue-400" />
                </div>

                <div className="bg-gray-800/50 p-6 rounded-2xl border border-gray-700 flex flex-col gap-6">
                    {/* --- Main Controls --- */}
                    <div className="flex flex-wrap items-center justify-between gap-4">
                        <div className="flex items-center gap-4">
                            <button onClick={handleStartPause} className="px-5 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold text-white transition-all flex items-center gap-2 shadow-lg">
                                {status.is_running ? <Pause size={18} /> : <Play size={18} />} {status.is_running ? 'Pause' : 'Start'}
                            </button>
                            <button onClick={handleReset} className="px-5 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold text-white transition-all flex items-center gap-2 shadow-lg">
                                <RotateCcw size={18} /> Reset
                            </button>
                        </div>
                        <div className="flex items-center">
                            <button onClick={() => setIsChartsVisible(v => !v)} className="px-5 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold text-white transition-all flex items-center gap-2 shadow-lg">
                                <AreaChart size={18} /> Show Charts
                            </button>
                        </div>
                    </div>

                    {/* --- Core AURA Parameters --- */}
                     <div className="pt-4 border-t border-gray-700">
                        <h3 className="text-lg font-semibold mb-3 text-gray-200">Core AURA Parameters</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            <div className="flex flex-col">
                                <label htmlFor="threshold-slider" className="text-sm mb-1 text-gray-400">Initial Threshold: <span className="font-bold text-white">{(threshold).toFixed(4)}</span></label>
                                <input id="threshold-slider" type="range" min="0.9" max="1" step="0.0001" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full" disabled={status.is_running} />
                            </div>
                            <div className="flex flex-col">
                                <label htmlFor="duration-slider" className="text-sm mb-1 text-gray-400">Initial Duration: <span className="font-bold text-white">{duration}</span></label>
                                <input id="duration-slider" type="range" min="1" max="200" step="1" value={duration} onChange={(e) => setDuration(parseInt(e.target.value))} className="w-full" disabled={status.is_running} />
                            </div>
                            <div className="flex flex-col">
                                <label htmlFor="nway-input" className="text-sm mb-1 text-gray-400">N-Way Comparison</label>
                                <input id="nway-input" type="number" min="2" max="10" step="1" value={nWayComparison} onChange={(e) => setNWayComparison(parseInt(e.target.value, 10) || 2)} disabled={status.is_running} className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white disabled:opacity-50" />
                            </div>
                            <div className="flex flex-col">
                                <label htmlFor="shadow-prob-input" className="text-sm mb-1 text-gray-400">Shadow Probability</label>
                                <input id="shadow-prob-input" type="number" min="0.01" max="1" step="0.01" value={shadowProb} onChange={(e) => setShadowProb(parseFloat(e.target.value) || 0.01)} disabled={status.is_running} className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white disabled:opacity-50" />
                            </div>
                        </div>
                    </div>

                    {/* --- Autonomous Retraining Triggers --- */}
                    <div className="pt-4 border-t border-gray-700">
                        <h3 className="text-lg font-semibold mb-3 text-gray-200">Autonomous Retraining Triggers</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div className="flex flex-col">
                                <label htmlFor="hybrid-fidelity-slider" className="text-sm mb-1 text-gray-400 flex items-center gap-2">
                                    <ShieldCheck size={14}/> Fidelity Threshold: <span className="font-bold text-white">{(hybridFidelityThreshold * 100).toFixed(2)}%</span>
                                </label>
                                <input id="hybrid-fidelity-slider" type="range" min="0.90" max="0.9999" step="0.0001" value={hybridFidelityThreshold} onChange={(e) => setHybridFidelityThreshold(parseFloat(e.target.value))} className="w-full" disabled={status.is_running} />
                            </div>
                            <div className="flex flex-col">
                                <label htmlFor="hybrid-interval-input" className="text-sm mb-1 text-gray-400 flex items-center gap-2">
                                    <History size={14}/> Max Interval (Timesteps)
                                </label>
                                <input id="hybrid-interval-input" type="number" min="500" max="10000" step="100" value={hybridMaxTimesteps} onChange={(e) => setHybridMaxTimesteps(parseInt(e.target.value, 10) || 500)} disabled={status.is_running} className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white disabled:opacity-50" />
                            </div>
                             <div className="flex flex-col">
                                <label htmlFor="collection-period-input" className="text-sm mb-1 text-gray-400 flex items-center gap-2">
                                    <BrainCircuit size={14}/> Collection Period
                                </label>
                                <input id="collection-period-input" type="number" min="50" max="1000" step="50" value={collectionPeriod} onChange={(e) => setCollectionPeriod(parseInt(e.target.value, 10) || 50)} disabled={status.is_running} className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white disabled:opacity-50" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}

const Home: FC = () => {
    return (
        <Suspense fallback={<div className="flex items-center justify-center h-screen bg-gray-900 text-white">Loading Application...</div>}>
            <ClientOnlyApp />
        </Suspense>
    );
};

export default Home;