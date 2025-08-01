'use client';

import { useState, useEffect, useRef, useCallback, Suspense, memo, FC, ReactNode } from 'react';
import dynamic from 'next/dynamic';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { Play, Pause, RotateCcw, BrainCircuit, AreaChart, X, Minus, ShieldCheck, History } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Draggable from 'react-draggable';
import { Resizable } from 're-resizable';

// --- TYPE DEFINITIONS (TypeScript) ---

interface Sensor {
    id: number;
    is_off: boolean;
}

interface Status {
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

interface ChartDataPoint {
    timestep: number;
    fidelity: number;
    powerSaved: number;
}

// --- Pre-calculated 3D coordinates for sensors ---
const SENSOR_3D_POSITIONS = [
    [-17, 0.6, -10], [-13, 0.6, 5], [-10, 0.6, -8], [-7, 0.6, 10],
    [-5, 0.6, -2], [-2, 0.6, 8], [0, 0.6, -11], [3, 0.6, 3],
    [5, 0.6, -5], [8, 0.6, 9], [11, 0.6, -9], [14, 0.6, 0],
    [17, 0.6, 7], [19, 0.6, -4], [-18, 0.6, 2], [-15, 0.6, -5],
    [-11, 0.6, 11], [-8, 0.6, -1], [-4, 0.6, 6], [-1, 0.6, -7],
    [2, 0.6, 1], [6, 0.6, -10], [9, 0.6, 5], [12, 0.6, -3],
    [15, 0.6, 10], [18, 0.6, -9], [-19, 0.6, 8], [19, 0.6, 11]
];

// --- 3D Sensor Component ---
const SensorNode: FC<{ id: number; isOff: boolean; position: [number, number, number] }> = ({ id, isOff, position }) => {
    const onColor = '#3b82f6';
    const offColor = '#4b5563';
    const emissiveIntensity = isOff ? 0 : 2.5;

    return (
        <mesh position={position} castShadow>
            <sphereGeometry args={[0.7, 32, 32]} />
            <meshStandardMaterial
                color={isOff ? offColor : onColor}
                emissive={isOff ? offColor : onColor}
                emissiveIntensity={emissiveIntensity}
                roughness={0.4}
                metalness={0.2}
            />
            <Text position={[0, 1.1, 0]} color="white" fontSize={0.6} anchorX="center" anchorY="middle" outlineWidth={0.02} outlineColor="black" >
                {id + 1}
            </Text>
        </mesh>
    );
};

// --- NEW COMPONENT: This contains the rotating group and the useFrame hook ---
const RotatingSceneContent: FC<{ sensors: Sensor[]; groundTexture: THREE.CanvasTexture | null }> = ({ sensors, groundTexture }) => {
    const sceneGroupRef = useRef<THREE.Group>(null);

    // This hook will run on every rendered frame
    useFrame((state, delta) => {
        if (sceneGroupRef.current) {
            // Slowly rotate the entire scene
            sceneGroupRef.current.rotation.y += delta * 0.05;
        }
    });

    return (
        <group ref={sceneGroupRef} position={[0, -1.0, 0]}>
             <group position={[0, -0.5, 0]}>
                <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                    <planeGeometry args={[45, 30]} />
                    <meshStandardMaterial map={groundTexture} color={!groundTexture ? '#166534' : undefined} roughness={1} />
                </mesh>
                <mesh position={[0, -1, 15]}><boxGeometry args={[45, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[0, -1, -15]}><boxGeometry args={[45, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[22.5, -1, 0]} rotation={[0, -Math.PI / 2, 0]}><boxGeometry args={[30, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[-22.5, -1, 0]} rotation={[0, Math.PI / 2, 0]}><boxGeometry args={[30, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[45, 30]} /><meshStandardMaterial color="#5d4037" /></mesh>
            </group>
            {sensors?.map((sensor) => (
                <SensorNode key={sensor.id} id={sensor.id} isOff={sensor.is_off} position={SENSOR_3D_POSITIONS[sensor.id] as [number, number, number]} />
            ))}
        </group>
    );
}

// --- 3D Scene Component ---
const FarmScene: FC<{ sensors: Sensor[] }> = memo(({ sensors }) => {
    const [groundTexture, setGroundTexture] = useState<THREE.CanvasTexture | null>(null);
    
    useEffect(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const context = canvas.getContext('2d');
        if (!context) return;

        // Create green grass gradient
        const grassGradient = context.createLinearGradient(0, 0, 0, 512);
        grassGradient.addColorStop(0, '#166534');
        grassGradient.addColorStop(1, '#14532d');
        context.fillStyle = grassGradient;
        context.fillRect(0, 0, 512, 512);

        // Add subtle texture to grass
        for (let i = 0; i < 10000; i++) {
            context.fillStyle = `rgba(0,0,0,${Math.random() * 0.1})`;
            context.fillRect(Math.random() * 512, Math.random() * 512, 2, 2);
        }

        // Add brown mud patches from the previous version
        const mudPatches = [
            { x: 0.2, y: 0.3, r: 0.2 }, { x: 0.7, y: 0.8, r: 0.25 }, { x: 0.8, y: 0.2, r: 0.15 },
        ];
        mudPatches.forEach(patch => {
            const mudGradient = context.createRadialGradient(
                patch.x * 512, patch.y * 512, 0,
                patch.x * 512, patch.y * 512, patch.r * 512
            );
            mudGradient.addColorStop(0, 'rgba(93, 64, 55, 0.7)');
            mudGradient.addColorStop(0.8, 'rgba(121, 85, 72, 0.4)');
            mudGradient.addColorStop(1, 'rgba(121, 85, 72, 0)');
            context.fillStyle = mudGradient;
            context.fillRect(0, 0, 512, 512);
        });

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        setGroundTexture(texture);
    }, []);

    return (
        <Canvas shadows camera={{ position: [-10, 20, 30], fov: 50 }} gl={{ alpha: true }} flat={true}>
            <ambientLight intensity={0.7} />
            <directionalLight position={[25, 30, 15]} intensity={1.5} castShadow shadow-mapSize-width={2048} shadow-mapSize-height={2048} shadow-camera-far={80} shadow-camera-left={-25} shadow-camera-right={25} shadow-camera-top={25} shadow-camera-bottom={-25} />
            {/* Render the new component inside the canvas */}
            <RotatingSceneContent sensors={sensors} groundTexture={groundTexture} />
            <OrbitControls enablePan={true} enableZoom={true} minPolarAngle={Math.PI / 8} maxPolarAngle={Math.PI / 2.1} minDistance={15} maxDistance={70} />
        </Canvas>
    );
});
FarmScene.displayName = 'FarmScene';

// --- UI Components ---
const MetricCard: FC<{ label: string; value: ReactNode; unit?: string; colorClass?: string; icon?: ReactNode; valueClass?: string }> = ({ label, value, unit, colorClass = 'text-white', icon, valueClass = 'text-xl lg:text-2xl' }) => (
    <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 h-full flex flex-col">
        <div className="flex items-center justify-center text-sm font-medium text-gray-400 mb-2">
            {icon}
            <span className="ml-2">{label}</span>
        </div>
        <div className="flex-grow flex items-center justify-center overflow-hidden">
             <div className={`${valueClass} font-bold ${colorClass}`}>{value}<span className="text-lg ml-1">{unit}</span></div>
        </div>
    </div>
);

const ChartsWindow: FC<{ data: ChartDataPoint[], onClose: () => void }> = ({ data, onClose }) => {
    const [isMinimized, setIsMinimized] = useState(false);
    const nodeRef = useRef(null);
    const dataMax = data.length > 0 ? data[data.length - 1].timestep : 0;
    const dataMin = data.length > 0 ? data[0].timestep : 0;

    return (
        <Draggable handle=".handle" nodeRef={nodeRef} defaultPosition={{x: 20, y: 180}}>
            <div ref={nodeRef} className="absolute z-50">
                <Resizable defaultSize={{ width: 550, height: 400 }} minWidth={400} minHeight={300} className="bg-gray-800/80 backdrop-blur-sm border border-gray-600 rounded-lg shadow-2xl flex flex-col">
                    <div className="handle cursor-move bg-gray-900/80 p-2 rounded-t-lg flex justify-between items-center">
                        <h3 className="font-bold text-white">Real-time System Metrics</h3>
                        <div className="flex items-center gap-2">
                            <button onClick={() => setIsMinimized(!isMinimized)} className="text-gray-400 hover:text-white"><Minus size={16} /></button>
                            <button onClick={onClose} className="text-gray-400 hover:text-white"><X size={16} /></button>
                        </div>
                    </div>
                    {!isMinimized && (
                        <div className="flex-grow p-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                                    <XAxis dataKey="timestep" stroke="#A0AEC0" name="Timestep" type="number" domain={[dataMin, dataMax]} allowDataOverflow={true} />
                                    <YAxis stroke="#A0AEC0" yAxisId="left" domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                                    <YAxis stroke="#A0AEC0" yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                                    <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4A5568', color: '#E2E8F0' }} formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']} />
                                    <Legend />
                                    <Line yAxisId="left" isAnimationActive={false} type="monotone" dataKey="fidelity" name="Fidelity" stroke="#3b82f6" dot={false} strokeWidth={2} />
                                    <Line yAxisId="right" isAnimationActive={false} type="monotone" dataKey="powerSaved" name="Power Saved" stroke="#22c55e" dot={false} strokeWidth={2} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </Resizable>
            </div>
        </Draggable>
    );
};

// --- Main Application Component ---
const ClientOnlyApp: FC = () => {
    const [status, setStatus] = useState<Status | null>(null);
    // Control states
    const [threshold, setThreshold] = useState(0.98);
    const [duration, setDuration] = useState(40);
    const [nWayComparison, setNWayComparison] = useState(2);
    const [shadowProb, setShadowProb] = useState(0.05);
    // Hybrid model states
    const [hybridFidelityThreshold, setHybridFidelityThreshold] = useState(0.97);
    const [hybridMaxTimesteps, setHybridMaxTimesteps] = useState(2880);
    const [collectionPeriod, setCollectionPeriod] = useState(200);

    const [isChartsVisible, setIsChartsVisible] = useState(false);
    const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    const API_BASE_URL = 'http://127.0.0.1:8000';

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/status`);
            if (!res.ok) throw new Error('Network response was not ok');
            const data: Status = await res.json();
            setStatus(data);

            if (data.is_running) {
                setChartData(prevData => {
                    const newPoint: ChartDataPoint = {
                        timestep: data.timestep,
                        fidelity: data.fidelity,
                        powerSaved: data.power_saved_percent / 100,
                    };
                    const newData = [...prevData, newPoint];
                    return newData.length > 300 ? newData.slice(newData.length - 300) : newData;
                });
            }
        } catch (error) {
            console.error("Failed to fetch status:", error);
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }
    }, []);

    const sendCommand = useCallback(async (command: string, body: object = {}) => {
        try {
            await fetch(`${API_BASE_URL}/${command}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            await fetchStatus();
        } catch (error) {
            console.error(`Failed to send command '${command}':`, error);
        }
    }, [fetchStatus]);

    const handleStartPause = () => {
        if (status?.is_running) {
            sendCommand('pause');
        } else {
            setChartData([]);
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

    useEffect(() => {
        if (status?.is_running && !intervalRef.current) {
            intervalRef.current = setInterval(fetchStatus, 300);
        } else if (!status?.is_running && intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [status?.is_running, fetchStatus]);

    useEffect(() => {
        fetchStatus(); // Initial fetch
    }, [fetchStatus]);

    const getPhaseInfo = (phase: Status['current_phase'], isRunning: boolean) => {
        if (!isRunning) {
            return { text: 'IDLE', color: 'text-white', size: 'text-xl lg:text-2xl' };
        }
        switch(phase) {
            case 'collecting': 
                return { text: 'COLLECTING', color: 'text-yellow-400', size: 'text-xl lg:text-2xl' };
            case 'shadow_op': 
                return { text: 'SHADOW POWERSAVING', color: 'text-cyan-400', size: 'text-base sm:text-lg lg:text-[12px]' }; // Gradual font size
            default: 
                return { text: phase.toUpperCase().replace('_', ' '), color: 'text-white', size: 'text-xl lg:text-2xl' };
        }
    }

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

// --- Dynamic import to ensure client-side only rendering ---
const Home = dynamic(() => Promise.resolve(ClientOnlyApp), {
    ssr: false,
    loading: () => <div className="flex items-center justify-center h-screen bg-gray-900 text-white">Loading Application...</div>
});

export default Home;

