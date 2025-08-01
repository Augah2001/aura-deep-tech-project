'use client';

import { FC, useRef, useState } from 'react';
import Draggable from 'react-draggable';
import { Resizable } from 're-resizable';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { X, Minus } from 'lucide-react';
import { ChartDataPoint } from '../lib/types';

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

export default ChartsWindow;
