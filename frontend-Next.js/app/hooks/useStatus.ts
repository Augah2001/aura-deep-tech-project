'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Status, ChartDataPoint } from '../lib/types';

export const API_BASE_URL = 'http://127.0.0.1:8000';
const WS_STATUS_URL = 'ws://127.0.0.1:8000/ws/status';

export const useStatus = () => {
    const [status, setStatus] = useState<Status | null>(null);
    const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
    const [transport, setTransport] = useState<'websocket' | 'polling' | 'connecting'>('connecting');
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const websocketRef = useRef<WebSocket | null>(null);

    const applyStatus = useCallback((data: Status) => {
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
    }, []);

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/status`);
            if (!res.ok) throw new Error('Network response was not ok');
            const data: Status = await res.json();
            applyStatus(data);
        } catch (error) {
            console.error("Failed to fetch status:", error);
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }
    }, [applyStatus]);

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

    useEffect(() => {
        const socket = new WebSocket(WS_STATUS_URL);
        websocketRef.current = socket;
        socket.onopen = () => setTransport('websocket');
        socket.onmessage = event => {
            try {
                applyStatus(JSON.parse(event.data));
            } catch (error) {
                console.error('Failed to parse websocket status:', error);
            }
        };
        socket.onerror = () => {
            setTransport('polling');
            if (!intervalRef.current) intervalRef.current = setInterval(fetchStatus, 700);
        };
        socket.onclose = () => {
            if (websocketRef.current === socket) {
                setTransport('polling');
                if (!intervalRef.current) intervalRef.current = setInterval(fetchStatus, 700);
            }
        };
        return () => {
            websocketRef.current = null;
            socket.close();
        };
    }, [applyStatus, fetchStatus]);

    useEffect(() => {
        if (transport !== 'websocket' && status?.is_running && !intervalRef.current) {
            intervalRef.current = setInterval(fetchStatus, 300);
        } else if ((transport === 'websocket' || !status?.is_running) && intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        };
    }, [status?.is_running, fetchStatus, transport]);

    useEffect(() => {
        fetchStatus(); // Initial fetch
    }, [fetchStatus]);

    return { status, chartData, sendCommand, setChartData, transport, fetchStatus };
};
