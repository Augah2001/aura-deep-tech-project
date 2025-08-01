'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Status, ChartDataPoint } from '../lib/types';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const useStatus = () => {
    const [status, setStatus] = useState<Status | null>(null);
    const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

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

    return { status, chartData, sendCommand, setChartData };
};
