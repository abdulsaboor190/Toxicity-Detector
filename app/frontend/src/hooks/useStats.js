import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function useStats(autoRefresh = false, intervalMs = 3000) {
    const [stats, setStats] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchStats = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await axios.get(`${API_URL}/stats`, { timeout: 5000 });
            setStats(res.data);
            setError(null);
        } catch (err) {
            setError('Could not fetch analytics data.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const resetStats = useCallback(async () => {
        try {
            await axios.post(`${API_URL}/stats/reset`, {}, { timeout: 5000 });
            await fetchStats();
        } catch {
            /* ignore */
        }
    }, [fetchStats]);

    useEffect(() => {
        fetchStats();
        if (autoRefresh) {
            const id = setInterval(fetchStats, intervalMs);
            return () => clearInterval(id);
        }
    }, [fetchStats, autoRefresh, intervalMs]);

    return { stats, isLoading, error, fetchStats, resetStats };
}
