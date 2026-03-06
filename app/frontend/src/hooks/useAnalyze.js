import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function useAnalyze() {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const analyze = async (message) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await axios.post(
                `${API_URL}/analyze`,
                { message },
                { timeout: 30000 }
            );
            return res.data;
        } catch (err) {
            const msg =
                err.response?.data?.detail ||
                err.message ||
                'Connection failed. Is the backend running?';
            setError(msg);
            return null;
        } finally {
            setIsLoading(false);
        }
    };

    const checkHealth = async () => {
        try {
            const res = await axios.get(`${API_URL}/health`, { timeout: 3000 });
            return res.data;
        } catch {
            return null;
        }
    };

    return { analyze, checkHealth, isLoading, error, clearError: () => setError(null) };
}
