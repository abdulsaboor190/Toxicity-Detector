import { useState, useEffect, lazy, Suspense } from 'react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import { useAnalyze } from './hooks/useAnalyze';

const SessionAnalytics = lazy(() => import('./components/analytics/SessionAnalytics'));
const ModelPerformance = lazy(() => import('./components/analytics/ModelPerformance'));
const BiasAudit = lazy(() => import('./components/analytics/BiasAudit'));

let nextId = 1;

const TABS = [
    { id: 'detector', label: '🛡️ Detector' },
    { id: 'session', label: '📊 Session Stats' },
    { id: 'model', label: '🤖 Model Perf' },
    { id: 'bias', label: '⚖️ Bias Audit' },
];

function TabSuspense({ children }) {
    return (
        <Suspense fallback={
            <div className="analytics-loading">
                <span className="spinner" />
                <span>Loading…</span>
            </div>
        }>
            {children}
        </Suspense>
    );
}

export default function App() {
    const [activeTab, setActiveTab] = useState('detector');
    const [messages, setMessages] = useState([]);
    const [backendOnline, setBackendOnline] = useState(false);
    const { analyze, checkHealth, isLoading, error, clearError } = useAnalyze();

    useEffect(() => {
        const check = async () => {
            const health = await checkHealth();
            setBackendOnline(health?.model_loaded ?? false);
        };
        check();
        const interval = setInterval(check, 10000);
        return () => clearInterval(interval);
    }, []);

    const handleSend = async (text) => {
        clearError();
        const result = await analyze(text);
        if (result) {
            setMessages((prev) => [
                ...prev,
                { id: nextId++, text: result.message, result, timestamp: Date.now() },
            ]);
        }
    };

    return (
        <div className="app">
            {/* ── Header ── */}
            <header className="header">
                <div className="header__title">
                    <img src="/tox_log.png" alt="Tox Logo" className="header__icon" style={{ borderRadius: '8px', objectFit: 'contain' }} />
                    Toxicity Detector
                </div>

                {/* ── Tab Bar ── */}
                <nav className="tab-bar" role="tablist">
                    {TABS.map(tab => (
                        <button
                            key={tab.id}
                            role="tab"
                            aria-selected={activeTab === tab.id}
                            className={`tab-btn ${activeTab === tab.id ? 'tab-btn--active' : ''}`}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            {tab.label}
                        </button>
                    ))}
                </nav>

                <div className="header__status">
                    <span className={`header__dot ${!backendOnline ? 'header__dot--offline' : ''}`} />
                    {backendOnline ? 'Model Ready' : 'Offline'}
                </div>
            </header>

            {/* ── Error Banner ── */}
            {error && activeTab === 'detector' && (
                <div className="error-banner">
                    <span>⚠</span>
                    <span>{error}</span>
                </div>
            )}

            {/* ── Tab Content ── */}
            <div className="tab-content">
                {activeTab === 'detector' && (
                    <>
                        <ChatWindow messages={messages} isLoading={isLoading} />
                        <InputBar onSend={handleSend} isLoading={isLoading} />
                    </>
                )}

                {activeTab === 'session' && (
                    <TabSuspense><SessionAnalytics /></TabSuspense>
                )}

                {activeTab === 'model' && (
                    <TabSuspense><ModelPerformance /></TabSuspense>
                )}

                {activeTab === 'bias' && (
                    <TabSuspense><BiasAudit /></TabSuspense>
                )}
            </div>
        </div>
    );
}
