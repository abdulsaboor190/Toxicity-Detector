import {
    PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    AreaChart, Area, Legend,
} from 'recharts';
import { useStats } from '../../hooks/useStats';

// ─── color palette ────────────────────────────────────────────────────────────
const SEV_COLORS = {
    clean: '#22c55e',
    mild: '#eab308',
    toxic: '#ef4444',
    severe: '#dc2626',
};

const LABEL_COLORS = [
    '#58a6ff', '#bc8cff', '#ef4444', '#f97316', '#eab308', '#22c55e',
];

const LABEL_DISPLAY = {
    toxic: 'Toxic',
    severe_toxic: 'Severe',
    obscene: 'Obscene',
    threat: 'Threat',
    insult: 'Insult',
    identity_hate: 'Identity Hate',
};

// ─── helpers ──────────────────────────────────────────────────────────────────
function fmtUptime(sec) {
    if (sec < 60) return `${sec}s`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}h ${m}m`;
}

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="chart-tooltip">
            {label && <div className="chart-tooltip__label">{label}</div>}
            {payload.map((p, i) => (
                <div key={i} className="chart-tooltip__row">
                    <span style={{ color: p.color || p.fill }}>■</span>
                    <span>{p.name ?? p.dataKey}: </span>
                    <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
                </div>
            ))}
        </div>
    );
};

// ─── KPI card ─────────────────────────────────────────────────────────────────
function KpiCard({ icon, label, value, sub, accent }) {
    return (
        <div className="kpi-card" style={{ '--kpi-accent': accent }}>
            <div className="kpi-card__icon">{icon}</div>
            <div className="kpi-card__body">
                <div className="kpi-card__value">{value}</div>
                <div className="kpi-card__label">{label}</div>
                {sub && <div className="kpi-card__sub">{sub}</div>}
            </div>
        </div>
    );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function SessionAnalytics() {
    const { stats, isLoading, error, fetchStats, resetStats } = useStats(true, 4000);

    if (isLoading && !stats) {
        return (
            <div className="analytics-loading">
                <span className="spinner" />
                <span>Loading analytics…</span>
            </div>
        );
    }

    if (error && !stats) {
        return (
            <div className="analytics-empty">
                <div style={{ fontSize: 40, marginBottom: 12 }}>📡</div>
                <div>Backend analytics unavailable.</div>
                <div style={{ opacity: 0.5, fontSize: 12, marginTop: 6 }}>
                    Make sure the backend is running on port 8000.
                </div>
            </div>
        );
    }

    const n = stats?.total_analyzed ?? 0;

    // Severity donut data
    const sevData = Object.entries(stats?.severity_counts ?? {})
        .map(([k, v]) => ({ name: k.charAt(0).toUpperCase() + k.slice(1), value: v, key: k }))
        .filter(d => d.value > 0);

    // Label bar data
    const labelData = Object.entries(stats?.label_counts ?? {})
        .map(([k, v], i) => ({ name: LABEL_DISPLAY[k] ?? k, count: v, avg: stats?.avg_label_scores?.[k] ?? 0, colorIdx: i }));

    // Timeline area data (last 30 points)
    const history = (stats?.history ?? []).slice(-30).map((h, i) => ({
        idx: i + 1,
        score: h.score,
        severity: h.severity,
    }));

    const toxicityPct = n > 0 ? ((stats.total_toxic / n) * 100).toFixed(1) : '0.0';

    return (
        <div className="analytics-panel">
            {/* Toolbar */}
            <div className="analytics-toolbar">
                <div className="analytics-toolbar__title">
                    <span className="analytics-toolbar__dot" />
                    Live Session Analytics
                </div>
                <div className="analytics-toolbar__actions">
                    <button className="ctrl-btn" onClick={fetchStats} title="Refresh">↻ Refresh</button>
                    <button className="ctrl-btn ctrl-btn--danger" onClick={resetStats} title="Reset session">⟳ Reset</button>
                </div>
            </div>

            {n === 0 ? (
                <div className="analytics-empty">
                    <div style={{ fontSize: 42, marginBottom: 12 }}>💬</div>
                    <div>No messages analyzed yet.</div>
                    <div style={{ opacity: 0.55, fontSize: 12, marginTop: 6 }}>
                        Switch to the <strong>Detector</strong> tab and send some messages!
                    </div>
                </div>
            ) : (
                <>
                    {/* KPI row */}
                    <div className="kpi-row">
                        <KpiCard icon="📊" label="Total Analyzed" value={n} accent="#58a6ff" />
                        <KpiCard icon="🚨" label="Toxic Messages" value={stats.total_toxic} accent="#ef4444" sub={`${toxicityPct}% rate`} />
                        <KpiCard icon="✅" label="Clean Messages" value={stats.severity_counts.clean ?? 0} accent="#22c55e" />
                        <KpiCard icon="⚡" label="Avg Latency" value={`${stats.avg_latency_ms}ms`} accent="#bc8cff" />
                        <KpiCard icon="⏱️" label="Session Uptime" value={fmtUptime(stats.session_uptime_s)} accent="#eab308" />
                    </div>

                    {/* Row 1: Donut + Bar */}
                    <div className="chart-row">
                        {/* Severity Donut */}
                        <div className="chart-card">
                            <div className="chart-card__title">Severity Distribution</div>
                            {sevData.length > 0 ? (
                                <ResponsiveContainer width="100%" height={220}>
                                    <PieChart>
                                        <Pie
                                            data={sevData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={90}
                                            paddingAngle={3}
                                            dataKey="value"
                                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                            labelLine={false}
                                        >
                                            {sevData.map((d) => (
                                                <Cell key={d.key} fill={SEV_COLORS[d.key] ?? '#8b949e'} stroke="transparent" />
                                            ))}
                                        </Pie>
                                        <Tooltip content={<CustomTooltip />} />
                                    </PieChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="chart-empty">No severity data yet</div>
                            )}
                            {/* Legend */}
                            <div className="donut-legend">
                                {Object.entries(SEV_COLORS).map(([k, c]) => (
                                    <div key={k} className="donut-legend__item">
                                        <span className="donut-legend__dot" style={{ background: c }} />
                                        <span className="donut-legend__text">{k}</span>
                                        <span className="donut-legend__count">{stats.severity_counts[k] ?? 0}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Label Frequency Bar */}
                        <div className="chart-card">
                            <div className="chart-card__title">Category Flag Frequency</div>
                            <ResponsiveContainer width="100%" height={220}>
                                <BarChart data={labelData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 11 }} />
                                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} allowDecimals={false} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Bar dataKey="count" name="Flagged Count" radius={[4, 4, 0, 0]}>
                                        {labelData.map((d, i) => (
                                            <Cell key={i} fill={LABEL_COLORS[d.colorIdx % LABEL_COLORS.length]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Row 2: Timeline + Avg Scores */}
                    <div className="chart-row">
                        {/* Score Timeline */}
                        <div className="chart-card chart-card--wide">
                            <div className="chart-card__title">Toxicity Score Timeline <span className="chart-card__sub">(last 30 messages)</span></div>
                            {history.length > 1 ? (
                                <ResponsiveContainer width="100%" height={200}>
                                    <AreaChart data={history} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                        <defs>
                                            <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis dataKey="idx" tick={{ fill: '#8b949e', fontSize: 11 }} label={{ value: 'Message #', position: 'insideBottom', offset: -2, fill: '#484f58', fontSize: 11 }} />
                                        <YAxis domain={[0, 1]} tick={{ fill: '#8b949e', fontSize: 11 }} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Area type="monotone" dataKey="score" name="Overall Score" stroke="#ef4444" strokeWidth={2} fill="url(#scoreGrad)" dot={{ r: 3, fill: '#ef4444', strokeWidth: 0 }} activeDot={{ r: 5 }} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="chart-empty">Send more messages to see the timeline</div>
                            )}
                        </div>

                        {/* Avg Label Scores */}
                        <div className="chart-card">
                            <div className="chart-card__title">Avg Score / Category</div>
                            <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={labelData} layout="vertical" margin={{ top: 4, right: 30, left: 60, bottom: 4 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                                    <XAxis type="number" domain={[0, 1]} tick={{ fill: '#8b949e', fontSize: 11 }} />
                                    <YAxis type="category" dataKey="name" tick={{ fill: '#8b949e', fontSize: 11 }} width={60} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Bar dataKey="avg" name="Avg Score" radius={[0, 4, 4, 0]}>
                                        {labelData.map((d, i) => (
                                            <Cell key={i} fill={LABEL_COLORS[d.colorIdx % LABEL_COLORS.length]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
