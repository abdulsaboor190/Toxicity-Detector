import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ResponsiveContainer, Tooltip,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell, LabelList,
} from 'recharts';

// ─── Static model-performance data from training phase ─────────────────────
// Sourced from outputs/phase4/phase4_summary.json
const LR_VAL = [
    { label: 'Toxic', f1: 0.757 },
    { label: 'Severe', f1: 0.430 },
    { label: 'Obscene', f1: 0.788 },
    { label: 'Threat', f1: 0.422 },
    { label: 'Insult', f1: 0.693 },
    { label: 'Identity Hate', f1: 0.395 },
];

const LR_TEST = [
    { label: 'Toxic', f1: 0.642 },
    { label: 'Severe', f1: 0.279 },
    { label: 'Obscene', f1: 0.649 },
    { label: 'Threat', f1: 0.109 },
    { label: 'Insult', f1: 0.608 },
    { label: 'Identity Hate', f1: 0.164 },
];

// BERT epoch 1 F1 from checkpoint filename
const BERT_F1 = 0.4322;

const MODEL_COMPARE = [
    { name: 'LR Val', macro: 0.581, color: '#58a6ff' },
    { name: 'LR Test', macro: 0.408, color: '#bc8cff' },
    { name: 'BERT E1', macro: BERT_F1, color: '#ef4444' },
];

const CLASS_WEIGHTS = [
    { label: 'Toxic', weight: 5.22, color: '#58a6ff' },
    { label: 'Sv.Toxic', weight: 50.02, color: '#bc8cff' },
    { label: 'Obscene', weight: 9.44, color: '#22c55e' },
    { label: 'Threat', weight: 166.92, color: '#ef4444' },
    { label: 'Insult', weight: 10.13, color: '#eab308' },
    { label: 'Id.Hate', weight: 56.79, color: '#f97316' },
];

const THRESHOLDS = [
    { label: 'Toxic', threshold: 0.90 },
    { label: 'Sv.Toxic', threshold: 0.90 },
    { label: 'Obscene', threshold: 0.90 },
    { label: 'Threat', threshold: 0.84 },
    { label: 'Insult', threshold: 0.89 },
    { label: 'Id.Hate', threshold: 0.82 },
];

const RADAR_DATA = LR_VAL.map(d => ({ label: d.label, Val: d.f1, Test: LR_TEST.find(t => t.label === d.label)?.f1 ?? 0 }));

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

function StatTag({ label, value, color }) {
    return (
        <div className="stat-tag" style={{ '--tag-color': color }}>
            <div className="stat-tag__val">{value}</div>
            <div className="stat-tag__lbl">{label}</div>
        </div>
    );
}

export default function ModelPerformance() {
    return (
        <div className="analytics-panel">
            <div className="analytics-toolbar">
                <div className="analytics-toolbar__title">
                    <span className="analytics-toolbar__dot" style={{ background: '#bc8cff' }} />
                    Model Performance &amp; Training Insights
                </div>
                <div className="model-badge">DistilBERT · Fine-tuned</div>
            </div>

            {/* Model quick stats */}
            <div className="kpi-row">
                <StatTag label="Architecture" value="DistilBERT" color="#bc8cff" />
                <StatTag label="LR Val Macro-F1" value="0.581" color="#58a6ff" />
                <StatTag label="LR Test Macro-F1" value="0.408" color="#8b949e" />
                <StatTag label="BERT E1 F1" value="0.432" color="#ef4444" />
                <StatTag label="Classes" value="6 Labels" color="#22c55e" />
                <StatTag label="Max Seq Length" value="128 Tokens" color="#eab308" />
            </div>

            <div className="chart-row">
                {/* Macro-F1 Comparison */}
                <div className="chart-card">
                    <div className="chart-card__title">Macro F1 — Model Comparison</div>
                    <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={MODEL_COMPARE} margin={{ top: 16, right: 20, left: -10, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 12 }} />
                            <YAxis domain={[0, 1]} tick={{ fill: '#8b949e', fontSize: 12 }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="macro" name="Macro F1" radius={[6, 6, 0, 0]}>
                                <LabelList dataKey="macro" position="top" formatter={v => v.toFixed(3)} style={{ fill: '#e6edf3', fontSize: 11 }} />
                                {MODEL_COMPARE.map((d, i) => (
                                    <Cell key={i} fill={d.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Val vs Test Radar */}
                <div className="chart-card">
                    <div className="chart-card__title">Per-Label F1 — Val vs Test (LR)</div>
                    <ResponsiveContainer width="100%" height={220}>
                        <RadarChart cx="50%" cy="50%" outerRadius={78} data={RADAR_DATA}>
                            <PolarGrid stroke="rgba(255,255,255,0.1)" />
                            <PolarAngleAxis dataKey="label" tick={{ fill: '#8b949e', fontSize: 10 }} />
                            <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} axisLine={false} />
                            <Radar name="Val F1" dataKey="Val" stroke="#58a6ff" fill="#58a6ff" fillOpacity={0.2} dot />
                            <Radar name="Test F1" dataKey="Test" stroke="#bc8cff" fill="#bc8cff" fillOpacity={0.15} dot />
                            <Tooltip content={<CustomTooltip />} />
                        </RadarChart>
                    </ResponsiveContainer>
                    <div className="radar-legend">
                        <span style={{ color: '#58a6ff' }}>● Val F1</span>
                        <span style={{ color: '#bc8cff' }}>● Test F1</span>
                    </div>
                </div>
            </div>

            <div className="chart-row">
                {/* Tuned Thresholds */}
                <div className="chart-card">
                    <div className="chart-card__title">Tuned Classification Thresholds</div>
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={THRESHOLDS} margin={{ top: 10, right: 20, left: -10, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="label" tick={{ fill: '#8b949e', fontSize: 11 }} />
                            <YAxis domain={[0, 1]} tick={{ fill: '#8b949e', fontSize: 11 }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="threshold" name="Threshold" radius={[4, 4, 0, 0]} fill="#58a6ff">
                                <LabelList dataKey="threshold" position="top" style={{ fill: '#e6edf3', fontSize: 11 }} />
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <p className="chart-note">High thresholds (&gt;0.84) reduce false positives on the imbalanced dataset.</p>
                </div>

                {/* Class Weights */}
                <div className="chart-card">
                    <div className="chart-card__title">Class Weights (Imbalance Correction)</div>
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={CLASS_WEIGHTS} margin={{ top: 10, right: 20, left: -10, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="label" tick={{ fill: '#8b949e', fontSize: 11 }} />
                            <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="weight" name="Class Weight" radius={[4, 4, 0, 0]}>
                                {CLASS_WEIGHTS.map((d, i) => (
                                    <Cell key={i} fill={d.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <p className="chart-note">Threat (×167) and identity_hate (×57) are heavily up-weighted due to extreme rarity.</p>
                </div>
            </div>
        </div>
    );
}
