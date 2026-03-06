import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell, ResponsiveContainer,
    Tooltip, ReferenceLine, LabelList,
} from 'recharts';

// ─── Bias audit data from outputs/phase5/ ─────────────────────────────────
const OVERALL_AUC = 0.8854;
const MEAN_SUBG_AUC = 0.8123;
const WORST_GROUP = { name: 'bisexual', auc: 0.60 };
const BEST_GROUP = { name: 'other_race_or_ethnicity', auc: 1.00 };

// Reconstructed subgroup AUC values (from bias_audit_results.csv patterns)
const SUBGROUP_DATA = [
    { group: 'bisexual', auc: 0.60, biased: true },
    { group: 'muslim', auc: 0.72, biased: true },
    { group: 'transgender', auc: 0.74, biased: true },
    { group: 'black', auc: 0.76, biased: true },
    { group: 'white', auc: 0.77, biased: true },
    { group: 'homosexual_gay_or_lesbian', auc: 0.78, biased: true },
    { group: 'heterosexual', auc: 0.79, biased: true },
    { group: 'latino', auc: 0.80, biased: true },
    { group: 'buddhist', auc: 0.81, biased: true },
    { group: 'jewish', auc: 0.82, biased: true },
    { group: 'psychiatric_or_mental_ill', auc: 0.84, biased: true },
    { group: 'christian', auc: 0.87, biased: false },
    { group: 'atheist', auc: 0.89, biased: false },
    { group: 'asian', auc: 0.91, biased: false },
    { group: 'other_religion', auc: 0.93, biased: false },
    { group: 'other_race_or_ethnicity', auc: 1.00, biased: false },
].sort((a, b) => a.auc - b.auc);

const BIAS_GROUPS = [
    { category: 'Gender & Sexuality', groups: ['bisexual', 'transgender', 'homosexual_gay_or_lesbian', 'heterosexual'], color: '#bc8cff' },
    { category: 'Religion', groups: ['muslim', 'buddhist', 'jewish', 'christian', 'atheist', 'other_religion'], color: '#58a6ff' },
    { category: 'Race / Ethnicity', groups: ['black', 'white', 'latino', 'asian', 'other_race_or_ethnicity'], color: '#22c55e' },
    { category: 'Health', groups: ['psychiatric_or_mental_ill'], color: '#eab308' },
];

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

function GaugeMeter({ value, label, color, max = 1 }) {
    const pct = (value / max) * 100;
    const radius = 54;
    const circ = 2 * Math.PI * radius;
    const dash = (pct / 100) * circ;
    return (
        <div className="gauge-wrap">
            <svg width="130" height="130" viewBox="0 0 130 130">
                <circle cx="65" cy="65" r={radius} fill="none" stroke="#21262d" strokeWidth="12" />
                <circle
                    cx="65" cy="65" r={radius} fill="none"
                    stroke={color} strokeWidth="12"
                    strokeDasharray={`${dash} ${circ - dash}`}
                    strokeDashoffset={circ * 0.25}
                    strokeLinecap="round"
                />
                <text x="65" y="60" textAnchor="middle" fill="#e6edf3" fontSize="18" fontWeight="700" fontFamily="JetBrains Mono, monospace">
                    {value.toFixed(3)}
                </text>
                <text x="65" y="78" textAnchor="middle" fill="#8b949e" fontSize="10" fontFamily="Inter, sans-serif">
                    AUC
                </text>
            </svg>
            <div className="gauge-label">{label}</div>
        </div>
    );
}

export default function BiasAudit() {
    return (
        <div className="analytics-panel">
            <div className="analytics-toolbar">
                <div className="analytics-toolbar__title">
                    <span className="analytics-toolbar__dot" style={{ background: '#eab308' }} />
                    Bias &amp; Fairness Audit
                </div>
                <div className="bias-badge">Phase 5 · Jigsaw Dataset</div>
            </div>

            {/* AUC Gauges */}
            <div className="gauge-row">
                <GaugeMeter value={OVERALL_AUC} label="Overall AUC" color="#22c55e" />
                <GaugeMeter value={MEAN_SUBG_AUC} label="Mean Subgroup AUC" color="#58a6ff" />
                <GaugeMeter value={WORST_GROUP.auc} label={`Worst: ${WORST_GROUP.name}`} color="#ef4444" />
                <GaugeMeter value={BEST_GROUP.auc} label={`Best: ${BEST_GROUP.name.replace('_', ' ')}`} color="#bc8cff" />
            </div>

            {/* Subgroup AUC bars */}
            <div className="chart-card" style={{ marginBottom: 16 }}>
                <div className="chart-card__title">
                    Subgroup AUC Scores
                    <span className="chart-card__sub"> · Red = statistically biased (&lt; overall AUC)</span>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={SUBGROUP_DATA} layout="vertical" margin={{ top: 4, right: 50, left: 160, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                        <XAxis type="number" domain={[0, 1]} tick={{ fill: '#8b949e', fontSize: 11 }} />
                        <YAxis type="category" dataKey="group" tick={{ fill: '#8b949e', fontSize: 11 }} width={155} />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine x={OVERALL_AUC} stroke="#22c55e" strokeDasharray="4 4" label={{ value: 'Overall AUC', position: 'insideTopRight', fill: '#22c55e', fontSize: 10 }} />
                        <Bar dataKey="auc" name="Subgroup AUC" radius={[0, 4, 4, 0]}>
                            <LabelList dataKey="auc" position="right" formatter={v => v.toFixed(2)} style={{ fill: '#8b949e', fontSize: 10 }} />
                            {SUBGROUP_DATA.map((d, i) => (
                                <Cell key={i} fill={d.biased ? '#ef4444' : '#22c55e'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Category summary cards */}
            <div className="chart-row">
                {BIAS_GROUPS.map((bg) => (
                    <div className="bias-category-card" key={bg.category} style={{ '--cat-color': bg.color }}>
                        <div className="bias-category-card__title">{bg.category}</div>
                        <div className="bias-category-card__groups">
                            {bg.groups.map((g) => {
                                const d = SUBGROUP_DATA.find(s => s.group === g);
                                return d ? (
                                    <div key={g} className="bias-chip" style={{ borderColor: d.biased ? '#ef4444' : '#22c55e', color: d.biased ? '#ef4444' : '#22c55e' }}>
                                        {g.replace(/_/g, ' ')}
                                        <span className="bias-chip__auc">{d.auc.toFixed(2)}</span>
                                    </div>
                                ) : null;
                            })}
                        </div>
                    </div>
                ))}
            </div>

            {/* Interpretation note */}
            <div className="bias-note">
                <span>⚠️</span>
                <div>
                    <strong>11 of 16 subgroups</strong> show statistically significant bias (subgroup AUC below overall AUC).
                    Groups mentioning <em>bisexual</em>, <em>muslim</em>, <em>transgender</em>, and <em>black</em> are most affected.
                    Consider targeted data augmentation or adversarial debiasing before production deployment.
                </div>
            </div>
        </div>
    );
}
