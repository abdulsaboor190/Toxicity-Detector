import { useState } from 'react';

const LABELS = [
    { key: 'toxic', label: 'toxic' },
    { key: 'severe_toxic', label: 'severe_toxic' },
    { key: 'obscene', label: 'obscene' },
    { key: 'threat', label: 'threat' },
    { key: 'insult', label: 'insult' },
    { key: 'identity_hate', label: 'identity_hate' },
];

function barClass(score) {
    if (score >= 0.5) return 'breakdown__bar-fill breakdown__bar-fill--high';
    if (score >= 0.2) return 'breakdown__bar-fill breakdown__bar-fill--mid';
    return 'breakdown__bar-fill breakdown__bar-fill--low';
}

function scoreColor(score) {
    if (score >= 0.5) return '#ef4444';
    if (score >= 0.2) return '#eab308';
    return '#22c55e';
}

export default function ScoreBreakdown({ scores, flagged = [] }) {
    if (!scores) return null;

    return (
        <div className="breakdown">
            <div className="breakdown__title">Score Breakdown</div>
            {LABELS.map(({ key, label }) => {
                const score = scores[key] || 0;
                const isFlagged = flagged.includes(key);
                return (
                    <div className="breakdown__row" key={key}>
                        <span className="breakdown__label">{label}</span>
                        <div className="breakdown__bar-track">
                            <div
                                className={barClass(score)}
                                style={{ width: `${Math.max(score * 100, 1)}%` }}
                            />
                        </div>
                        <span
                            className="breakdown__score"
                            style={{ color: scoreColor(score) }}
                        >
                            {score.toFixed(2)}
                        </span>
                        <span className="breakdown__flag">
                            {isFlagged ? '⚑' : ''}
                        </span>
                    </div>
                );
            })}
        </div>
    );
}
