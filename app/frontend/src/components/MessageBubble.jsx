import { useState } from 'react';
import ToxicityBadge from './ToxicityBadge';
import ScoreBreakdown from './ScoreBreakdown';

export default function MessageBubble({ data }) {
    const [showBreakdown, setShowBreakdown] = useState(false);
    const { text, result, timestamp } = data;

    if (!result) return null;

    const { severity, overall_score, scores, flagged_categories, processing_time_ms } = result;
    const time = new Date(timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });

    return (
        <div
            className={`message message--${severity}`}
            onMouseEnter={() => setShowBreakdown(true)}
            onMouseLeave={() => setShowBreakdown(false)}
        >
            {severity === 'severe' && (
                <span style={{ marginRight: 6 }}>🚨</span>
            )}

            <div className="message__text">{text}</div>

            {flagged_categories.length > 0 && (
                <div className="message__flags">
                    {flagged_categories.map((cat) => (
                        <span className="flag-chip" key={cat}>
                            {cat.replace('_', ' ')}
                        </span>
                    ))}
                </div>
            )}

            <div className="message__footer">
                <ToxicityBadge
                    severity={severity}
                    score={overall_score}
                    onClick={() => setShowBreakdown(!showBreakdown)}
                />
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                    <span className="message__latency">{processing_time_ms.toFixed(0)}ms</span>
                    <span className="message__time">{time}</span>
                </div>
            </div>

            {showBreakdown && (
                <ScoreBreakdown scores={scores} flagged={flagged_categories} />
            )}
        </div>
    );
}
