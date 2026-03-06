export default function ToxicityBadge({ severity, score, onClick }) {
    const icons = {
        clean: '✓',
        mild: '◆',
        toxic: '⚠',
        severe: '⚡',
    };

    return (
        <span
            className={`badge badge--${severity}`}
            onClick={onClick}
            title="Click for score breakdown"
        >
            {icons[severity] || '?'} {severity.toUpperCase()} {score.toFixed(2)}
        </span>
    );
}
