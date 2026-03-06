import { useState, useRef, useEffect } from 'react';

export default function InputBar({ onSend, isLoading }) {
    const [text, setText] = useState('');
    const inputRef = useRef(null);

    useEffect(() => {
        inputRef.current?.focus();
    }, [isLoading]);

    const handleSubmit = (e) => {
        e.preventDefault();
        const trimmed = text.trim();
        if (!trimmed || isLoading) return;
        onSend(trimmed);
        setText('');
    };

    return (
        <form className="input-bar" onSubmit={handleSubmit}>
            <input
                ref={inputRef}
                type="text"
                className="input-bar__field"
                placeholder="Type a message to analyze for toxicity..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={isLoading}
                autoComplete="off"
                id="message-input"
            />
            <button
                type="submit"
                className="input-bar__btn"
                disabled={isLoading || !text.trim()}
                id="send-button"
                aria-label="Analyze message"
            >
                {isLoading ? <span className="spinner" /> : '→'}
            </button>
        </form>
    );
}
