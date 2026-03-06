import { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';

export default function ChatWindow({ messages, isLoading }) {
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    if (messages.length === 0 && !isLoading) {
        return (
            <div className="chat-window">
                <div className="chat-window__empty">
                    <div className="chat-window__empty-icon">🛡️</div>
                    <div className="chat-window__empty-text">
                        Chat Toxicity Detector<br />
                        <span style={{ opacity: 0.6 }}>
                            Type a message below to analyze it in real-time.<br />
                            Each message is scored across 6 toxicity categories.
                        </span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-window" id="chat-window">
            {messages.map((msg) => (
                <MessageBubble key={msg.id} data={msg} />
            ))}

            {isLoading && (
                <div className="message message--loading">
                    <span className="spinner" />
                    <span className="message__text">Analyzing message...</span>
                </div>
            )}

            <div ref={endRef} />
        </div>
    );
}
