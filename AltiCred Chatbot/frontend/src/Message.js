import React from 'react';

function Message({ text, sender }) {
  const isBot = sender === 'bot';
  const className = `message ${isBot ? 'bot' : 'user'}`;

  // You can use a library like 'markdown-to-jsx' to render markdown if needed
  const renderText = (text) => {
    if (text.startsWith('**Advice:**')) {
      return (
        <>
          <p>Your score is calculated.</p>
          <p className="advice-text">{text.replace('**Advice:** ', '')}</p>
        </>
      );
    }
    return text;
  };

  return (
    <div className={className}>
      <div className="message-bubble">{renderText(text)}</div>
    </div>
  );
}

export default Message;