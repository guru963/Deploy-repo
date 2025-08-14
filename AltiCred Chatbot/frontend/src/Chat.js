import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';

function Chat({ userType, onBack }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const initialMessage = `Hello, ${userType}! Please enter the relevant details you have to calculate your AltiCred Score.`;
    setMessages([{ text: initialMessage, sender: 'bot' }]);
  }, [userType]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    // Add user message to the chat
    const userMessage = { text: input, sender: 'user' };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // Clear input field
    setInput('');

    // Call the Flask API
    const response = await fetch('http://localhost:5000/get_score', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_type: userType, message: input }),
    });

    const data = await response.json();

    if (response.ok) {
      const score = data.score.toFixed(4);
      const advice = data.advice;
      setMessages(prevMessages => [...prevMessages, { text: `Your AltiCred Score is: ${score}`, sender: 'bot' }]);
      setMessages(prevMessages => [...prevMessages, { text: `**Advice:** ${advice}`, sender: 'bot' }]);
    } else {
      setMessages(prevMessages => [...prevMessages, { text: `Error: ${data.error}`, sender: 'bot' }]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="header">
        <button onClick={onBack}>&#x2190;</button>
        <h3>AltiCred Chatbot - {userType}</h3>
      </div>
      <div className="message-list">
        {messages.map((msg, index) => (
          <Message key={index} text={msg.text} sender={msg.sender} />
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter your details here..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
}

export default Chat;