import React, { useState } from 'react';
import Chat from './Chat';
import './styles.css';

function App() {
  const [userType, setUserType] = useState(null);

  const handleUserTypeSelect = (type) => {
    setUserType(type);
  };

  return (
    <div className="app-container">
      {!userType ? (
        <div className="user-type-selector">
          <h1>Welcome to AltiCred!</h1>
          <p>Please select your user type to begin:</p>
          <div className="user-type-buttons">
            <button onClick={() => handleUserTypeSelect('salaried')}>Salaried Employee</button>
            <button onClick={() => handleUserTypeSelect('student')}>Student</button>
            <button onClick={() => handleUserTypeSelect('farmer')}>Farmer</button>
          </div>
        </div>
      ) : (
        <Chat userType={userType} onBack={() => setUserType(null)} />
      )}
    </div>
  );
}

export default App;