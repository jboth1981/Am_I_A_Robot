import React, { useState } from 'react';

function App() {
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');
  const [score, setScore] = useState({ correct: 0, total: 0 });

  const handleInputChange = async (e) => {
    const newValue = e.target.value;

    // Validate input: only allow 0 and 1
    if (/^[01]*$/.test(newValue)) {
      // Find the new character typed (if any)
      const newChar = newValue.length > inputHistory.length ? newValue.slice(-1) : null;

      setInputHistory(newValue);

      if (newChar) {
        // Send history to backend for prediction
        const response = await fetch('https://api.amiarobot.ca/predict/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ history: newValue })
        });
        const data = await response.json();

        const wasCorrect = data.prediction === newChar;
        setScore({
          correct: score.correct + (wasCorrect ? 1 : 0),
          total: score.total + 1,
        });
        setPrediction(data.prediction);
      }
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Check if you are a Robot! ***Spoiler: you are </h1>
      <p>Type a sequence of 0s and 1s and we will see if you are a free agent generating numbers as you please, or a deterministic system:</p>
      <input
        type="text"
        value={inputHistory}
        onChange={handleInputChange}
        placeholder="Enter 0s and 1s"
        style={{ fontSize: 18, width: '100%', maxWidth: 300 }}
      />
      <p>Input so far: {inputHistory}</p>
      <p>Prediction for next: <strong>{prediction}</strong></p>
      <p>Score: {score.correct}/{score.total} correct</p>
    </div>
  );
}

export default App;

