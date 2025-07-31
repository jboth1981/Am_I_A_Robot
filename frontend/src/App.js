import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [isTyping, setIsTyping] = useState(false);

  const handleInputChange = async (e) => {
    const newValue = e.target.value;

    // Validate input: only allow 0 and 1
    if (/^[01]*$/.test(newValue)) {
      // Find the new character typed (if any)
      const newChar = newValue.length > inputHistory.length ? newValue.slice(-1) : null;

      setInputHistory(newValue);

      if (newChar) {
        setIsTyping(true);
        try {
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
        } catch (error) {
          console.error('Error fetching prediction:', error);
        } finally {
          setIsTyping(false);
        }
      }
    }
  };

  const accuracy = score.total > 0 ? ((score.correct / score.total) * 100).toFixed(1) : 0;
  const isHuman = accuracy < 60; // If accuracy is low, you're more "human"

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="title-main">Am I A Robot?</span>
            <span className="title-subtitle">The Turing Test for Randomness</span>
          </h1>
        </header>

        <main className="main">
          <div className="description">
            <p>
              Type a sequence of 0s and 1s. We'll analyze your pattern to determine if you're 
              generating truly random numbers or following predictable human patterns.
            </p>
          </div>

          <div className="input-section">
            <label htmlFor="binary-input" className="input-label">
              Enter your binary sequence:
            </label>
            <input
              id="binary-input"
              type="text"
              value={inputHistory}
              onChange={handleInputChange}
              placeholder="Type 0s and 1s here..."
              className="binary-input"
              maxLength="100"
            />
            <div className="input-hint">
              Only 0s and 1s are allowed • Max 100 characters
            </div>
          </div>

          <div className="results-section">
            <div className="result-card">
              <h3>Your Input</h3>
              <div className="binary-display">
                {inputHistory || <span className="placeholder">Start typing...</span>}
              </div>
            </div>

            <div className="result-card">
              <h3>AI Prediction</h3>
              <div className="prediction-display">
                {isTyping ? (
                  <span className="loading">Analyzing...</span>
                ) : prediction ? (
                  <span className="prediction">{prediction}</span>
                ) : (
                  <span className="placeholder">Waiting for input...</span>
                )}
              </div>
            </div>
          </div>

          <div className="stats-section">
            <div className="stat-card">
              <div className="stat-number">{score.total}</div>
              <div className="stat-label">Total Predictions</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{score.correct}</div>
              <div className="stat-label">Correct Predictions</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{accuracy}%</div>
              <div className="stat-label">Accuracy</div>
            </div>
          </div>

          {score.total > 0 && (
            <div className="verdict-section">
              <div className={`verdict-card ${isHuman ? 'human' : 'robot'}`}>
                <h2>Verdict</h2>
                <div className="verdict-result">
                  {isHuman ? (
                    <>
                      <span className="verdict-icon">🧠</span>
                      <span className="verdict-text">You appear to be HUMAN</span>
                      <span className="verdict-explanation">
                        Your pattern shows predictable human behavior
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="verdict-icon">🤖</span>
                      <span className="verdict-text">You appear to be a ROBOT</span>
                      <span className="verdict-explanation">
                        Your pattern is too predictable for a human
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>
            This is a demonstration of pattern recognition in binary sequences. 
            The AI predicts the next digit based on frequency analysis.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;

