import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [isTyping, setIsTyping] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [predictionMethod, setPredictionMethod] = useState('frequency');

  const handleInputChange = async (e) => {
    const newValue = e.target.value;

    // Only allow adding characters, not deleting
    if (newValue.length < inputHistory.length) {
      return; // Prevent deletion
    }

    // Validate input: only allow 0 and 1
    if (/^[01]*$/.test(newValue)) {
      // Find the new character typed (if any)
      const newChar = newValue.length > inputHistory.length ? newValue.slice(-1) : null;

      setInputHistory(newValue);

      // Auto-resize the textarea
      setTimeout(() => {
        const textarea = document.getElementById('binary-input');
        autoResizeTextarea(textarea);
      }, 0);

      // Check if we've reached 100 characters
      if (newValue.length >= 100) {
        setIsCompleted(true);
      }

      if (newChar) {
        // Compare previous prediction with the character that was just typed
        if (prediction) {
          const wasCorrect = prediction === newChar;
          console.log(`Frontend DEBUG: Comparing prediction='${prediction}' with newChar='${newChar}', wasCorrect=${wasCorrect}`);
          
          setScore({
            correct: score.correct + (wasCorrect ? 1 : 0),
            total: score.total + 1,
          });
        }

        // Get new prediction for the next character
        setIsTyping(true);
        try {
          console.log(`Frontend DEBUG: Getting prediction for next character, history='${newValue}'`);
          
          const response = await fetch('https://api.amiarobot.ca/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              history: newValue,
              method: predictionMethod
            })
          });
          const data = await response.json();
          
          console.log(`Frontend DEBUG: Received prediction for next='${data.prediction}'`);
          setPrediction(data.prediction);
        } catch (error) {
          console.error('Error fetching prediction:', error);
        } finally {
          setIsTyping(false);
        }
      } else if (newValue.length > 0) {
        // Initial prediction when first character is typed
        setIsTyping(true);
        try {
          console.log(`Frontend DEBUG: Getting initial prediction, history='${newValue}'`);
          
          const response = await fetch('https://api.amiarobot.ca/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              history: newValue,
              method: predictionMethod
            })
          });
          const data = await response.json();
          
          console.log(`Frontend DEBUG: Received initial prediction='${data.prediction}'`);
          setPrediction(data.prediction);
        } catch (error) {
          console.error('Error fetching prediction:', error);
        } finally {
          setIsTyping(false);
        }
      }
    }
  };

  // Keep cursor at the end of input
  const handleInputClick = (e) => {
    e.target.setSelectionRange(e.target.value.length, e.target.value.length);
  };

  const handleInputKeyDown = (e) => {
    // Prevent arrow keys, home, end, etc. from moving cursor
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(e.key)) {
      e.preventDefault();
    }
  };

  // Auto-resize textarea
  const autoResizeTextarea = (element) => {
    if (element) {
      element.style.height = 'auto';
      element.style.height = Math.min(element.scrollHeight, 300) + 'px';
    }
  };

  const accuracy = score.total > 0 ? ((score.correct / score.total) * 100).toFixed(1) : 0;
  const isHuman = accuracy < 60; // If accuracy is low, you're more "human"

  // Scroll to verdict when completed
  React.useEffect(() => {
    if (isCompleted && score.total > 0) {
      const verdictSection = document.getElementById('verdict-section');
      if (verdictSection) {
        setTimeout(() => {
          verdictSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
        }, 500); // Small delay for dramatic effect
      }
    }
  }, [isCompleted, score.total]);

  // Auto-resize textarea when input changes
  React.useEffect(() => {
    const textarea = document.getElementById('binary-input');
    if (textarea) {
      autoResizeTextarea(textarea);
    }
  }, [inputHistory]);

  // Reset function to start over
  const handleTryAgain = () => {
    setInputHistory('');
    setPrediction('');
    setScore({ correct: 0, total: 0 });
    setIsTyping(false);
    setIsCompleted(false);
    setPredictionMethod('frequency');
  };

  return (
    <div className="app">
      <div className="container">
                 <header className="header">
           <h1 className="title">
             <span className="title-main">Are You a Robot?</span>
           </h1>
         </header>

        <main className="main">
          <div className="description">
            <p>
              Most people believe they have free will. But what if a computer could predict your next choice before you know you chose it? In the box below, enter a series of 0s and 1s. Be as unpredictable as possible!
            </p>
          </div>

                    <div className="input-section">
            <div className="method-selector">
              <label htmlFor="prediction-method">Prediction Method:</label>
              <select
                id="prediction-method"
                value={predictionMethod}
                onChange={(e) => setPredictionMethod(e.target.value)}
                className="method-select"
                disabled={inputHistory.length > 0}
              >
                <option value="frequency">Frequency Analysis</option>
                <option value="pattern">Pattern Recognition</option>
              </select>
              <div className="method-description">
                {predictionMethod === 'frequency' && (
                  <span>Predicts the most frequently occurring digit</span>
                )}
                {predictionMethod === 'pattern' && (
                  <span>Uses pattern rules: 000→0, 111→1, otherwise repeats last digit</span>
                )}
              </div>
            </div>
            
            <textarea
              id="binary-input"
              value={inputHistory}
              onChange={handleInputChange}
              onClick={handleInputClick}
              onKeyDown={handleInputKeyDown}
              placeholder="Type 0s and 1s here..."
              className="binary-input"
              maxLength="100"
              readOnly={isCompleted}
              rows="3"
            />
            <div className="input-hint">
              Only 0s and 1s are allowed • Enter 100 characters for full evaluation
              {inputHistory.length > 0 && (
                <span className="character-count"> • {inputHistory.length}/100 characters</span>
              )}
              {inputHistory.length >= 90 && !isCompleted && (
                <span className="completion-hint"> • Almost there! The grand reveal awaits...</span>
              )}
              {isCompleted && (
                <span className="completion-message"> • Evaluation complete! Scroll down for the reveal!</span>
              )}
            </div>
          </div>

          <div className="results-section">
            <div className="result-card">
              <h3>We predict you'll choose:</h3>
              <div className="method-indicator">
                Using: {predictionMethod === 'frequency' ? 'Frequency Analysis' : 'Pattern Recognition'}
              </div>
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
               <div className="stat-label">Prediction Accuracy</div>
             </div>
           </div>
           
           {score.total > 0 && (
             <div className="restart-section">
               <button onClick={handleTryAgain} className="try-again-btn-small">
                 🔄 Start Over
               </button>
             </div>
           )}

                     {isCompleted && score.total > 0 && (
             <div id="verdict-section" className="verdict-section">
               <div className={`verdict-card ${isHuman ? 'human' : 'robot'}`}>
                 <h2>🎉 The Grand Reveal 🎉</h2>
                 <div className="verdict-result">
                   {isHuman ? (
                     <>
                       <span className="verdict-icon">🧠</span>
                       <span className="verdict-text">You appear to be HUMAN</span>
                       <span className="verdict-explanation">
                         Your pattern shows predictable human behavior. You have free will, but it's not as random as you think!
                       </span>
                     </>
                   ) : (
                     <>
                       <span className="verdict-icon">🤖</span>
                       <span className="verdict-text">You appear to be a ROBOT</span>
                       <span className="verdict-explanation">
                         Your pattern is too predictable for a human. Are you sure you're not a machine?
                       </span>
                     </>
                   )}
                 </div>
                 <button onClick={handleTryAgain} className="try-again-btn">
                   🔄 Try Again
                 </button>
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

