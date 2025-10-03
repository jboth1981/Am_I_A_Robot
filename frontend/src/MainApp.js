import React, { useState } from 'react';
import { useAuth } from './contexts/AuthContext';
import { authService } from './services/authService';
import About from './components/About';
import Sidebar from './components/Sidebar';
import './App.css';
import './components/AuthForm.css';

function MainApp() {
  const [currentPage, setCurrentPage] = useState('app'); // 'app' or 'about'
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [isTyping, setIsTyping] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [predictionMethod, setPredictionMethod] = useState('frequency');
  const { user, token, logout, isGuest } = useAuth();

  // Handle page changes from sidebar
  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  // Handle authentication actions from sidebar
  const handleAuthAction = (action) => {
    // This will trigger the auth wrapper to show login/register forms
    // For now, we'll just logout to show the auth forms
    logout();
  };

  const handleInputChange = async (e) => {
    const newValue = e.target.value;

    // Block input while prediction is being calculated
    if (isTyping) {
      return; // Prevent input while waiting for prediction
    }

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
          console.log(`MainApp DEBUG: Getting prediction for next character, history='${newValue}'`);
          const startTime = performance.now();
          
          const data = await authService.predict({ 
            history: newValue,
            method: predictionMethod
          }, token);
          
          const endTime = performance.now();
          console.log(`MainApp DEBUG: Prediction took ${endTime - startTime}ms`);
          console.log(`MainApp DEBUG: Received prediction for next='${data.prediction}'`);
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
          console.log(`MainApp DEBUG: Getting initial prediction, history='${newValue}'`);
          const startTime = performance.now();
          
          const data = await authService.predict({ 
            history: newValue,
            method: predictionMethod
          }, token);
          
          const endTime = performance.now();
          console.log(`MainApp DEBUG: Initial prediction took ${endTime - startTime}ms`);
          console.log(`MainApp DEBUG: Received initial prediction='${data.prediction}'`);
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

  // Save submission when completed (for authenticated users only)
  React.useEffect(() => {
    if (isCompleted && score.total > 0 && user && token) {
      const saveSubmissionData = async () => {
        try {
          const submissionData = {
            binary_sequence: inputHistory,
            prediction_method: predictionMethod,
            total_predictions: score.total,
            correct_predictions: score.correct,
            accuracy_percentage: parseFloat(accuracy),
            is_human_result: isHuman
          };
          
          await authService.saveSubmission(submissionData, token);
          console.log('Submission saved successfully');
        } catch (error) {
          console.error('Error saving submission:', error);
        }
      };
      
      saveSubmissionData();
    }
  }, [isCompleted, score.total, user, token, inputHistory, predictionMethod, score.correct, accuracy, isHuman]);

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
      <Sidebar 
        currentPage={currentPage}
        onPageChange={handlePageChange}
        onAuthAction={handleAuthAction}
      />
      
      <div className="main-content">
        <div className="container">
          {/* Render About page */}
          {currentPage === 'about' && <About />}
          
          {/* Render Main Game */}
          {currentPage === 'app' && (
            <main className="main">
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
                  placeholder={isTyping ? "Processing prediction..." : "Type 0s and 1s here..."}
                  className={`binary-input ${isTyping ? 'processing' : ''}`}
                  maxLength="100"
                  readOnly={isCompleted || isTyping}
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
                    Start Over
                  </button>
                </div>
              )}

              {isCompleted && score.total > 0 && (
                <div id="verdict-section" className="verdict-section">
                  <div className={`verdict-card ${isHuman ? 'human' : 'robot'}`}>
                    <h2>The Grand Reveal</h2>
                    <div className="verdict-result">
                      {isHuman ? (
                        <>
                          <span className="verdict-icon">Human</span>
                          <span className="verdict-text">You appear to be HUMAN</span>
                          <span className="verdict-explanation">
                            Your pattern shows predictable human behavior. You have free will, but it's not as random as you think!
                          </span>
                        </>
                      ) : (
                        <>
                          <span className="verdict-icon">Robot</span>
                          <span className="verdict-text">You appear to be a ROBOT</span>
                          <span className="verdict-explanation">
                            Your pattern is too predictable for a human. Are you sure you're not a machine?
                          </span>
                        </>
                      )}
                    </div>
                    <button onClick={handleTryAgain} className="try-again-btn">
                      Try Again
                    </button>
                  </div>
                </div>
              )}
            </main>
          )}
        </div>
      </div>
    </div>
  );
}

export default MainApp;