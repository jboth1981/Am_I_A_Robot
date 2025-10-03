import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { authService } from '../services/authService';

const GamePage = () => {
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');
  const [score, setScore] = useState({ correct: 0, total: 0, predictions: [] });
  const [isTyping, setIsTyping] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [predictionMethod, setPredictionMethod] = useState('frequency');
  const [showMethodDropdown, setShowMethodDropdown] = useState(false);
  const [hidePredictions, setHidePredictions] = useState(false);
  const inputRef = React.useRef(null);
  const initialPredictionMade = React.useRef(false);
  const { user, token } = useAuth();

  const handleInputChange = async (e) => {
    const inputValue = e.target.value;

    // Block input while prediction is being calculated
    if (isTyping) {
      e.target.value = '';
      return;
    }

    // Only allow single digit (0 or 1)
    if (inputValue.length > 1 || (inputValue.length === 1 && !/[01]/.test(inputValue))) {
      e.target.value = '';
      return;
    }

    // If a valid digit was entered
    if (inputValue.length === 1) {
      const newChar = inputValue;
      const newHistory = inputHistory + newChar;

      // Clear the input for next digit and maintain focus
      e.target.value = '';

      // Update history
      setInputHistory(newHistory);

      // Store the prediction that was made for this position and check if it was correct
      if (prediction) {
        const wasCorrect = prediction === newChar;
        setScore(prev => ({
          correct: prev.correct + (wasCorrect ? 1 : 0),
          total: prev.total + 1,
          predictions: [...prev.predictions, prediction]
        }));
      }

      // Check if we've reached 100 characters
      if (newHistory.length >= 100) {
        setIsCompleted(true);
        return; // Don't make a new prediction for the 101st digit
      }

      // Get new prediction for next character (only if not completed)
      setIsTyping(true);
      setPrediction('');
      
      try {
        const data = await authService.predict({ 
          history: newHistory,
          method: predictionMethod
        }, token);
        
        setPrediction(data.prediction);
      } catch (error) {
        console.error('Error fetching prediction:', error);
        setPrediction('?');
      } finally {
        setIsTyping(false);
        // Keep focus without setTimeout to avoid mobile keyboard reset
        if (inputRef.current && !isCompleted) {
          // Use requestAnimationFrame instead of setTimeout for smoother mobile experience
          requestAnimationFrame(() => {
            if (inputRef.current && !isCompleted) {
              inputRef.current.focus();
            }
          });
        }
      }
    }
  };

  const handleInputKeyDown = (e) => {
    // Allow backspace/delete to clear the field, but prevent other navigation
    if (e.key === 'Backspace' || e.key === 'Delete') {
      e.target.value = '';
    }
    // Only allow 0, 1, backspace, delete, and tab
    if (!/[01]/.test(e.key) && !['Backspace', 'Delete', 'Tab'].includes(e.key)) {
      e.preventDefault();
    }
  };

  const handleMethodSelect = (method) => {
    setPredictionMethod(method);
    setShowMethodDropdown(false);
  };

  const methodOptions = [
    {
      value: 'frequency',
      label: 'Frequency Analysis',
      description: 'Predicts the most frequently occurring digit in your sequence'
    },
    {
      value: 'pattern',
      label: 'Pattern Recognition',
      description: 'Uses pattern rules: 000→0, 111→1, otherwise repeats the last digit'
    }
  ];

  const unpredictabilityRate = score.total > 0 ? (((score.total - score.correct) / score.total) * 100).toFixed(1) : 0;
  const isHuman = unpredictabilityRate > 40; // If unpredictability is high, you're more "human"

  // Get initial prediction when component mounts or method changes (only when no digits typed)
  React.useEffect(() => {
    if (inputHistory.length === 0 && !isTyping && (!initialPredictionMade.current || !prediction)) {
      const getInitialPrediction = async () => {
        setIsTyping(true);
        try {
          const data = await authService.predict({ 
            history: '',
            method: predictionMethod
          }, token);
          
          setPrediction(data.prediction);
          initialPredictionMade.current = true;
        } catch (error) {
          console.error('Error fetching initial prediction:', error);
          setPrediction('?');
          initialPredictionMade.current = true;
        } finally {
          setIsTyping(false);
        }
      };
      
      getInitialPrediction();
    }
  }, [predictionMethod, inputHistory.length, isTyping, token]);

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
            accuracy_percentage: parseFloat((score.correct / score.total * 100).toFixed(1)), // Keep original accuracy for backend
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
  }, [isCompleted, score.total, user, token, inputHistory, predictionMethod, score.correct, unpredictabilityRate, isHuman]);

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

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (showMethodDropdown && !event.target.closest('.custom-dropdown')) {
        setShowMethodDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showMethodDropdown]);


  // Reset function to start over
  const handleTryAgain = () => {
    setInputHistory('');
    setPrediction('');
    setScore({ correct: 0, total: 0, predictions: [] });
    setIsTyping(false);
    setIsCompleted(false);
    setPredictionMethod('frequency');
    setShowMethodDropdown(false);
    setHidePredictions(false); // Reset hide predictions checkbox
    initialPredictionMade.current = false; // Reset the ref so initial prediction will be made again
    // Focus the input after reset using requestAnimationFrame for better mobile support
    requestAnimationFrame(() => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    });
    // Initial prediction will be triggered automatically by useEffect
  };

  return (
    <main className="game-main">
      <div className="game-layout">
        {/* Help Container - Invisible box above input boxes */}
        <div className="help-container">
          <div className="help-button-container">
            <div className="help-button" title="How to Play">
              <span className="help-icon">?</span>
              <div className="help-tooltip">
                <h4>How to Play</h4>
                <ol>
                  <li><strong>Choose a prediction method</strong> (Frequency Analysis or Pattern Recognition)</li>
                  <li><strong>Start typing digits</strong> (0 or 1) in the input box</li>
                  <li><strong>Watch the AI predict</strong> your next digit based on your pattern</li>
                  <li><strong>See the results</strong> in real-time - green for correct predictions, red for incorrect</li>
                  <li><strong>Complete 100 digits</strong> to get your final humanity score!</li>
                </ol>
                <p><em>The goal: Be unpredictable! If the AI can't guess your pattern, you're more human.</em></p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Top Row - Input, Prediction, and Method */}
        <div className="top-section">
          {/* Left - Input and Prediction */}
          <div className="input-prediction-section">
            <div className="input-prediction-row">
              <div className="input-area">
                <label>User Entry</label>
                <input
                  ref={inputRef}
                  type="tel"
                  inputMode="numeric"
                  pattern="[01]"
                  value=""
                  onChange={handleInputChange}
                  onKeyDown={handleInputKeyDown}
                  className={`single-digit-input ${isTyping ? 'processing' : ''}`}
                  maxLength="1"
                  disabled={isCompleted || isTyping}
                  autoFocus
                />
              </div>
              
              <div className="prediction-area">
                <label>Prediction</label>
                <div className="prediction-display">
                  {isTyping ? (
                    <span className="loading">...</span>
                  ) : hidePredictions && prediction ? (
                    <span className="hidden-prediction">?</span>
                  ) : prediction ? (
                    <span className="prediction">{prediction}</span>
                  ) : (
                    <span className="placeholder">-</span>
                  )}
                </div>
              </div>
            </div>

            {/* Compact Stats */}
            <div className="compact-stats">
              <span>Total: {score.total}</span>
              <span># Not Predicted: {score.total - score.correct}</span>
              <span>% Not Predicted: {score.total > 0 ? ((score.total - score.correct) / score.total * 100).toFixed(1) : 0}%</span>
            </div>
          </div>

          {/* Right - Prediction Method */}
          <div className="method-section">
            <div className="method-selector">
              <label>Method:</label>
              <div className="custom-dropdown">
                <button
                  className={`dropdown-button ${inputHistory.length > 0 ? 'disabled' : ''}`}
                  onClick={() => inputHistory.length === 0 && setShowMethodDropdown(!showMethodDropdown)}
                  disabled={inputHistory.length > 0}
                >
                  {methodOptions.find(opt => opt.value === predictionMethod)?.label}
                  <span className="dropdown-arrow">▼</span>
                </button>
                
                {showMethodDropdown && inputHistory.length === 0 && (
                  <div className="dropdown-menu">
                    {methodOptions.map((option) => (
                      <div
                        key={option.value}
                        className={`dropdown-item ${predictionMethod === option.value ? 'selected' : ''}`}
                        onClick={() => handleMethodSelect(option.value)}
                      >
                        <div className="dropdown-item-content">
                          <div className="dropdown-label" title={option.description}>
                            {option.label}
                          </div>
                          <div className="dropdown-tooltip">
                            {option.description}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            <div className="hide-predictions-option">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={hidePredictions}
                  onChange={(e) => setHidePredictions(e.target.checked)}
                  className="hide-predictions-checkbox"
                />
                Hide Prediction
              </label>
            </div>
          </div>
        </div>

        {/* Bottom Row - History Window */}
        <div className="history-section">
          <div className="history-display">
            <div className="history-comparison">
              <div className="history-row">
                <div className="history-label">User Entry:</div>
                <div className="history-digits user-digits">
                  {Array.from({ length: 100 }, (_, index) => {
                    const digit = inputHistory[index];
                    return (
                      <span 
                        key={index} 
                        className={`digit grid-cell ${digit ? 'filled' : 'empty'}`}
                      >
                        {digit || ''}
                      </span>
                    );
                  })}
                </div>
              </div>
              <div className="history-row">
                <div className="history-label">Predictions:</div>
                <div className="history-digits prediction-digits">
                  {Array.from({ length: 100 }, (_, index) => {
                    const userDigit = inputHistory[index];
                    const predictedDigit = score.predictions && score.predictions[index] 
                      ? score.predictions[index] 
                      : null;
                    
                    let className = 'digit grid-cell';
                    let displayContent = '';
                    
                    // Show current prediction (for next digit to be typed) or past prediction results
                    if (index === inputHistory.length && prediction && !isCompleted) {
                      // This is the next digit position - show current prediction (or hide if checkbox is checked)
                      className += ' prediction-next';
                      displayContent = hidePredictions ? '?' : prediction;
                    } else if (!userDigit) {
                      // Empty cell - not filled yet
                      className += ' empty';
                      displayContent = '';
                    } else if (!predictedDigit) {
                      // No prediction was made for this position
                      className += ' placeholder';
                      displayContent = '-';
                    } else {
                      // Has both user digit and prediction - show if it was correct
                      const wasCorrect = predictedDigit === userDigit;
                      className += wasCorrect ? ' correct' : ' incorrect';
                      displayContent = predictedDigit;
                    }
                    
                    return (
                      <span key={index} className={className}>
                        {displayContent}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
          <div className="progress-info">
            <span className="progress-text">{inputHistory.length}/100 digits completed</span>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${(inputHistory.length / 100) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Reset Button */}
      {score.total > 0 && (
        <div className="game-controls">
          <button onClick={handleTryAgain} className="reset-btn">
            Start Over
          </button>
        </div>
      )}

      {/* Final Results */}
      {isCompleted && score.total > 0 && (
        <div id="verdict-section" className="verdict-section">
          <div className={`verdict-card ${isHuman ? 'human' : 'robot'}`}>
            <h2>The Final Verdict</h2>
            <div className="verdict-result">
              {isHuman ? (
                <>
                  <span className="verdict-text">You appear to be HUMAN</span>
                  <span className="verdict-explanation">
                    You successfully avoided being predictable {unpredictabilityRate}% of the time. Congratulations on your agency! Use it wisely!
                  </span>
                </>
              ) : (
                <>
                  <span className="verdict-text">You appear to be a ROBOT</span>
                  <span className="verdict-explanation">
                    You only avoided being predictable {unpredictabilityRate}% of the time. Are you sure you're human?
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
};

export default GamePage;
