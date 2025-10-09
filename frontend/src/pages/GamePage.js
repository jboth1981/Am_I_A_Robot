import React, { useState, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { authService } from '../services/authService';

const GamePage = () => {

  const [predictionMethod, setPredictionMethod] = useState('frequency');
  const [showMethodDropdown, setShowMethodDropdown] = useState(false);
  
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');

  const [score, setScore] = useState({ correct: 0, total: 0, predictions: [] });

  const [isPredictionInProgress, setIsPredictionInProgress] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  const [hidePredictions, setHidePredictions] = useState(false);
  const [currentInput, setCurrentInput] = useState('');
  const [isInitialLoading, setIsInitialLoading] = useState(false);
  const inputRef = useRef(null);
  const initialPredictionMade = useRef(false);
  const { user, token } = useAuth();

  // This function is called when the user types a digit into the input field
  const handleInputChange = async (e) => {
    const inputValue = e.target.value;

    // When a user enters a digit we need to know the following information:
    // 1. Did they enter a valid digit (0 or 1)?
    // 2. What is the digit they entered?
    // 3. What was the prediction for the current digit?
    // (There should always be a prediction waiting for the user to type)
    
    // Using this information we follow these steps:
    // 1. If the user entered a valid digit:
      // 1.1. Clear the input field.
      // 1.2. Update the input history to include the new digit.
      // 1.3. Compare the prior prediction with the new digit.
      // 1.4. Update the score to include the new digit.
      // 1.5. If this is the last input needed (ie the 100th entry):
        // 1.5.1. Set the isCompleted flag to true.
      // 1.6 Otherwise:
        // 1.6.1. Set the isPredictionInProgress flag to true. This prevents the user from 
        // typing another digit until the prediction is complete and updated.
        // 1.6.2. Send the history to the backend to get a new prediction.
        // 1.6.3. Update the prediction to the new prediction.
        // 1.6.4. Set the isPredictionInProgress flag to false. This allows the user to type another digit.
    // 2. If the user entered an invalid digit:
      // 2.1. Clear the input field.
      // 2.2. Return.

    // Block input while prediction is being calculated
    if (isPredictionInProgress) {
      setCurrentInput('');
      return;
    }

    // Check if digit is valid. We only allow single digit (0 or 1).
    const isValidInput = inputValue.length === 1 && /[01]/.test(inputValue);
    
    if (!isValidInput) {
      setCurrentInput('');
      return;
    }

    // Process the valid input
    const newChar = inputValue;
    setCurrentInput(''); // Clear the input for next digit and maintain focus    

    const newHistory = inputHistory + newChar;    
    setInputHistory(newHistory); // Update history

    // Store the prediction that was made for this position and check if it was correct
    if (prediction) {
      const wasCorrect = (prediction === newChar);
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
    setIsPredictionInProgress(true);
    
    // Don't clear prediction immediately - keep showing previous prediction while loading
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
      setIsPredictionInProgress(false);
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
  };

  const handleInputKeyDown = (e) => {
    // Allow backspace/delete to clear the field, but prevent other navigation
    if (e.key === 'Backspace' || e.key === 'Delete') {
      setCurrentInput('');
    }
    // Only allow 0, 1, backspace, delete, and tab
    if (!/[01]/.test(e.key) && !['Backspace', 'Delete', 'Tab'].includes(e.key)) {
      e.preventDefault();
    }
  };

  const handleMethodSelect = async (method) => {
    setPredictionMethod(method);
    setShowMethodDropdown(false);
    
    // Get initial prediction for the new method (only if no digits typed yet)
    if (inputHistory.length === 0) {
      setIsInitialLoading(true);
      try {
        const data = await authService.predict({ 
          history: '',
          method: method
        }, token);
        
        setPrediction(data.prediction);
        initialPredictionMade.current = true;
      } catch (error) {
        console.error('Error fetching initial prediction:', error);
        setPrediction('?');
        initialPredictionMade.current = true;
      } finally {
        setIsInitialLoading(false);
      }
    }
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
    },
    {
      value: 'transformer',
      label: 'AI Transformer',
      description: 'Uses a neural network trained on binary sequences to predict the next digit'
    }
  ];

  const unpredictabilityRate = score.total > 0 ? (((score.total - score.correct) / score.total) * 100).toFixed(1) : 0;
  const isHuman = unpredictabilityRate > 40; // If unpredictability is high, you're more "human"

  // Get initial prediction when component mounts (only when no digits typed)
  React.useEffect(() => {
    if (inputHistory.length === 0 && !initialPredictionMade.current) {
      const getInitialPrediction = async () => {
        setIsInitialLoading(true);
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
          setIsInitialLoading(false);
        }
      };
      
      getInitialPrediction();
    }
  }, [inputHistory.length, token]); // Removed predictionMethod from dependencies

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
    setIsPredictionInProgress(false);
    setIsCompleted(false);
    setIsInitialLoading(false); // Reset initial loading state
    // Keep the current prediction method instead of resetting to 'frequency'
    // setPredictionMethod('frequency'); // Removed this line
    setShowMethodDropdown(false);
    setHidePredictions(false); // Reset hide predictions checkbox
    setCurrentInput(''); // Clear the input field
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
                  value={currentInput}
                  onChange={handleInputChange}
                  onKeyDown={handleInputKeyDown}
                  className={`single-digit-input ${isPredictionInProgress ? 'processing' : ''}`}
                  maxLength="1"
                  disabled={isCompleted || isPredictionInProgress}
                  autoFocus
                />
              </div>
              
              <div className="prediction-area">
                <label>Prediction</label>
                <div className="prediction-display">
                  {isPredictionInProgress || isInitialLoading ? (
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
                    } else if (!userDigit && index === 0 && prediction && !isCompleted) {
                      // Show initial prediction at position 0 before user types anything
                      className += ' prediction-initial';
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
