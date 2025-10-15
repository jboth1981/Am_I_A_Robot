import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { authService } from '../services/authService';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

const GamePage = () => {

  const [predictionMethod, setPredictionMethod] = useState('transformer');
  const [showMethodDropdown, setShowMethodDropdown] = useState(false);
  
  const [inputHistory, setInputHistory] = useState('');
  const [prediction, setPrediction] = useState('');

  const [score, setScore] = useState({ correct: 0, total: 0, predictions: [] });

  const [isPredictionInProgress, setIsPredictionInProgress] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  const [hidePredictions, setHidePredictions] = useState(false);
  const [showVerdict, setShowVerdict] = useState(false);
  const [showHumanText, setShowHumanText] = useState(false);
  const [verdictStage, setVerdictStage] = useState(0); // 0: hidden, 1: "You Are A:", 2: stamp, 3: tagline, 4: full text
  const [unpredictabilityHistory, setUnpredictabilityHistory] = useState([]);
  const initialPredictionMade = useRef(false);
  const handleKeyPressRef = useRef();
  const lastProcessedPosition = useRef(-1);
  const mobileInputRef = useRef(null);
  const { user, token } = useAuth();
  // Ensure a focusable input exists for mobile keyboards
  useEffect(() => {
    if (mobileInputRef && mobileInputRef.current) {
      mobileInputRef.current.focus();
    }
  }, []);

  // Chart configuration constants
  const CHART_COLORS = {
    robot: 'rgba(220, 53, 69, 0.1)',    // Light red
    human: 'rgba(40, 167, 69, 0.1)',   // Light green
    line: '#6c757d',                    // Light grey
    predicted: '#dc3545',               // Red for predicted
    unpredicted: '#28a745',             // Green for unpredicted
  };

  // Helper function to determine if prediction was correct
  const wasPredictionCorrect = (index) => {
    return score.predictions?.[index] && 
           inputHistory[index] && 
           score.predictions[index] !== inputHistory[index];
  };

  // Chart.js configuration
  const chartData = {
    labels: Array.from({ length: 100 }, (_, index) => index + 1),
    datasets: [
      // Robot region background (0-50%)
      {
        label: 'Robot Region',
        data: Array(100).fill(50),
        backgroundColor: CHART_COLORS.robot,
        borderColor: 'transparent',
        borderWidth: 0,
        pointRadius: 0,
        fill: 'origin',
        tension: 0,
      },
      // Human region background (50-100%)
      {
        label: 'Human Region',
        data: Array(100).fill(50),
        backgroundColor: CHART_COLORS.human,
        borderWidth: 0,
        pointRadius: 0,
        fill: 'end',
        tension: 0,
      },
      // Main unpredictability line
      {
        label: 'Unpredictability %',
        data: Array.from({ length: 100 }, (_, index) => 
          index < unpredictabilityHistory.length ? unpredictabilityHistory[index] : null
        ),
        borderColor: CHART_COLORS.line,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: Array.from({ length: 100 }, (_, index) => {
          if (index >= unpredictabilityHistory.length) return 'transparent';
          return wasPredictionCorrect(index) ? CHART_COLORS.unpredicted : CHART_COLORS.predicted;
        }),
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
        tension: 0.1,
        fill: false,
        spanGaps: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Disable animations
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            if (context.datasetIndex === 2) { // Main unpredictability line
              const index = context.dataIndex;
              const status = wasPredictionCorrect(index) ? 'Unpredicted (Human-like)' : 'Predicted (Robot-like)';
              return `Entry ${context.label}: ${context.parsed.y}% unpredictable - ${status}`;
            }
            return null;
          }
        }
      },
    },
    scales: {
      x: {
        title: {
          display: false // Remove "Entry Number" label
        },
        min: 1,
        max: 100,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          stepSize: 10, // Show only 10, 20, 30, etc.
          min: 10,
          max: 100,
          callback: function(value) {
            return value % 10 === 0 ? value : ''; // Only show multiples of 10
          },
          maxRotation: 0, // Keep labels horizontal
          minRotation: 0
        }
      },
      y: {
        title: {
          display: false // Remove "Unpredictability %" label
        },
        min: 0,
        max: 100,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          callback: function(value) {
            return value + '%';
          },
          stepSize: 25, // Force ticks at 0, 25, 50, 75, 100
          min: 0,
          max: 100
        }
      },
    },
    elements: {
      point: {
        hoverBackgroundColor: '#1e7e34',
      },
    },
  };

  // Handle global keyboard input
  const handleKeyPress = useCallback(async (e) => {
    // Only process if game is not completed
    if (isCompleted) {
      return;
    }

    // Check if key is 0 or 1
    if (e.key !== '0' && e.key !== '1') {
      return;
    }

    // Prevent processing if already in progress
    if (isPredictionInProgress) {
      return;
    }

    const newChar = e.key;
    
    // Prevent duplicate processing (React StrictMode causes double execution)
    if (lastProcessedPosition.current >= inputHistory.length) {
      return;
    }
    
    // Mark this position as processed
    lastProcessedPosition.current = inputHistory.length;
    
    // Set processing flag
    setIsPredictionInProgress(true);

    try {
      // Process the character
      const newHistory = inputHistory + newChar;
      
      // Capture the current prediction BEFORE updating history
      const currentPrediction = prediction;
      
      // Update history
      setInputHistory(newHistory);

              // Store the prediction that was made for this position and check if it was correct
              if (currentPrediction) {
                const wasCorrect = (currentPrediction === newChar);
                setScore(prev => {
                  const newScore = {
                    correct: prev.correct + (wasCorrect ? 1 : 0),
                    total: prev.total + 1,
                    predictions: [...prev.predictions, currentPrediction]
                  };
                  
                  return newScore;
                });
              }

              // Check if we've reached 100 characters
              if (newHistory.length >= 100) {
                setIsCompleted(true);
                // Start the multi-stage verdict animation sequence
                setShowVerdict(true);
                setVerdictStage(1); // Show "You Are A:"
                
                setTimeout(() => {
                  setVerdictStage(2); // Show stamp
                }, 1500);
                
                setTimeout(() => {
                  setVerdictStage(3); // Show tagline
                }, 4000);
                
                setTimeout(() => {
                  setVerdictStage(4); // Show full text
                }, 6000);
                return;
              }

      // Get new prediction for next character
      try {
        const data = await authService.predict({ 
          history: newHistory,
          method: predictionMethod
        }, token);
        setPrediction(data.prediction);
      } catch (error) {
        console.error('Error fetching prediction:', error);
        setPrediction('?');
      }
    } finally {
      // Always reset processing flag
      setIsPredictionInProgress(false);
    }
  }, [isCompleted, isPredictionInProgress, inputHistory, prediction, predictionMethod, token]);

  // Keep the ref updated with the latest handleKeyPress function
  useEffect(() => {
    handleKeyPressRef.current = handleKeyPress;
  }, [handleKeyPress]);

  // Update chart when score changes (separate from score calculation)
  useEffect(() => {
    if (score.total > 0) {
      const unpredictabilityPercent = Math.round((1 - score.correct / score.total) * 100);
      setUnpredictabilityHistory(prevHistory => {
        // Only add if this is a new entry (prevent duplicates)
        if (prevHistory.length < score.total) {
          return [...prevHistory, unpredictabilityPercent];
        }
        return prevHistory;
      });
    }
  }, [score.total, score.correct]);



  const handleMethodSelect = async (method) => {
    setPredictionMethod(method);
    setShowMethodDropdown(false);
    
    // Get initial prediction for the new method (only if no digits typed yet)
    if (inputHistory.length === 0) {
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
      label: 'Transformer',
      description: 'Uses a neural network trained on binary sequences to predict the next digit'
    }
  ];

  const unpredictabilityRate = score.total > 0 ? (((score.total - score.correct) / score.total) * 100).toFixed(1) : 0;
  const isHuman = unpredictabilityRate >= 50; // If unpredictability is 50% or higher, you're more "human"

  // Add global keyboard listener
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      handleKeyPressRef.current(e);
    };

    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []); // Empty dependency array - listener added once, ref keeps it current

  // Get initial prediction when component mounts (only when no digits typed)
  React.useEffect(() => {
    if (inputHistory.length === 0 && !initialPredictionMade.current) {
      const getInitialPrediction = async () => {
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
        }
      };
      
      getInitialPrediction();
    }
  }, [inputHistory.length, token, predictionMethod]);

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
    setShowVerdict(false);
    setShowHumanText(false);
    setVerdictStage(0);
    setUnpredictabilityHistory([]);
    lastProcessedPosition.current = -1; // Reset position tracking
    // Keep the current prediction method instead of resetting to 'frequency'
    // setPredictionMethod('frequency'); // Removed this line
    setShowMethodDropdown(false);
    setHidePredictions(false); // Reset hide predictions checkbox
    initialPredictionMade.current = false; // Reset the ref so initial prediction will be made again
    // Initial prediction will be triggered automatically by useEffect
  };

  return (
    <main className="game-main">
      <div 
        className="game-layout"
        onClick={() => mobileInputRef.current && mobileInputRef.current.focus()}
        onTouchStart={() => mobileInputRef.current && mobileInputRef.current.focus()}
      >
        {/* Invisible offscreen input to trigger the on-screen keyboard on mobile */}
        <input
          id="mobile-binary-input"
          ref={mobileInputRef}
          autoFocus
          inputMode="numeric"
          pattern="[01]*"
          onKeyDown={handleKeyPress}
          onBlur={() => mobileInputRef.current && mobileInputRef.current.focus()}
          aria-hidden="true"
          style={{
            position: 'absolute',
            left: '-9999px',
            width: 1,
            height: 1,
            opacity: 0,
            border: 0,
            padding: 0,
            outline: 'none'
          }}
        />
        
        {/* Method Selection */}
        <div className="game-method-section">
          <div className="game-method-controls">
            <div className="method-row">
              <div className="method-controls-group">
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
                
                <div className="help-button-container">
                  <div className="help-button" title="How to Play">
                    <span className="help-icon">?</span>
                    <div className="help-tooltip">
                      <h4>How to Play</h4>
                      <ol>
                        <li><strong>Choose a prediction method</strong> (Frequency Analysis, Pattern Recognition, or Transformer)</li>
                        <li><strong>Type digits anywhere</strong> (press 0 or 1 keys anywhere on the page)</li>
                        <li><strong>Watch the AI predict</strong> your next digit based on your pattern</li>
                        <li><strong>See the results</strong> in real-time - green for correct predictions, red for incorrect</li>
                        <li><strong>Complete 100 digits</strong> to get your final humanity score!</li>
                      </ol>
                      <p><em>The goal: Be unpredictable! If the AI can't guess your pattern, you're more human.</em></p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Start Over Button */}
              {score.total > 0 && (
                <div className="start-over-container">
                  <button onClick={handleTryAgain} className="reset-btn">
                    Start Over
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Grid Section */}
        <div className="grid-section">
          <div className="history-display">
            <div className="history-comparison">
              {Array.from({ length: 2 }, (_, rowIndex) => (
                <div key={rowIndex} className="history-row-pair">
                  {/* User Entry Row */}
                  <div className="history-row">
                    <div className="history-label">{rowIndex === 0 ? "User Entry:" : ""}</div>
                    <div className="history-spacer"></div>
                    <div className="history-digits user-digits">
                      {Array.from({ length: 50 }, (_, colIndex) => {
                        const index = rowIndex * 50 + colIndex;
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
                  
                  {/* Prediction Row */}
                  <div className="history-row">
                    <div className="history-label">{rowIndex === 0 ? "Predictions:" : ""}</div>
                    <div className="history-spacer"></div>
                    <div className="history-digits prediction-digits">
                      {Array.from({ length: 50 }, (_, colIndex) => {
                        const index = rowIndex * 50 + colIndex;
                        const userDigit = inputHistory[index];
                        // Use stored predictions for past positions, current prediction for next position
                        const predictedDigit = index < inputHistory.length
                          ? score.predictions && score.predictions[index] 
                            ? score.predictions[index]  // Stored prediction for past digits
                            : null
                          : index === inputHistory.length && prediction && !isCompleted
                            ? prediction  // Current prediction for next digit
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
              ))}
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

        {/* Bottom Section - Stats and Chart */}
        <div className="game-bottom-section">
          <div className="stats-chart-container">
            {/* Stats Section */}
            <div className="game-stats-section">
              <div className="game-stats-card">
                <div className="game-stats-vertical">
                  <div className="game-stat-row">
                    <span className="game-stat-label">Total:</span>
                    <span className="game-stat-number">{score.total}</span>
                  </div>
                  <div className="game-stat-row">
                    <span className="game-stat-label">Not Predicted:</span>
                    <span className="game-stat-number">{score.total - score.correct}</span>
                  </div>
                  <div className="game-stat-row">
                    <span className="game-stat-label">Unpredictability:</span>
                    <span className="game-stat-number">{score.total > 0 ? ((score.total - score.correct) / score.total * 100).toFixed(1) : 0}%</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Unpredictability Graph */}
            <div className="game-graph-section">
              <div className="game-graph-card">
                <div className="chart-container">
                  <Line data={chartData} options={chartOptions} />
                  {/* Chart region labels */}
                  <div className="chart-label chart-label-robot">ROBOT</div>
                  <div className="chart-label chart-label-human">HUMAN</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>


      {/* Final Results */}
      {isCompleted && score.total > 0 && (
        <div id="verdict-section" className="verdict-section">
          <div className={`verdict-card ${isHuman ? 'human' : 'robot'}`}>
            <div className="verdict-header">
              {showVerdict && verdictStage >= 1 && (
                <div className="verdict-main-text">
                  <span className="verdict-prompt">
                    You Are A:
                  </span>
                  {verdictStage >= 2 && (
                    <div className="stamp-container">
                      <span className="verdict-label">
                        {isHuman ? 'HUMAN' : 'ROBOT'}
                      </span>
                    </div>
                  )}
                </div>
              )}
              {verdictStage >= 3 && (
                <div className="verdict-tagline">
                  <em>{isHuman ? 'The chaos within prevails!' : 'Order has triumphed!'}</em>
                </div>
              )}
            </div>
            {verdictStage >= 4 && (
              <div className="verdict-details">
                <div className="verdict-result">
                  <span className="verdict-explanation">
                    {isHuman ? (
                      <>You outsmarted the model with {unpredictabilityRate}% unpredictability.<br/>Nicely done — free will still has a fighting chance!</>
                    ) : (
                      <>The model saw through your moves — only {unpredictabilityRate}% escaped prediction.<br/>Don't worry. Even robots make choices… kind of.</>
                    )}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
};

export default GamePage;
