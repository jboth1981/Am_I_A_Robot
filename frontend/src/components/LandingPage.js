import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import Sidebar from './Sidebar';

const LandingPage = ({ onAuthAction }) => {
  const { playAsGuest } = useAuth();
  const [showMoreInfo, setShowMoreInfo] = useState(false);

  const handleAuthAction = (action) => {
    if (action === 'guest') {
      playAsGuest();
    } else {
      onAuthAction(action);
    }
  };

  const handleGetStarted = () => {
    onAuthAction('login');
  };

  return (
    <div className="app landing-page">
      <Sidebar 
        currentPage="landing"
        onPageChange={() => {}} // Landing page doesn't navigate away
        onAuthAction={handleAuthAction}
      />
      
      <div className="main-content">
        <div className="container">
          {/* Simple Hero Section */}
          <div className="landing-hero-simple">
            <h1 className="landing-title">Are You A Robot?</h1>
            <p className="landing-subtitle">
              Think you're unpredictable? Our AI will be the judge of that.
            </p>
            
            <div className="landing-actions">
              <button 
                className="get-started-btn"
                onClick={handleGetStarted}
              >
                Get Started
              </button>
              
              <button 
                className="learn-more-btn"
                onClick={() => setShowMoreInfo(!showMoreInfo)}
              >
                {showMoreInfo ? '↑ Less Info' : '↓ Learn More'}
              </button>
            </div>
          </div>

          {/* Expandable Content */}
          {showMoreInfo && (
            <div className="landing-expandable">
              <div className="landing-description-expanded">
                <p>
                  Enter a sequence of 0s and 1s while trying to be as random as possible. 
                  Our advanced prediction algorithms will analyze your patterns and determine 
                  whether you behave more like a human or a robot.
                </p>
              </div>

              <div className="landing-how-it-works">
                <h2>How It Works</h2>
                <div className="steps">
                  <div className="step">
                    <div className="step-number">1</div>
                    <h4>Enter Your Sequence</h4>
                    <p>Type 0s and 1s trying to be as unpredictable as possible. Humans naturally create patterns even when trying to be random, while true randomness is surprisingly difficult to achieve.</p>
                  </div>
                  <div className="step">
                    <div className="step-number">2</div>
                    <h4>AI Analyzes Patterns</h4>
                    <p>Our transformer neural network analyzes your input patterns in real-time, looking for subtle biases, frequency patterns, and sequential dependencies that reveal human cognitive limitations.</p>
                  </div>
                  <div className="step">
                    <div className="step-number">3</div>
                    <h4>Get Your Verdict</h4>
                    <p>After 100 digits, discover if you're more human or robot based on predictability. The AI calculates an accuracy score - lower scores indicate more human-like unpredictability.</p>
                  </div>
                </div>
              </div>

              <div className="landing-cta-expanded">
                <div className="cta-content">
                  <h2>Ready to Test Your Humanity?</h2>
                  <p>Choose how you'd like to proceed:</p>
                  
                  <div className="cta-buttons">
                    <button 
                      className="cta-btn primary"
                      onClick={() => handleAuthAction('register')}
                    >
                      Sign Up
                    </button>
                    <button 
                      className="cta-btn secondary"
                      onClick={() => handleAuthAction('login')}
                    >
                      Log In
                    </button>
                    <button 
                      className="cta-btn guest"
                      onClick={() => handleAuthAction('guest')}
                    >
                      Try as Guest
                    </button>
                  </div>
                  
                  <p className="cta-note">
                    Guest mode lets you play immediately, but your results won't be saved.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
