import React from 'react';
import '../App.css';

function About() {
  return (
    <div className="about-page">
      <div className="about-content">
        <h1 className="about-title">About Am I A Robot?</h1>
        
        <div className="about-section">
          <h2>The Challenge</h2>
          <p>
            Most people believe they have free will. But what if a computer could predict your next choice before you know you chose it? 
            In this experiment, you'll enter a series of 0s and 1s while trying to be as unpredictable as possible.
          </p>
        </div>

        <div className="about-section">
          <h2>How It Works</h2>
          <p>
            Our AI uses two different prediction methods to anticipate your next digit:
          </p>
          <ul>
            <li><strong>Frequency Analysis:</strong> Predicts the most frequently occurring digit in your sequence</li>
            <li><strong>Pattern Recognition:</strong> Uses pattern rules (000→0, 111→1, otherwise repeats last digit)</li>
          </ul>
        </div>

        <div className="about-section">
          <h2>The Verdict</h2>
          <p>
            After 100 characters, we'll analyze how predictable your choices were. If our AI correctly predicts more than 60% of your choices, 
            you might be more "robot-like" than you think. If it predicts less than 60%, congratulations - you've demonstrated human unpredictability!
          </p>
        </div>

        <div className="about-section">
          <h2>The Science</h2>
          <p>
            This is a demonstration of pattern recognition in binary sequences. The AI predicts the next digit based on 
            frequency analysis and pattern recognition algorithms.
          </p>
          <p>
            This experiment demonstrates the fascinating intersection of human psychology, pattern recognition, and artificial intelligence. 
            Even when we try to be random, humans often fall into predictable patterns that machines can exploit.
          </p>
        </div>
      </div>
    </div>
  );
}

export default About;
