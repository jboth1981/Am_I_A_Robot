import React from 'react';
import '../App.css';

function About() {
  return (
    <div className="about-page">
      <div className="about-content">
        <h1 className="about-title">About <em>Am I A Robot</em></h1>
        <p className="project-tagline">
          An interactive experiment exploring the tension between free will and predictability.
        </p>

        <div className="about-section">
          <h2>Project Overview</h2>
          <p>
            This project explores the tension between our <em>sense of free will</em> and the <em>predictability of human behavior</em>. 
            While we often feel that our choices are freely made and independent of outside influence, research in neuroscience and cognitive psychology 
            suggests that many of our decisions may be more predictable than we realize.
          </p>
          <p>
            The experiment is inspired by studies showing that the outcomes of seemingly free decisions can be predicted several seconds before 
            a person becomes consciously aware of choosing. This project translates that concept into an interactive experiment, using machine learning 
            to test whether algorithms can detect hidden patterns in what feels like "random" human input.
          </p>
          <p><strong>Scientific References:</strong></p>
          <ul>
            <li>
              <a 
                href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3625266/" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                Soon, C. S., et al. (2013). <em>Predicting free choices for abstract intentions</em>
              </a>
            </li>
            <li>
              <a 
                href="https://en.wikipedia.org/wiki/Neuroscience_of_free_will" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                Neuroscience of free will (Wikipedia)
              </a>
            </li>
          </ul>
        </div>

        <div className="about-section">
          <h2>Technical Implementation</h2>
          <p>
            This full-stack web application demonstrates modern machine learning and software engineering practices. 
            A custom-trained transformer neural network predicts binary sequences in real time, supported by a robust backend API 
            and interactive data visualization.
          </p>

          <h3>Machine Learning Pipeline</h3>
          <ul>
            <li><strong>Model:</strong> Custom transformer with 4 attention heads</li>
            <li><strong>Framework:</strong> PyTorch with GPU acceleration</li>
            <li><strong>Optimization:</strong> Systematic grid search across learning rates, batch sizes, and architectural parameters</li>
            <li><strong>Performance:</strong> Achieved <strong>63.9%</strong> validation accuracy on binary sequence prediction</li>
          </ul>

          <h3>Full-Stack Architecture</h3>
          <ul>
            <li><strong>Frontend:</strong> React.js with Chart.js for real-time data visualization</li>
            <li><strong>Backend:</strong> FastAPI with PostgreSQL database</li>
            <li><strong>Deployment:</strong> Docker containerization with production-ready configuration</li>
            <li><strong>API:</strong> RESTful endpoints supporting real-time prediction</li>
          </ul>

          <p>
            <strong>Source Code:</strong>{' '}
            <a 
              href="https://github.com/jboth1981/Am_I_A_Robot" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </p>
        </div>

        <div className="about-section">
          <h2>About the Developer</h2>
          <p>
            This project was developed as a demonstration of full-stack machine learning capabilities—combining modern web technologies 
            with deep learning techniques. It highlights skills in neural network architecture design, hyperparameter optimization, 
            and end-to-end production deployment.
          </p>
          <p>
            <strong>Contact:</strong>{' '}
            <a href="mailto:jordan.bothwell@gmail.com">jordan.bothwell@gmail.com</a>{' '}|{' '}
            <a 
              href="https://www.linkedin.com/in/jordan-bothwell-b8746110/" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              LinkedIn
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

export default About;