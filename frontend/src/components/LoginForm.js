import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { authService } from '../services/authService';
import './AuthForm.css';

export const LoginForm = ({ onSwitchToRegister, onPlayAsGuest, onBackToLanding }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [forgotPasswordEmail, setForgotPasswordEmail] = useState('');
  const [forgotPasswordLoading, setForgotPasswordLoading] = useState(false);
  const [forgotPasswordMessage, setForgotPasswordMessage] = useState('');
  const { login, error, clearError } = useAuth();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (error) {
      clearError();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    const result = await login(formData.username, formData.password);
    
    if (!result.success) {
      setIsLoading(false);
    }
    // If successful, the AuthContext will handle the state update
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    setForgotPasswordLoading(true);
    setForgotPasswordMessage('');
    
    try {
      await authService.requestPasswordReset(forgotPasswordEmail);
      setForgotPasswordMessage('If the email exists, a password reset link has been sent to your inbox.');
    } catch (error) {
      setForgotPasswordMessage('Error requesting password reset. Please try again.');
    }
    
    setForgotPasswordLoading(false);
  };

  if (showForgotPassword) {
    return (
      <div className="auth-form-container">
        <div className="auth-form-card">
          <h2 className="auth-form-title">Reset Password</h2>
          <p className="auth-form-subtitle">Enter your email to receive a reset link</p>
          
          {forgotPasswordMessage && (
            <div className="auth-message">
              {forgotPasswordMessage}
            </div>
          )}
          
          <form onSubmit={handleForgotPassword} className="auth-form">
            <div className="form-group">
              <label htmlFor="email" className="form-label">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={forgotPasswordEmail}
                onChange={(e) => setForgotPasswordEmail(e.target.value)}
                className="form-input"
                placeholder="Enter your email"
                required
                disabled={forgotPasswordLoading}
              />
            </div>
            
            <button
              type="submit"
              className="auth-submit-btn"
              disabled={forgotPasswordLoading}
            >
              {forgotPasswordLoading ? 'Sending...' : 'Send Reset Link'}
            </button>
          </form>
          
          <div className="auth-switch">
            <button
              type="button"
              onClick={() => setShowForgotPassword(false)}
              className="auth-switch-btn"
              disabled={forgotPasswordLoading}
            >
              ← Back to Login
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-form-container">
      <div className="auth-form-card">
        <h2 className="auth-form-title">Log In</h2>
        
        {error && (
          <div className="auth-error">
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="username" className="form-label">
              Username
            </label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="form-input"
              placeholder="Enter your username"
              required
              disabled={isLoading}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password" className="form-label">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="form-input"
              placeholder="Enter your password"
              required
              disabled={isLoading}
            />
            <div style={{ marginTop: '4px', textAlign: 'right' }}>
              <button
                type="button"
                onClick={() => setShowForgotPassword(true)}
                className="auth-switch-btn"
                disabled={isLoading}
                style={{ fontSize: '0.8rem', padding: '0' }}
              >
                Forgot Password?
              </button>
            </div>
          </div>
          
          <button
            type="submit"
            className="auth-submit-btn"
            disabled={isLoading}
          >
            {isLoading ? 'Logging In...' : 'Log In'}
          </button>
        </form>
        
        <div className="auth-switch">
          <div className="guest-option">
            <p>Or</p>
            <button
              type="button"
              onClick={onPlayAsGuest}
              className="guest-btn auth-submit-btn"
              disabled={isLoading}
            >
              Play as Guest
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};