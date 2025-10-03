import React, { useState } from 'react';
import { authService } from '../services/authService';
import './AuthForm.css';

export const PasswordResetForm = ({ token, onSuccess, onCancel }) => {
  const [formData, setFormData] = useState({
    newPassword: '',
    confirmPassword: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear errors when user starts typing
    if (error) {
      setError('');
    }
    if (message) {
      setMessage('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate passwords match
    if (formData.newPassword !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    // Validate password strength
    if (formData.newPassword.length < 6) {
      setError('Password must be at least 6 characters long');
      return;
    }

    setIsLoading(true);
    setError('');
    setMessage('');

    try {
      await authService.resetPassword(token, formData.newPassword);
      setMessage('Password reset successfully! You can now log in with your new password.');
      setTimeout(() => {
        onSuccess();
      }, 2000);
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Failed to reset password. The link may have expired.';
      setError(errorMessage);
    }
    
    setIsLoading(false);
  };

  return (
    <div className="auth-form-container">
      <div className="auth-form-card">
        <h2 className="auth-form-title">Reset Password</h2>
        <p className="auth-form-subtitle">Enter your new password</p>
        
        {message && (
          <div className="auth-message success">
            {message}
          </div>
        )}
        
        {error && (
          <div className="auth-message error">
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="newPassword" className="form-label">
              New Password
            </label>
            <input
              type="password"
              id="newPassword"
              name="newPassword"
              value={formData.newPassword}
              onChange={handleChange}
              className="form-input"
              placeholder="Enter new password"
              required
              disabled={isLoading}
              minLength="6"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="confirmPassword" className="form-label">
              Confirm Password
            </label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              className="form-input"
              placeholder="Confirm new password"
              required
              disabled={isLoading}
              minLength="6"
            />
          </div>
          
          <button
            type="submit"
            className="auth-submit-btn"
            disabled={isLoading}
          >
            {isLoading ? 'Resetting...' : 'Reset Password'}
          </button>
        </form>
        
        <div className="auth-switch">
          <button
            type="button"
            onClick={onCancel}
            className="auth-switch-btn"
            disabled={isLoading}
          >
            ← Back to Login
          </button>
        </div>
      </div>
    </div>
  );
};