import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { LoginForm } from './LoginForm';
import { RegisterForm } from './RegisterForm';
import './AuthForm.css';

export const AuthWrapper = ({ children }) => {
  const [isLoginMode, setIsLoginMode] = useState(true);
  const { loading, playAsGuest } = useAuth();

  if (loading) {
    return (
      <div className="auth-form-container">
        <div className="auth-form-card">
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <div style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>
              Loading...
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {isLoginMode ? (
        <LoginForm onSwitchToRegister={() => setIsLoginMode(false)} onPlayAsGuest={playAsGuest} />
      ) : (
        <RegisterForm onSwitchToLogin={() => setIsLoginMode(true)} onPlayAsGuest={playAsGuest} />
      )}
    </>
  );
};