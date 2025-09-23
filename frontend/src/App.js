import React, { useState, useEffect } from 'react';
import { useAuth } from './contexts/AuthContext';
import { AuthWrapper } from './components/AuthWrapper';
import { PasswordResetForm } from './components/PasswordResetForm';
import MainApp from './MainApp';

function App() {
  const { loading, canAccessApp } = useAuth();
  const [resetToken, setResetToken] = useState(null);
  const [showResetForm, setShowResetForm] = useState(false);

  // Check for password reset token in URL
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    
    if (token) {
      setResetToken(token);
      setShowResetForm(true);
      // Clean up the URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }, []);

  const handleResetSuccess = () => {
    setShowResetForm(false);
    setResetToken(null);
  };

  const handleResetCancel = () => {
    setShowResetForm(false);
    setResetToken(null);
  };

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.95)', 
          padding: '40px', 
          borderRadius: '12px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.2rem', color: '#7f8c8d' }}>
            Loading...
          </div>
        </div>
      </div>
    );
  }

  // Show password reset form if token is present
  if (showResetForm && resetToken) {
    return (
      <PasswordResetForm 
        token={resetToken} 
        onSuccess={handleResetSuccess}
        onCancel={handleResetCancel}
      />
    );
  }

  return (
    <>
      {canAccessApp ? <MainApp /> : <AuthWrapper />}
    </>
  );
}

export default App;