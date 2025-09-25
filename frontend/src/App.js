import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import { PasswordResetForm } from './components/PasswordResetForm';
import MainLayout from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import GamePage from './pages/GamePage';
import AboutPage from './pages/AboutPage';
import AuthPage from './pages/AuthPage';

// Component that handles auth logic and routing
function AppContent() {
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
        background: 'var(--white)'
      }}>
        <div style={{ 
          background: 'var(--card-bg)', 
          padding: '40px', 
          borderRadius: '8px',
          textAlign: 'center',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          border: '1px solid #f0f0f0'
        }}>
          <div style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>
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
    <Routes>
      {/* Public routes */}
      <Route 
        path="/" 
        element={
          <MainLayout>
            {canAccessApp ? <Navigate to="/play" replace /> : <HomePage />}
          </MainLayout>
        } 
      />
      
      {/* Auth route - only show if not authenticated */}
      <Route 
        path="/auth" 
        element={
          canAccessApp ? <Navigate to="/play" replace /> : (
            <MainLayout>
              <AuthPage />
            </MainLayout>
          )
        } 
      />
      
      {/* Protected/Public routes */}
      <Route 
        path="/play" 
        element={
          <MainLayout>
            <GamePage />
          </MainLayout>
        } 
      />
      
      <Route 
        path="/about" 
        element={
          <MainLayout>
            <AboutPage />
          </MainLayout>
        } 
      />
      
      {/* Catch all route - redirect to home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

// Main App component that provides Router context
function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;