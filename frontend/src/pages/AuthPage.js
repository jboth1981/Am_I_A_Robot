import React from 'react';
import { AuthWrapper } from '../components/AuthWrapper';
import { useNavigate, useSearchParams } from 'react-router-dom';

const AuthPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const handleBackToLanding = () => {
    navigate('/');
  };

  // Check if we should start in registration mode
  const mode = searchParams.get('mode') || 'login';

  return <AuthWrapper onBackToLanding={handleBackToLanding} defaultMode={mode} />;
};

export default AuthPage;
