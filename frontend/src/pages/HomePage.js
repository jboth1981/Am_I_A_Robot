import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import LandingPage from '../components/LandingPage';

const HomePage = () => {
  const navigate = useNavigate();
  const { playAsGuest } = useAuth();

  const handleAuthAction = (action) => {
    if (action === 'guest') {
      playAsGuest();
    } else {
      navigate('/auth');
    }
  };

  return <LandingPage onAuthAction={handleAuthAction} />;
};

export default HomePage;
