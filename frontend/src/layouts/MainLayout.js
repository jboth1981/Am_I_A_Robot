import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Sidebar from '../components/Sidebar';

const MainLayout = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { playAsGuest } = useAuth();

  // Map routes to page names for sidebar
  const getCurrentPage = () => {
    switch (location.pathname) {
      case '/':
        return 'landing';
      case '/play':
        return 'play';
      case '/about':
        return 'about';
      case '/auth':
        return 'auth';
      default:
        return 'landing';
    }
  };

  const handlePageChange = (page) => {
    switch (page) {
      case 'play':
        navigate('/play');
        break;
      case 'about':
        navigate('/about');
        break;
      case 'landing':
      default:
        navigate('/');
        break;
    }
  };

  const handleAuthAction = (action) => {
    if (action === 'guest') {
      playAsGuest();
    } else if (action === 'register') {
      navigate('/auth?mode=register');
    } else {
      navigate('/auth');
    }
  };

  return (
    <div className="app">
      <Sidebar 
        currentPage={getCurrentPage()}
        onPageChange={handlePageChange}
        onAuthAction={handleAuthAction}
      />
      
      <div className="main-content">
        <div className="container">
          {children}
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
