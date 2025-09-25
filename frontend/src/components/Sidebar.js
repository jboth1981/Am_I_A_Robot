import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Sidebar = ({ currentPage, onPageChange, onAuthAction }) => {
  const { user, isGuest, logout } = useAuth();
  const navigate = useNavigate();

  const menuItems = [
    { id: 'play', label: 'Play', path: '/play' },
    { id: 'about', label: 'About', path: '/about' },
    // Add more menu items here as needed
    // { id: 'stats', label: 'Statistics', path: '/stats' },
    // { id: 'leaderboard', label: 'Leaderboard', path: '/leaderboard' },
  ];


  const handleLogout = () => {
    logout();
  };

  const handleAuthClick = (mode) => {
    navigate(`/auth?mode=${mode}`);
  };

  return (
    <div className="sidebar">
      {/* Logo Section */}
      <div className="sidebar-logo">
        <Link to="/" className="sidebar-logo-link">
          <img 
            src="/images/am_i_a_robot_large.png" 
            alt="Am I A Robot Logo" 
          />
        </Link>
      </div>

      {/* Navigation Menu */}
      <div className="sidebar-nav">
        {menuItems.map((item) => (
          <Link
            key={item.id}
            to={item.path}
            className={`sidebar-nav-item ${currentPage === item.id ? 'active' : ''}`}
          >
            {item.label}
          </Link>
        ))}
      </div>

      {/* Authentication Section */}
      {!user && !isGuest && (
        <div className="sidebar-auth">
          <Link 
            to="/auth?mode=register"
            className="sidebar-auth-btn signup"
          >
            Sign Up
          </Link>
          <Link 
            to="/auth"
            className="sidebar-auth-btn login"
          >
            Log In
          </Link>
        </div>
      )}

      {/* User Profile Section */}
      {user && !isGuest && (
        <div className="sidebar-user-profile">
          <div className="sidebar-user-info">
            <div className="sidebar-user-name">{user.username}</div>
            <div className="sidebar-user-email">{user.email}</div>
          </div>
          <button 
            className="sidebar-logout-btn"
            onClick={handleLogout}
          >
            Logout
          </button>
        </div>
      )}

      {/* Guest Notice Section */}
      {isGuest && (
        <div className="sidebar-guest-notice">
          <div className="sidebar-guest-title">Playing as Guest</div>
          <div className="sidebar-guest-subtitle">Your progress won't be saved</div>
          <button 
            className="sidebar-login-btn"
            onClick={() => handleAuthClick('login')}
          >
            Log In to Save Progress
          </button>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
