# Authentication Setup

This frontend now includes JWT authentication with **guest mode** support that integrates with your FastAPI backend.

## Features Added

### 🔐 Authentication System
- **Login Form**: Username and password authentication
- **Register Form**: New user registration with email validation
- **Guest Mode**: Play without authentication (no stats saved)
- **JWT Token Management**: Automatic token storage and validation
- **User Profile**: Display current user info with logout functionality

### 🎮 Guest Mode Features
- **No Registration Required**: Users can play immediately
- **Full Functionality**: Access to all robot detection features
- **No Data Persistence**: Progress and stats are not saved
- **Easy Upgrade**: One-click option to login and save progress

### 📁 New Files Created
- `src/contexts/AuthContext.js` - Authentication state management
- `src/services/authService.js` - API calls for authentication
- `src/components/LoginForm.js` - Login form component
- `src/components/RegisterForm.js` - Registration form component
- `src/components/AuthWrapper.js` - Authentication wrapper component
- `src/components/AuthForm.css` - Styles for authentication forms
- `src/MainApp.js` - Main application component (renamed from App.js)
- `.env` - Environment configuration

### 🔧 Configuration

The app uses the following environment variable:
- `REACT_APP_API_URL` - Backend API URL (defaults to `http://localhost:8000`)

### 🚀 How It Works

1. **First Visit**: Users see login/register forms with a "Play as Guest" option
2. **Guest Mode**: Users can play immediately without any authentication
3. **Full Access**: Guests get access to all robot detection features
4. **Authentication**: Users can register or login for enhanced features
5. **Token Management**: JWT tokens are automatically stored and sent with API requests
6. **Logout**: Users can logout, which clears their session and returns to login screen

### 🔗 Backend Integration

The frontend integrates with your existing FastAPI endpoints:
- `POST /register/` - User registration
- `POST /login/` - User authentication
- `GET /me/` - Get current user info
- `POST /predict/` - Make predictions (now requires authentication)

### 🎨 UI/UX Features

- **Responsive Design**: Works on mobile and desktop
- **Error Handling**: Clear error messages for failed authentication
- **Loading States**: Visual feedback during authentication
- **Form Validation**: Client-side validation for registration
- **Smooth Transitions**: Animated form switching between login/register

The authentication system is now fully integrated and ready to use!