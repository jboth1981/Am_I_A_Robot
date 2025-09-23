-- Database initialization script for Am I A Robot application
-- This script runs when the PostgreSQL container starts for the first time
-- Uses IF NOT EXISTS to avoid overwriting existing tables

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Add a comment to the table
COMMENT ON TABLE users IS 'User accounts for Am I A Robot application';
COMMENT ON COLUMN users.hashed_password IS 'Bcrypt hashed password - never store plain text';
COMMENT ON COLUMN users.is_active IS 'Whether the user account is active and can log in';