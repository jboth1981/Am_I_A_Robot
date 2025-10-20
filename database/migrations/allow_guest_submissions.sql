-- Migration to allow guest submissions in existing submissions table
-- Guest users will have user_id = NULL

-- Make user_id nullable to allow guest submissions
ALTER TABLE submissions ALTER COLUMN user_id DROP NOT NULL;

-- Add session_id for guest users to group submissions from same session
ALTER TABLE submissions ADD COLUMN session_id VARCHAR(255);

-- Add indexes for guest submissions
CREATE INDEX idx_submissions_session_id ON submissions(session_id);
CREATE INDEX idx_submissions_guest ON submissions(user_id) WHERE user_id IS NULL;

-- Add comments
COMMENT ON COLUMN submissions.user_id IS 'User ID for logged-in users, NULL for guest users';
COMMENT ON COLUMN submissions.session_id IS 'Browser session ID for guest users to group submissions from same session';
