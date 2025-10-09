-- Migration to add prediction details to submissions table
-- Run this to add the new columns for storing individual predictions and confidence scores

ALTER TABLE submissions 
ADD COLUMN predictions_json TEXT,
ADD COLUMN confidence_scores_json TEXT,
ADD COLUMN average_confidence FLOAT;

-- Update the prediction_method column to allow 'transformer' as a valid value
-- (This is already handled by the VARCHAR(20) type, but adding a comment for clarity)
COMMENT ON COLUMN submissions.prediction_method IS 'Prediction method used: frequency, pattern, or transformer';

-- Add comments for the new columns
COMMENT ON COLUMN submissions.predictions_json IS 'JSON string containing all individual predictions made during the session';
COMMENT ON COLUMN submissions.confidence_scores_json IS 'JSON string containing confidence scores for each prediction';
COMMENT ON COLUMN submissions.average_confidence IS 'Average confidence score across all predictions in the session';
