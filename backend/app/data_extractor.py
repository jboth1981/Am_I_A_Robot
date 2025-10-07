"""
Data Extraction Script for Binary Transformer Training
Extracts binary sequences from the database for training the transformer model.
"""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Tuple, Optional
import pandas as pd
from datetime import datetime
import json

# Add the app directory to the path so we can import our models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models import User, Submission, Base


class DataExtractor:
    """Extract and prepare training data from the database"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the data extractor
        Args:
            database_url: Database connection string. If None, uses environment variable.
        """
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", 
                "postgresql://robot_user:local_password@localhost:5432/am_i_a_robot_local"
            )
        
        self.engine = create_engine(database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = SessionLocal()
        
        print(f"Connected to database: {database_url.split('@')[-1]}")  # Hide credentials
    
    def get_all_sequences(self) -> List[str]:
        """
        Extract all binary sequences from the database
        Returns:
            List of binary sequences as strings
        """
        submissions = self.db.query(Submission).all()
        sequences = [submission.binary_sequence for submission in submissions]
        
        print(f"Extracted {len(sequences)} sequences from database")
        return sequences
    
    def get_sequences_by_method(self, method: str) -> List[str]:
        """
        Extract sequences by prediction method
        Args:
            method: 'frequency' or 'pattern'
        Returns:
            List of binary sequences
        """
        submissions = self.db.query(Submission).filter(
            Submission.prediction_method == method
        ).all()
        
        sequences = [submission.binary_sequence for submission in submissions]
        print(f"Extracted {len(sequences)} sequences for method '{method}'")
        return sequences
    
    def get_sequences_by_performance(self, min_unpredictability: float = 0.0) -> List[str]:
        """
        Extract sequences based on unpredictability performance
        Args:
            min_unpredictability: Minimum unpredictability rate (0.0 to 1.0)
        Returns:
            List of binary sequences from high-performing users
        """
        # Calculate unpredictability rate: (total - correct) / total
        submissions = self.db.query(Submission).all()
        
        filtered_sequences = []
        for submission in submissions:
            unpredictability = (submission.total_predictions - submission.correct_predictions) / submission.total_predictions
            if unpredictability >= min_unpredictability:
                filtered_sequences.append(submission.binary_sequence)
        
        print(f"Extracted {len(filtered_sequences)} sequences with unpredictability >= {min_unpredictability}")
        return filtered_sequences
    
    def get_human_vs_robot_sequences(self) -> Tuple[List[str], List[str]]:
        """
        Get sequences separated by human vs robot classification
        Returns:
            Tuple of (human_sequences, robot_sequences)
        """
        human_submissions = self.db.query(Submission).filter(
            Submission.is_human_result == True
        ).all()
        
        robot_submissions = self.db.query(Submission).filter(
            Submission.is_human_result == False
        ).all()
        
        human_sequences = [s.binary_sequence for s in human_submissions]
        robot_sequences = [s.binary_sequence for s in robot_submissions]
        
        print(f"Human sequences: {len(human_sequences)}")
        print(f"Robot sequences: {len(robot_sequences)}")
        
        return human_sequences, robot_sequences
    
    def get_database_stats(self) -> dict:
        """Get comprehensive statistics about the database"""
        stats = {}
        
        # Basic counts
        total_users = self.db.query(User).count()
        total_submissions = self.db.query(Submission).count()
        
        stats['total_users'] = total_users
        stats['total_submissions'] = total_submissions
        
        # Submission statistics
        submissions = self.db.query(Submission).all()
        
        if submissions:
            # Method breakdown
            frequency_count = sum(1 for s in submissions if s.prediction_method == 'frequency')
            pattern_count = sum(1 for s in submissions if s.prediction_method == 'pattern')
            
            # Human vs robot classification
            human_count = sum(1 for s in submissions if s.is_human_result)
            robot_count = total_submissions - human_count
            
            # Performance statistics
            accuracies = [s.accuracy_percentage for s in submissions]
            unpredictabilities = [(s.total_predictions - s.correct_predictions) / s.total_predictions * 100 
                                for s in submissions]
            
            stats.update({
                'method_breakdown': {
                    'frequency': frequency_count,
                    'pattern': pattern_count
                },
                'classification_breakdown': {
                    'human': human_count,
                    'robot': robot_count
                },
                'performance_stats': {
                    'avg_accuracy': sum(accuracies) / len(accuracies),
                    'avg_unpredictability': sum(unpredictabilities) / len(unpredictabilities),
                    'min_accuracy': min(accuracies),
                    'max_accuracy': max(accuracies),
                    'min_unpredictability': min(unpredictabilities),
                    'max_unpredictability': max(unpredictabilities)
                }
            })
        
        return stats
    
    def export_training_data(
        self, 
        output_file: str = "training_data.json",
        min_unpredictability: float = 0.0,
        include_metadata: bool = True
    ):
        """
        Export training data to a JSON file
        Args:
            output_file: Output filename
            min_unpredictability: Minimum unpredictability threshold
            include_metadata: Whether to include metadata about each sequence
        """
        submissions = self.db.query(Submission).all()
        
        training_data = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'total_sequences': len(submissions),
                'min_unpredictability_filter': min_unpredictability,
                'database_stats': self.get_database_stats()
            },
            'sequences': []
        }
        
        filtered_count = 0
        for submission in submissions:
            unpredictability = (submission.total_predictions - submission.correct_predictions) / submission.total_predictions
            
            if unpredictability >= min_unpredictability:
                sequence_data = {
                    'sequence': submission.binary_sequence,
                }
                
                if include_metadata:
                    sequence_data.update({
                        'method': submission.prediction_method,
                        'total_predictions': submission.total_predictions,
                        'correct_predictions': submission.correct_predictions,
                        'accuracy': submission.accuracy_percentage,
                        'unpredictability': unpredictability * 100,
                        'is_human': submission.is_human_result,
                        'completed_at': submission.completed_at.isoformat() if submission.completed_at else None
                    })
                
                training_data['sequences'].append(sequence_data)
                filtered_count += 1
        
        training_data['metadata']['filtered_sequences'] = filtered_count
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Exported {filtered_count} sequences to {output_file}")
        return training_data
    
    def create_train_val_split(
        self, 
        sequences: List[str], 
        val_ratio: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Split sequences into training and validation sets
        Args:
            sequences: List of binary sequences
            val_ratio: Ratio of validation data (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        Returns:
            Tuple of (train_sequences, val_sequences)
        """
        import random
        random.seed(random_seed)
        
        # Shuffle sequences
        shuffled = sequences.copy()
        random.shuffle(shuffled)
        
        # Split
        val_size = int(len(shuffled) * val_ratio)
        val_sequences = shuffled[:val_size]
        train_sequences = shuffled[val_size:]
        
        print(f"Train sequences: {len(train_sequences)}")
        print(f"Validation sequences: {len(val_sequences)}")
        
        return train_sequences, val_sequences
    
    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract training data for binary transformer")
    parser.add_argument("--output", "-o", default="training_data.json", help="Output file")
    parser.add_argument("--min-unpredictability", "-u", type=float, default=0.0, 
                       help="Minimum unpredictability threshold (0.0-1.0)")
    parser.add_argument("--method", "-m", choices=['frequency', 'pattern'], 
                       help="Filter by prediction method")
    parser.add_argument("--stats-only", action="store_true", help="Only show database statistics")
    parser.add_argument("--database-url", help="Database connection string")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = DataExtractor(args.database_url)
    
    try:
        if args.stats_only:
            # Show statistics only
            stats = extractor.get_database_stats()
            print("\nDatabase Statistics:")
            print("=" * 50)
            print(json.dumps(stats, indent=2))
        else:
            # Extract and export data
            if args.method:
                sequences = extractor.get_sequences_by_method(args.method)
            else:
                sequences = extractor.get_sequences_by_performance(args.min_unpredictability)
            
            if sequences:
                extractor.export_training_data(
                    args.output, 
                    args.min_unpredictability
                )
                
                # Show some examples
                print(f"\nExample sequences:")
                for i, seq in enumerate(sequences[:5]):
                    print(f"  {i+1}: {seq}")
                if len(sequences) > 5:
                    print(f"  ... and {len(sequences) - 5} more")
            else:
                print("No sequences found matching the criteria.")
    
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
