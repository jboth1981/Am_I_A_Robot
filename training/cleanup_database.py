#!/usr/bin/env python3
"""
Database Cleanup Script
Removes sequences shorter than 100 digits from the database
"""

import os
import sys
sys.path.append('/app/src')

from data_extractor import DataExtractor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def cleanup_short_sequences(min_length=100):
    """Remove sequences shorter than min_length from database"""
    
    print(f"=== DATABASE CLEANUP: Removing sequences < {min_length} digits ===")
    print()
    
    # Connect to database
    extractor = DataExtractor()
    
    # Get current stats
    stats = extractor.get_database_stats()
    print(f"Before cleanup:")
    print(f"  Total submissions: {stats['total_submissions']}")
    print()
    
    # Get all sequences
    sequences = extractor.get_all_sequences()
    print(f"Analyzing {len(sequences)} sequences...")
    
    # Find short sequences
    short_sequences = []
    for seq in sequences:
        if len(seq) < min_length:
            short_sequences.append(seq)
    
    print(f"Found {len(short_sequences)} sequences shorter than {min_length} digits:")
    for i, seq in enumerate(short_sequences):
        print(f"  {i+1}. Length {len(seq):2d}: {seq}")
    
    if len(short_sequences) == 0:
        print("No short sequences found! Database is already clean.")
        return
    
    print()
    print(f"Sequences to keep (>= {min_length} digits): {len(sequences) - len(short_sequences)}")
    print()
    
    # Confirm deletion
    confirm = input(f"Delete {len(short_sequences)} short sequences? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete short sequences
    print("Deleting short sequences...")
    
    # Get database connection
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@postgres_local:5432/am_i_a_robot_local')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        deleted_count = 0
        
        for seq in short_sequences:
            # Delete submissions with this sequence
            result = session.execute(
                text("DELETE FROM submissions WHERE sequence = :sequence"),
                {"sequence": seq}
            )
            deleted_count += result.rowcount
            print(f"  Deleted sequence '{seq}' ({len(seq)} digits) - {result.rowcount} submissions")
        
        session.commit()
        print(f"\n✓ Cleanup completed!")
        print(f"  Deleted {deleted_count} submissions")
        print(f"  Removed {len(short_sequences)} short sequences")
        
        # Get updated stats
        extractor_after = DataExtractor()
        stats_after = extractor_after.get_database_stats()
        print(f"\nAfter cleanup:")
        print(f"  Total submissions: {stats_after['total_submissions']}")
        print(f"  Reduction: {stats['total_submissions'] - stats_after['total_submissions']} submissions")
        
    except Exception as e:
        session.rollback()
        print(f"Error during cleanup: {e}")
        raise
    finally:
        session.close()
        extractor.close()

if __name__ == "__main__":
    cleanup_short_sequences(min_length=100)
