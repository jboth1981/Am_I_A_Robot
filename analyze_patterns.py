#!/usr/bin/env python3
"""
Analyze patterns in binary sequences from the database.
Look for n-gram patterns like "01" -> next digit probability.
"""

import psycopg2
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple
import pandas as pd

# Database connection settings
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'am_i_a_robot_local',
    'user': 'robot_user',
    'password': 'local_password'
}

def connect_to_database():
    """Connect to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_sequences():
    """Fetch all binary sequences from the database"""
    conn = connect_to_database()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        query = """
        SELECT 
            s.id,
            u.username,
            s.binary_sequence,
            s.prediction_method,
            s.accuracy_percentage,
            s.completed_at
        FROM submissions s
        JOIN users u ON s.user_id = u.id
        ORDER BY s.completed_at DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        sequences = []
        for row in results:
            sequences.append({
                'id': row[0],
                'username': row[1],
                'sequence': row[2],
                'method': row[3],
                'accuracy': row[4],
                'completed_at': row[5]
            })
        
        cursor.close()
        conn.close()
        
        return sequences
        
    except Exception as e:
        print(f"Error fetching sequences: {e}")
        return []

def analyze_ngram_patterns(sequences: List[dict], n: int = 2) -> Dict:
    """
    Analyze n-gram patterns in sequences.
    For example, n=2 looks at patterns like "01" -> next digit
    """
    
    pattern_counts = defaultdict(Counter)
    total_patterns = Counter()
    
    print(f"\nAnalyzing {n}-gram patterns...")
    
    for seq_data in sequences:
        sequence = seq_data['sequence']
        
        # Extract n-grams and their following digits
        for i in range(len(sequence) - n):
            pattern = sequence[i:i+n]
            next_digit = sequence[i+n]
            
            pattern_counts[pattern][next_digit] += 1
            total_patterns[pattern] += 1
    
    # Calculate probabilities
    pattern_probabilities = {}
    
    for pattern in pattern_counts:
        total = total_patterns[pattern]
        probabilities = {}
        
        for digit in ['0', '1']:
            count = pattern_counts[pattern][digit]
            probabilities[digit] = count / total if total > 0 else 0
        
        pattern_probabilities[pattern] = {
            'probabilities': probabilities,
            'total_occurrences': total,
            'counts': dict(pattern_counts[pattern])
        }
    
    return pattern_probabilities

def analyze_specific_patterns(sequences: List[dict]) -> Dict:
    """Analyze specific interesting patterns"""
    
    patterns_to_check = [
        "01",  # Your example
        "10",  # Opposite of your example
        "00",  # Double zero
        "11",  # Double one
        "101", # Alternating pattern
        "010", # Reverse alternating
        "000", # Triple zero
        "111", # Triple one
    ]
    
    results = {}
    
    for n in [2, 3]:
        results[f"{n}_gram"] = analyze_ngram_patterns(sequences, n)
    
    return results

def print_pattern_analysis(results: Dict):
    """Print formatted analysis results"""
    
    print("\n" + "="*60)
    print("PATTERN ANALYSIS RESULTS")
    print("="*60)
    
    # 2-gram analysis
    print("\n2-GRAM PATTERNS (Previous 2 digits → Next digit probability)")
    print("-" * 50)
    
    patterns_2gram = results.get('2_gram', {})
    
    # Sort by total occurrences for most common patterns
    sorted_patterns = sorted(patterns_2gram.items(), 
                           key=lambda x: x[1]['total_occurrences'], 
                           reverse=True)
    
    for pattern, data in sorted_patterns:
        if data['total_occurrences'] >= 3:  # Only show patterns with enough data
            prob_0 = data['probabilities']['0']
            prob_1 = data['probabilities']['1']
            total = data['total_occurrences']
            
            print(f"Pattern '{pattern}' → Next digit (n={total}):")
            print(f"  Probability of '0': {prob_0:.3f} ({data['counts'].get('0', 0)} times)")
            print(f"  Probability of '1': {prob_1:.3f} ({data['counts'].get('1', 0)} times)")
            
            # Highlight interesting patterns
            if abs(prob_0 - 0.5) > 0.2:  # Strong bias
                bias = "toward '0'" if prob_0 > 0.5 else "toward '1'"
                print(f"  ⚠️  Strong bias {bias}!")
            
            print()
    
    # 3-gram analysis
    print("\n3-GRAM PATTERNS (Previous 3 digits → Next digit probability)")
    print("-" * 50)
    
    patterns_3gram = results.get('3_gram', {})
    sorted_patterns_3 = sorted(patterns_3gram.items(), 
                              key=lambda x: x[1]['total_occurrences'], 
                              reverse=True)
    
    for pattern, data in sorted_patterns_3:
        if data['total_occurrences'] >= 2:  # Lower threshold for 3-grams
            prob_0 = data['probabilities']['0']
            prob_1 = data['probabilities']['1']
            total = data['total_occurrences']
            
            print(f"Pattern '{pattern}' → Next digit (n={total}):")
            print(f"  Probability of '0': {prob_0:.3f} ({data['counts'].get('0', 0)} times)")
            print(f"  Probability of '1': {prob_1:.3f} ({data['counts'].get('1', 0)} times)")
            
            if abs(prob_0 - 0.5) > 0.3:  # Even stronger bias for smaller samples
                bias = "toward '0'" if prob_0 > 0.5 else "toward '1'"
                print(f"  ⚠️  Strong bias {bias}!")
            
            print()

def analyze_sequence_statistics(sequences: List[dict]):
    """Analyze basic statistics about the sequences"""
    
    if not sequences:
        print("No sequences found!")
        return
    
    print(f"\nDATASET OVERVIEW")
    print("-" * 30)
    print(f"Total sequences: {len(sequences)}")
    print(f"Total digits: {sum(len(seq['sequence']) for seq in sequences):,}")
    
    # Analyze digit distribution
    all_digits = ''.join(seq['sequence'] for seq in sequences)
    ones_count = all_digits.count('1')
    zeros_count = all_digits.count('0')
    total_digits = len(all_digits)
    
    print(f"Overall digit distribution:")
    print(f"  0s: {zeros_count:,} ({zeros_count/total_digits*100:.1f}%)")
    print(f"  1s: {ones_count:,} ({ones_count/total_digits*100:.1f}%)")
    
    # Analyze by user
    users = {}
    for seq in sequences:
        username = seq['username']
        if username not in users:
            users[username] = []
        users[username].append(seq)
    
    print(f"\nSequences by user:")
    for username, user_sequences in users.items():
        print(f"  {username}: {len(user_sequences)} sequences")
    
    # Analyze prediction methods
    methods = Counter(seq['method'] for seq in sequences)
    print(f"\nPrediction methods used:")
    for method, count in methods.items():
        print(f"  {method}: {count} sequences")

def save_analysis_results(results: Dict, sequences: List[dict], filename: str = "pattern_analysis.json"):
    """Save analysis results to a JSON file"""
    
    output = {
        'metadata': {
            'total_sequences': len(sequences),
            'total_digits': sum(len(seq['sequence']) for seq in sequences),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        },
        'sequences': sequences,
        'pattern_analysis': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nAnalysis results saved to {filename}")

def main():
    print("Fetching sequences from database...")
    sequences = fetch_sequences()
    
    if not sequences:
        print("No sequences found in database!")
        print("Make sure you have completed some tests as a logged-in user.")
        return
    
    # Basic statistics
    analyze_sequence_statistics(sequences)
    
    # Pattern analysis
    results = analyze_specific_patterns(sequences)
    print_pattern_analysis(results)
    
    # Save results
    save_analysis_results(results, sequences)
    
    # Answer your specific question
    print("\n" + "="*60)
    print("ANSWER TO YOUR SPECIFIC QUESTION:")
    print("="*60)
    
    pattern_01 = results.get('2_gram', {}).get('01')
    if pattern_01:
        prob_0_after_01 = pattern_01['probabilities']['0']
        total_01 = pattern_01['total_occurrences']
        print(f"When the 2 digits prior are '01':")
        print(f"  Probability of next digit being '0': {prob_0_after_01:.3f}")
        print(f"  Probability of next digit being '1': {1-prob_0_after_01:.3f}")
        print(f"  (Based on {total_01} occurrences)")
        
        if prob_0_after_01 > 0.6:
            print("  → Strong tendency to follow '01' with '0'")
        elif prob_0_after_01 < 0.4:
            print("  → Strong tendency to follow '01' with '1'")
        else:
            print("  → No strong bias detected")
    else:
        print("Pattern '01' not found in the dataset (or insufficient occurrences)")

if __name__ == "__main__":
    main()
