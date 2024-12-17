import pandas as pd
from spotify_profiler.main import EnhancedMusicAnalyzer
from datetime import datetime

def test_small_batch():
    """Test the processor with a small batch of data"""
    # Create a small test DataFrame with proper datetime
    test_data = {
        'master_metadata_track_name': ['Bohemian Rhapsody', 'Imagine', 'Yesterday'],
        'master_metadata_album_artist_name': ['Queen', 'John Lennon', 'The Beatles'],
        'ts': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),  # Convert to datetime
        'ms_played': [1000, 2000, 3000]
    }
    df = pd.DataFrame(test_data)
    
    # Initialize analyzer
    analyzer = EnhancedMusicAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_history(df)
    
    print("\nTest Results:")
    if 'basic_stats' in results:
        stats = results['basic_stats']
        print(f"\nBasic Statistics:")
        print(f"Total tracks: {stats.get('total_tracks', 0)}")
        print(f"Unique artists: {stats.get('unique_artists', 0)}")
        print(f"Total time (hours): {stats.get('total_time_hours', 0):.2f}")
    
    if 'temporal_analysis' in results:
        print("\nTemporal Analysis:")
        temp = results['temporal_analysis']
        print(f"Most active hour: {max(temp.get('hourly_distribution', {}).items(), key=lambda x: x[1])[0]}")
    
if __name__ == "__main__":
    test_small_batch()
