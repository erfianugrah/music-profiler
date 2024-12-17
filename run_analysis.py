import os
import json
from datetime import datetime
from spotify_profiler.main import EnhancedMusicAnalyzer
from spotify_profiler.utils.json_helper import save_results
from pathlib import Path
import pandas as pd
import logging
import sys

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def load_spotify_data(directory_path: str) -> pd.DataFrame:
    """Load Spotify JSON files from directory"""
    all_data = []
    
    # Get all JSON files in directory
    json_files = list(Path(directory_path).glob('*.json'))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {directory_path}")
        
    print(f"Found {len(json_files)} JSON files")
    
    # Load each file
    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
    
    return pd.DataFrame(all_data)

def print_menu():
    """Display the main menu"""
    print("\nSpotify Music Profiler")
    print("=====================")
    print("1. Start new analysis")
    print("2. Resume previous analysis")
    print("3. Reset and start over")
    print("\nPress Ctrl+C at any time to pause the analysis")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Show menu and get user choice
        print_menu()
        choice = input("\nSelect an option (1-3): ").strip()
        
        # Get input directory from user
        print("\nEnter the directory path containing your Spotify JSON files:")
        directory_path = input().strip()
        
        # Load data
        print("\nLoading Spotify data...")
        df = load_spotify_data(directory_path)
        print(f"Loaded {len(df)} entries")
        
        # Initialize analyzer
        print("\nInitializing analyzer...")
        analyzer = EnhancedMusicAnalyzer()
        
        # Handle different options
        if choice == "1":
            analyzer.batch_processor.reset()
        elif choice == "2":
            state_file = input("\nEnter state file path (press Enter for default 'processor_state.pkl'): ").strip()
            if state_file:
                analyzer.batch_processor.state_file = Path(state_file)
            analyzer.batch_processor.resume()
        elif choice == "3":
            analyzer.batch_processor.reset()
        else:
            print("Invalid option selected. Starting new analysis...")
            analyzer.batch_processor.reset()
        
        # Run analysis
        print("\nRunning analysis...")
        try:
            results = analyzer.analyze_history(df)
            
            # Create output directory if it doesn't exist
            output_dir = Path('analysis_results')
            output_dir.mkdir(exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f'analysis_results_{timestamp}.json'
            
            print(f"\nSaving results to {output_file}...")
            save_results(results, output_file)
            
            print("\nAnalysis complete! Results saved successfully.")
            print(f"You can find your results in: {output_file}")
            
            # Print basic stats
            print("\nQuick Summary:")
            if 'basic_stats' in results:
                print(f"Total tracks analyzed: {results['basic_stats']['total_tracks']}")
                print(f"Unique artists: {results['basic_stats']['unique_artists']}")
                print(f"Total listening time: {results['basic_stats']['total_time_hours']:.2f} hours")
            
            if 'artist_analysis' in results and 'top_artists' in results['artist_analysis']:
                top_artists = results['artist_analysis']['top_artists']
                if isinstance(top_artists, dict) and 'by_plays' in top_artists:
                    top_artist = list(top_artists['by_plays'].keys())[0]
                    print(f"Most played artist: {top_artist}")
            
        except KeyboardInterrupt:
            print("\nProcessing paused. You can resume later using option 2.")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nError: {str(e)}")
        print("Check analysis.log for more details")

if __name__ == "__main__":
    main()
