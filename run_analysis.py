import os
import json
from datetime import datetime
from spotify_profiler.main import EnhancedMusicAnalyzer
from spotify_profiler.user_comparison import UserComparisonAnalyzer
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
    
    json_files = list(Path(directory_path).glob('*.json'))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {directory_path}")
        
    print(f"Found {len(json_files)} JSON files")
    
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
    print("4. Compare two analysis results")
    print("5. Exit")
    print("\nPress Ctrl+C at any time to pause the analysis")

def compare_analyses():
    """Compare two analysis results"""
    print("\nEnter path to first analysis result file:")
    file1 = input().strip()
    print("Enter path to second analysis result file:")
    file2 = input().strip()
    
    try:
        # Load analysis results
        with open(file1, 'r') as f:
            results1 = json.load(f)
        with open(file2, 'r') as f:
            results2 = json.load(f)
            
        # Initialize analyzer and run comparison
        analyzer = UserComparisonAnalyzer()
        comparison_results = analyzer.compare_users(results1, results2)
        
        # Get match probability
        match_probability = comparison_results['match_probability']
        detailed = comparison_results['detailed_comparison']
        
        # Print results
        print("\nComparison Results:")
        print("=" * 50)
        print(f"\nMatch Probability: {match_probability:.2f}%")
        print("\nThis score indicates how likely it is to find this level of")
        print("music taste similarity in the general population.")
        print("\nKey Insights:")
        
        # Print shared music stats
        shared = detailed['shared_music']
        print(f"- Common tracks: {shared['common_track_count']}")
        print(f"- Music overlap: {shared['overlap_percentage']:.1f}%")
        
        # Print top shared tracks if any
        if shared['shared_tracks']:
            print("\nTop Shared Tracks:")
            for track in shared['shared_tracks'][:3]:
                print(f"- {track['track']}")
        
        # Save results
        output_dir = Path('comparison_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'comparison_results_{timestamp}.json'
        
        save_results(comparison_results, output_file)
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        logging.error("Comparison failed", exc_info=True)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            print_menu()
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == "5":
                print("\nExiting...")
                break
                
            if choice == "4":
                compare_analyses()
                continue
            
            print("\nEnter the directory path containing your Spotify JSON files:")
            directory_path = input().strip()
            
            print("\nLoading Spotify data...")
            df = load_spotify_data(directory_path)
            print(f"Loaded {len(df)} entries")
            
            print("\nInitializing analyzer...")
            analyzer = EnhancedMusicAnalyzer()
            
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
            
            print("\nRunning analysis...")
            try:
                results = analyzer.analyze_history(df)
                
                output_dir = Path('analysis_results')
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = output_dir / f'analysis_results_{timestamp}.json'
                
                print(f"\nSaving results to {output_file}...")
                save_results(results, output_file)
                
                print("\nAnalysis complete! Results saved successfully.")
                print(f"You can find your results in: {output_file}")
                
                # Print basic stats
                if 'user_metrics' in results and 'listening_stats' in results['user_metrics']:
                    stats = results['user_metrics']['listening_stats']
                    print("\nQuick Summary:")
                    print(f"Total tracks analyzed: {stats['total_tracks']}")
                    print(f"Unique tracks: {stats['unique_tracks']}")
                    print(f"Unique artists: {stats['unique_artists']}")
                    print(f"Total listening time: {stats['total_time_hours']:.2f} hours")
                
            except KeyboardInterrupt:
                print("\nProcessing paused. You can resume later using option 2.")
                sys.exit(0)
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            print(f"\nError: {str(e)}")
            print("Check analysis.log for more details")

if __name__ == "__main__":
    main()
