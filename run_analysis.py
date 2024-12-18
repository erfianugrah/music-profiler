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
from typing import List, Optional

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
    print("4. Compare two analysis results")
    print("5. Launch visualization dashboard")
    print("6. Exit")
    print("\nPress Ctrl+C at any time to pause the analysis")

def select_analysis_file(prompt: str) -> Optional[Path]:
    """Display a menu of available analysis files and get user selection"""
    files = list_analysis_files()
    
    if not files:
        print("\nNo analysis files found in analysis_results directory.")
        return None
        
    print(f"\n{prompt}")
    print("Available analysis files:")
    for i, file in enumerate(files, 1):
        # Get file creation time
        timestamp = datetime.fromtimestamp(file.stat().st_mtime)
        # Get file size in MB
        size = file.stat().st_size / (1024 * 1024)
        print(f"{i}. {file.name}")
        print(f"   Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {size:.2f} MB")
    
    while True:
        try:
            choice = input("\nEnter number (or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                return None
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(files):
                return files[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def select_file_for_visualization() -> Optional[Path]:
    """Display a menu of available files for visualization"""
    analysis_files = list(Path('analysis_results').glob('analysis_results_*.json'))
    comparison_files = list(Path('comparison_results').glob('comparison_*.json'))
    
    if not analysis_files and not comparison_files:
        print("\nNo analysis or comparison files found.")
        return None
        
    print("\nAvailable files for visualization:")
    print("\nAnalysis Results:")
    for i, file in enumerate(analysis_files, 1):
        # Get file creation time
        timestamp = datetime.fromtimestamp(file.stat().st_mtime)
        # Get file size in MB
        size = file.stat().st_size / (1024 * 1024)
        print(f"{i}. {file.name}")
        print(f"   Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {size:.2f} MB")
    
    print("\nComparison Results:")
    offset = len(analysis_files)
    for i, file in enumerate(comparison_files, offset + 1):
        timestamp = datetime.fromtimestamp(file.stat().st_mtime)
        size = file.stat().st_size / (1024 * 1024)
        print(f"{i}. {file.name}")
        print(f"   Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {size:.2f} MB")
    
    while True:
        try:
            choice = input("\nEnter number to select file (or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                return None
                
            choice_idx = int(choice)
            all_files = analysis_files + comparison_files
            
            if 1 <= choice_idx <= len(all_files):
                return all_files[choice_idx - 1]
            else:
                print(f"Please enter a number between 1 and {len(all_files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def list_analysis_files(directory: str = "analysis_results") -> List[Path]:
    """List all analysis result files in the specified directory"""
    directory = Path(directory)
    if not directory.exists():
        return []
    
    # Get all JSON files that start with "analysis_results"
    files = sorted(directory.glob("analysis_results_*.json"), 
                  key=lambda x: x.stat().st_mtime, 
                  reverse=True)  # Sort by modification time, newest first
    return files

def launch_dashboard_menu():
    """Handle the dashboard launch process with file selection"""

    logger = logging.getLogger(__name__)
    # First show the file selection menu
    selected_file = select_file_for_visualization()
    if not selected_file:
        print("\nNo file selected. Returning to main menu...")
        return

    # Then launch the dashboard with the selected file
    try:
        from spotify_profiler.dashboard import SpotifyDashboard
        print(f"\nLaunching dashboard with file: {selected_file.name}")
        dashboard = SpotifyDashboard()
        
        print("\nDashboard is running at http://localhost:8050")
        print("Press Ctrl+C to stop the dashboard")
        print("\nNote: The terminal will be occupied while the dashboard is running.")
        print("Close the browser window and press Ctrl+C to return to the main menu.")
        
        dashboard.run_server(debug=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped. Returning to main menu...")
    except Exception as e:
        print(f"\nError launching dashboard: {str(e)}")
        logger.error(f"Dashboard launch failed: {e}", exc_info=True)

def compare_analyses():
    """Compare two analysis results with named users"""
    logger = logging.getLogger(__name__)
    
    try:
        # Get first user info
        print("\nEnter name for the first user (e.g., 'John' or 'User 1'):")
        user1_name = input().strip() or "User 1"
        
        # Select first file
        file1 = select_analysis_file(f"Select analysis file for {user1_name}:")
        if not file1:
            print("Comparison cancelled.")
            return
        
        # Get second user info
        print(f"\nEnter name for the second user (e.g., 'Jane' or 'User 2'):")
        user2_name = input().strip() or "User 2"
        
        # Select second file
        file2 = select_analysis_file(f"Select analysis file for {user2_name}:")
        if not file2:
            print("Comparison cancelled.")
            return
        
        # Initialize analyzer and run comparison
        analyzer = UserComparisonAnalyzer()
        comparison_results = analyzer.compare_users(str(file1), str(file2))
        
        # Print comparison results
        print(f"\nComparison Results: {user1_name} vs {user2_name}")
        print("=" * (24 + len(user1_name) + len(user2_name)))
        
        # Overall similarity
        print(f"\nOverall Similarity Score: {comparison_results['overall_similarity']:.2f}%")
        
        # Basic stats comparison
        basic = comparison_results['basic_comparison']
        print("\nBasic Statistics:")
        print(f"Listening time similarity: {basic['listening_time_similarity']:.2f}%")
        comparisons = basic['comparisons']['total_time_hours']
        print(f"- {user1_name}: {comparisons['user1']:.2f} hours")
        print(f"- {user2_name}: {comparisons['user2']:.2f} hours")
        
        print(f"Artist variety similarity: {basic['artist_variety_similarity']:.2f}%")
        artists = basic['comparisons']['unique_artists']
        print(f"- {user1_name}: {artists['user1']} unique artists")
        print(f"- {user2_name}: {artists['user2']} unique artists")
        
        print(f"Track variety similarity: {basic['track_variety_similarity']:.2f}%")
        tracks = basic['comparisons']['unique_tracks']
        print(f"- {user1_name}: {tracks['user1']} unique tracks")
        print(f"- {user2_name}: {tracks['user2']} unique tracks")
        
        # Save detailed results
        output_dir = Path('comparison_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'comparison_{user1_name}_vs_{user2_name}_{timestamp}.json'
        
        # Add user names and file sources to results before saving
        comparison_results['user_names'] = {
            'user1': user1_name,
            'user2': user2_name
        }
        comparison_results['source_files'] = {
            'user1': str(file1),
            'user2': str(file2)
        }
        
        save_results(comparison_results, output_file)
        print(f"\nDetailed comparison results saved to: {output_file}")
        
        # Ask if user wants to view results in dashboard
        view_dashboard = input("\nWould you like to view the comparison in the dashboard? (y/n): ").strip().lower()
        if view_dashboard == 'y':
            launch_dashboard_menu()
        
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}", exc_info=True)
        print(f"Error during comparison: {str(e)}")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            print_menu()
            choice = input("\nSelect an option (1-6): ").strip()
            
            if choice == "6":
                print("\nExiting...")
                break
                
            if choice == "4":
                compare_analyses()
                continue
                
            if choice == "5":
                launch_dashboard_menu()
                continue
            
            print("\nEnter the directory path containing your Spotify JSON files:")
            directory_path = input().strip()
            
            if not Path(directory_path).exists():
                print(f"Error: Directory not found: {directory_path}")
                continue
            
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
                    print(f"Total tracks: {stats['total_tracks']}")
                    print(f"Unique tracks: {stats['unique_tracks']}")
                    print(f"Unique artists: {stats['unique_artists']}")
                    print(f"Total listening time: {stats['total_time_hours']:.2f} hours")
                
                # Ask if user wants to view results in dashboard
                view_dashboard = input("\nWould you like to view the results in the dashboard? (y/n): ").strip().lower()
                if view_dashboard == 'y':
                    launch_dashboard_menu()
                
            except KeyboardInterrupt:
                print("\nProcessing paused. You can resume later using option 2.")
                sys.exit(0)
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            print(f"\nError: {str(e)}")
            print("Check analysis.log for more details")

if __name__ == "__main__":
    main()
