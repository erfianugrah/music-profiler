# Spotify Profiler

A comprehensive tool for analyzing Spotify listening history with enhanced metadata from MusicBrainz. This tool provides detailed insights into listening patterns, genre preferences, and music taste analysis with comparison capabilities.

## Features

- Analyze Spotify listening history data
- Fetch enhanced metadata from MusicBrainz API
- Generate detailed listening pattern insights
- Compare listening habits between users
- Batch processing with resume capability
- Rate-limiting and caching support
- Comprehensive data analysis including:
  - Temporal patterns
  - Genre preferences
  - Artist engagement
  - Listening session analysis
  - User similarity metrics

## Prerequisites

- Python 3.8 or higher
- Local MusicBrainz server running on localhost:5000
- Spotify Developer Account credentials
- SQLite (for caching)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spotify-profiler
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

## Configuration

1. Create a `.env` file in the project root with your Spotify API credentials:
```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
```

2. Ensure your local MusicBrainz server is running on localhost:5000

## Usage

### Basic Analysis

1. Run the analyzer:
```bash
python run_analysis.py
```

2. Select from the following options:
   - Start new analysis
   - Resume previous analysis
   - Reset and start over
   - Compare two analysis results
   - Exit

3. When prompted, provide the directory path containing your Spotify JSON files.

### User Comparison

To compare listening habits between two users:

1. Run the analyzer and select option 4 (Compare two analysis results)
2. Enter names for both users
3. Select the analysis result files to compare
4. View the detailed comparison results

## Output

The tool generates several types of output files:

- Analysis results: `analysis_results/analysis_results_TIMESTAMP.json`
- Comparison results: `comparison_results/comparison_USER1_vs_USER2_TIMESTAMP.json`
- Logs:
  - `logs/music_profiler_debug_TIMESTAMP.log`
  - `logs/music_profiler_error_TIMESTAMP.log`
  - `logs/api_TIMESTAMP.log`

## Analysis Metrics

### Basic Statistics
- Total tracks played
- Unique tracks and artists
- Total listening time
- Average daily listening time
- Track completion rates

### Temporal Analysis
- Hour-by-hour listening patterns
- Daily distribution
- Session analysis
- Peak listening periods

### Genre Analysis
- Genre preferences
- Genre diversity
- Time distribution across genres
- Genre affinity scores

### Artist Analysis
- Artist engagement levels
- Artist preference correlation
- Listening time per artist
- Genre distribution per artist

### Comparison Metrics
- Overall similarity score
- Artist overlap
- Genre compatibility
- Temporal pattern similarity
- Listening habit correlation

## Project Structure

```
spotify-profiler/
├── src/
│   └── spotify_profiler/
│       ├── main.py                 # Core analysis logic
│       ├── api_client.py           # API interaction handling
│       ├── batch_processor.py      # Batch processing logic
│       ├── analysis_insights.py    # Insight generation
│       ├── user_comparison.py      # User comparison logic
│       ├── cache_manager.py        # Cache handling
│       └── utils/
│           ├── json_helper.py      # JSON utilities
│           └── logging_config.py   # Logging configuration
├── tests/
│   └── test_processor.py          # Test suite
├── run_analysis.py                # Main execution script
├── setup.py                       # Package configuration
└── requirements.txt               # Dependencies
```

## Error Handling

The tool includes comprehensive error handling:
- API rate limiting protection
- Connection retry logic
- State preservation for long-running processes
- Detailed logging at multiple levels

## Logging

Logs are organized into three categories:
- Debug logs: Detailed processing information
- Error logs: Error tracking and debugging
- API logs: API interaction monitoring

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Implement new analysis metrics in `analysis_insights.py`
2. Add new API endpoints in `api_client.py`
3. Update comparison logic in `user_comparison.py`
4. Add corresponding tests in `tests/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- Spotify Web API
- MusicBrainz API
