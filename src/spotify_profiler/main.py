from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
from .api_client import MusicAPIClient
from .batch_processor import BatchProcessor
from .analysis_insights import AnalysisInsights
from .logging_config import setup_logging

class EnhancedMusicAnalyzer:
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.api_client = MusicAPIClient()
        self.batch_processor = BatchProcessor(self.api_client)
        self.insights_analyzer = AnalysisInsights()  # Add this line

    def analyze_history(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze listening history with metrics optimized for user comparison"""
        try:
            # Input validation and preprocessing
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            required_columns = [
                'master_metadata_track_name',
                'master_metadata_album_artist_name',
                'ts',
                'ms_played'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            df['ts'] = pd.to_datetime(df['ts'])
            
            # Process data
            self.logger.info("Starting data enrichment...")
            enriched_df = self.batch_processor.process_data(df)
            
            if isinstance(enriched_df, list):
                enriched_df = pd.DataFrame(enriched_df)
            
            if not pd.api.types.is_datetime64_any_dtype(enriched_df['ts']):
                enriched_df['ts'] = pd.to_datetime(enriched_df['ts'])

            # Create analysis results
            results = {
                'user_metrics': self._analyze_user_metrics(enriched_df),
                'tracks': self._analyze_tracks(enriched_df),
                'genres': self._analyze_genres(enriched_df),
                'artists': self._analyze_artists(enriched_df),
                'temporal_patterns': self._analyze_temporal_patterns(enriched_df)
            }
            results['insights'] = self.insights_analyzer.generate_insights(results)
            self.logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise e

    def _analyze_user_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate high-level user metrics for comparison"""
        total_time = df['ms_played'].sum()
        active_days = (df['ts'].max() - df['ts'].min()).days + 1

        return {
            'listening_stats': {
                'total_tracks': len(df),
                'unique_tracks': df['master_metadata_track_name'].nunique(),
                'unique_artists': df['master_metadata_album_artist_name'].nunique(),
                'total_time_hours': total_time / (1000 * 60 * 60),
                'average_daily_time': total_time / (1000 * 60 * 60 * max(1, active_days)),
                'completion_rate': self._calculate_completion_rate(df)
            },
            'activity_period': {
                'first_listen': df['ts'].min().isoformat(),
                'last_listen': df['ts'].max().isoformat(),
                'days_active': active_days
            },
            'engagement_metrics': {
                'skip_rate': (df['skipped'].sum() / len(df) * 100) if 'skipped' in df.columns else None,
                'shuffle_rate': (df['shuffle'].sum() / len(df) * 100) if 'shuffle' in df.columns else None
            }
        }

    def _analyze_tracks(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze tracks with metrics useful for comparison"""
        tracks = {}
        grouped = df.groupby(['master_metadata_track_name', 'master_metadata_album_artist_name'])
        
        for (track_name, artist_name), track_df in grouped:
            track_key = f"{track_name} - {artist_name}"
            api_data = track_df.iloc[0].get('api_data', {}).get('musicbrainz', {})
            
            # Get track duration and ensure it's a number
            try:
                duration_ms = int(api_data.get('length', 0))
            except (TypeError, ValueError):
                duration_ms = 0
            
            # Calculate completion statistics
            completion_rate = None
            if duration_ms > 0:
                try:
                    completion_rate = (float(track_df['ms_played'].mean()) / duration_ms * 100)
                except (TypeError, ValueError):
                    completion_rate = None
            
            # Extract genres
            genres = []
            if api_data and 'artist_data' in api_data:
                artist_tags = api_data['artist_data'].get('tags', [])
                genres = [
                    tag for tag in artist_tags
                    if any(genre_term in tag.lower() 
                          for genre_term in ['metal', 'rock', 'punk', 'jazz', 'pop', 
                                           'electronic', 'hip hop', 'classical', 'blues'])
                ][:5]

            # Ensure all numeric values are properly converted
            try:
                play_count = len(track_df)
                total_time = float(track_df['ms_played'].sum())
                skip_count = int(track_df['skipped'].sum()) if 'skipped' in track_df.columns else 0
            except (TypeError, ValueError):
                play_count = 0
                total_time = 0.0
                skip_count = 0

            tracks[track_key] = {
                'metadata': {
                    'name': track_name,
                    'artist': artist_name,
                    'album': track_df.iloc[0].get('master_metadata_album_album_name'),
                    'release_year': self._extract_release_year(api_data),
                    'duration_ms': duration_ms,
                    'genres': genres
                },
                'listening_data': {
                    'play_count': play_count,
                    'total_time': total_time,
                    'completion_rate': float(completion_rate) if completion_rate is not None else None,
                    'skip_count': skip_count,
                    'first_played': track_df['ts'].min().isoformat(),
                    'last_played': track_df['ts'].max().isoformat()
                }
            }

        return tracks

    def _analyze_genres(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze genre preferences and patterns"""
        genre_stats = defaultdict(lambda: {
            'track_count': 0,
            'total_time': 0,
            'artists': set(),
            'tracks': set(),
            'hourly_distribution': defaultdict(float)
        })
        
        # Collect genre statistics from tracks
        for _, row in df.iterrows():
            api_data = row.get('api_data', {}).get('musicbrainz', {})
            if api_data and 'artist_data' in api_data:
                artist_tags = api_data['artist_data'].get('tags', [])
                genres = [
                    tag for tag in artist_tags
                    if any(genre_term in tag.lower() 
                          for genre_term in ['metal', 'rock', 'punk', 'jazz', 'pop', 
                                           'electronic', 'hip hop', 'classical', 'blues'])
                ][:5]
                
                hour = row['ts'].hour
                for genre in genres:
                    genre_stats[genre]['track_count'] += 1
                    genre_stats[genre]['total_time'] += row['ms_played']
                    genre_stats[genre]['artists'].add(row['master_metadata_album_artist_name'])
                    genre_stats[genre]['tracks'].add(row['master_metadata_track_name'])
                    genre_stats[genre]['hourly_distribution'][hour] += row['ms_played']

        # Calculate genre affinities and patterns
        total_time = sum(stats['total_time'] for stats in genre_stats.values())
        
        return {
            'genre_preferences': {
                genre: {
                    'play_share': (stats['track_count'] / len(df) * 100),
                    'time_share': (stats['total_time'] / total_time * 100) if total_time > 0 else 0,
                    'unique_artists': len(stats['artists']),
                    'unique_tracks': len(stats['tracks']),
                    'peak_hours': self._get_peak_hours(stats['hourly_distribution'])
                }
                for genre, stats in genre_stats.items()
            },
            'top_genres': dict(sorted(
                {genre: stats['track_count'] for genre, stats in genre_stats.items()}.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }

    def _analyze_artists(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze artist preferences and listening patterns"""
        artist_stats = defaultdict(lambda: {
            'track_count': 0,
            'total_time': 0,
            'unique_tracks': set(),
            'genres': set(),
            'completion_rates': [],
            'hourly_distribution': defaultdict(float)
        })
        
        # Collect artist statistics
        for _, row in df.iterrows():
            artist = row['master_metadata_album_artist_name']
            artist_stats[artist]['track_count'] += 1
            artist_stats[artist]['total_time'] += row['ms_played']
            artist_stats[artist]['unique_tracks'].add(row['master_metadata_track_name'])
            
            # Add genres from MusicBrainz data
            api_data = row.get('api_data', {}).get('musicbrainz', {})
            if api_data and 'artist_data' in api_data:
                artist_tags = api_data['artist_data'].get('tags', [])
                genres = [
                    tag for tag in artist_tags
                    if any(genre_term in tag.lower() 
                          for genre_term in ['metal', 'rock', 'punk', 'jazz', 'pop', 
                                           'electronic', 'hip hop', 'classical', 'blues'])
                ]
                artist_stats[artist]['genres'].update(genres)
            
            # Track completion rate
            if 'master_metadata_track_duration_ms' in row and row['master_metadata_track_duration_ms'] > 0:
                completion_rate = (row['ms_played'] / row['master_metadata_track_duration_ms'] * 100)
                artist_stats[artist]['completion_rates'].append(completion_rate)
            
            # Add to hourly distribution
            hour = row['ts'].hour
            artist_stats[artist]['hourly_distribution'][hour] += row['ms_played']

        return {
            'artist_profiles': {
                artist: {
                    'play_count': stats['track_count'],
                    'total_time_minutes': stats['total_time'] / (1000 * 60),
                    'unique_tracks': len(stats['unique_tracks']),
                    'genres': list(stats['genres']),
                    'avg_completion_rate': (
                        np.mean(stats['completion_rates']) 
                        if stats['completion_rates'] else None
                    ),
                    'peak_hours': self._get_peak_hours(stats['hourly_distribution'])
                }
                for artist, stats in artist_stats.items()
            },
            'top_artists': dict(sorted(
                {artist: stats['track_count'] for artist, stats in artist_stats.items()}.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal listening patterns"""
        df_time = df.copy()
        df_time['hour'] = df_time['ts'].dt.hour
        df_time['day'] = df_time['ts'].dt.day_name()
        
        # Calculate distributions
        hourly_dist = df_time.groupby('hour')['ms_played'].sum()
        daily_dist = df_time.groupby('day')['ms_played'].sum()
        
        # Calculate active periods
        total_time = df_time['ms_played'].sum()
        
        return {
            'distributions': {
                'hourly': hourly_dist.to_dict(),
                'daily': daily_dist.to_dict()
            },
            'peak_periods': {
                'hours': self._get_peak_periods(hourly_dist, 3),
                'days': self._get_peak_periods(daily_dist, 3)
            },
            'listening_sessions': self._analyze_listening_sessions(df)
        }

    def _analyze_listening_sessions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze listening session patterns"""
        SESSION_GAP = pd.Timedelta(minutes=30)
        
        # Sort by timestamp
        df_sorted = df.sort_values('ts')
        
        # Find session boundaries
        time_diffs = df_sorted['ts'].diff()
        session_starts = time_diffs > SESSION_GAP
        
        # Calculate session statistics
        session_lengths = []
        tracks_per_session = []
        current_session_length = pd.Timedelta(0)
        current_session_tracks = 0
        
        for i, row in df_sorted.iterrows():
            if i > 0 and session_starts[i]:
                session_lengths.append(float(current_session_length.total_seconds() / 60))
                tracks_per_session.append(current_session_tracks)
                current_session_length = pd.Timedelta(0)
                current_session_tracks = 0
            
            current_session_length += pd.Timedelta(milliseconds=row['ms_played'])
            current_session_tracks += 1
        
        # Add last session
        if current_session_tracks > 0:
            session_lengths.append(float(current_session_length.total_seconds() / 60))
            tracks_per_session.append(current_session_tracks)
        
        return {
            'average_session_minutes': np.mean(session_lengths) if session_lengths else 0,
            'average_tracks_per_session': np.mean(tracks_per_session) if tracks_per_session else 0,
            'total_sessions': len(session_lengths)
        }

    # Helper methods
    def _calculate_completion_rate(self, df: pd.DataFrame) -> float:
        """Calculate overall track completion rate"""
        if 'master_metadata_track_duration_ms' not in df.columns:
            return None
        
        valid_durations = df['master_metadata_track_duration_ms'] > 0
        if not valid_durations.any():
            return None
            
        completion_rates = df[valid_durations]['ms_played'] / df[valid_durations]['master_metadata_track_duration_ms']
        return float(completion_rates.mean() * 100)

    def _extract_release_year(self, api_data: Dict) -> Optional[int]:
        """Extract release year from MusicBrainz data"""
        if api_data and 'release_events' in api_data:
            release_dates = [
                event.get('date', '').split('-')[0] 
                for event in api_data.get('release_events', [])
                if event.get('date')
            ]
            if release_dates:
                try:
                    return min(int(year) for year in release_dates if year.isdigit())
                except ValueError:
                    pass
        return None

    def _get_peak_hours(self, hourly_dist: Dict[int, float]) -> List[Tuple[int, float]]:
        """Get the top 3 hours by listening time"""
        sorted_hours = sorted(
            hourly_dist.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_hours[:3]

    def _get_peak_periods(self, series: pd.Series, n: int = 3) -> List[Tuple[str, float]]:
        """Get the top n periods by activity"""
        sorted_periods = sorted(
            series.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_periods[:n]

    def export_results(self, results: Dict[str, Any], filename: str) -> None:
        """Export analysis results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results exported successfully to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise

if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = EnhancedMusicAnalyzer()
        
        # Load streaming history
        history_df = pd.read_json('StreamingHistory.json')
        
        # Run analysis
        results = analyzer.analyze_history(history_df)
        
        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analyzer.export_results(results, f'analysis_results_{timestamp}.json')
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
