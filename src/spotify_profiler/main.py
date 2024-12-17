from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
from .api_client import MusicAPIClient
from .batch_processor import BatchProcessor
from .logging_config import setup_logging

class EnhancedMusicAnalyzer:
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.api_client = MusicAPIClient()
        self.batch_processor = BatchProcessor(self.api_client)

    def analyze_history(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze listening history"""
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

            # Analyze data
            self.logger.info("Analyzing enriched data...")
            
            # First get the track analysis as it's the base for other analyses
            tracks_analysis = self._analyze_tracks_comprehensive(enriched_df)
            
            results = {
                'tracks': tracks_analysis,
                'genres': self._analyze_genres(tracks_analysis),
                'artists': self._analyze_artists(tracks_analysis),
                'listening_stats': {
                    'overview': {
                        'total_tracks_played': len(enriched_df),
                        'unique_tracks': enriched_df['master_metadata_track_name'].nunique(),
                        'unique_artists': enriched_df['master_metadata_album_artist_name'].nunique(),
                        'total_time_hours': enriched_df['ms_played'].sum() / (1000 * 60 * 60),
                        'time_range': {
                            'start': enriched_df['ts'].min().strftime('%Y-%m-%d'),
                            'end': enriched_df['ts'].max().strftime('%Y-%m-%d'),
                            'days': (enriched_df['ts'].max() - enriched_df['ts'].min()).days
                        }
                    },
                    'temporal': self._analyze_temporal_patterns(enriched_df)
                }
            }
            
            self.logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise e

    def _analyze_tracks_comprehensive(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Comprehensive per-track analysis with focused MusicBrainz enrichment"""
        tracks = {}
        
        # Group by track and artist
        grouped = df.groupby(['master_metadata_track_name', 'master_metadata_album_artist_name'])
        
        for (track_name, artist_name), track_df in grouped:
            track_key = f"{track_name} - {artist_name}"
            api_data = track_df.iloc[0].get('api_data', {})
            
            # Extract essential metadata from MusicBrainz
            mb_data = api_data.get('musicbrainz', {})
            release_year = None
            track_length = None
            
            # Get earliest release year from release events
            if mb_data and 'release_events' in mb_data:
                release_dates = [
                    event.get('date', '').split('-')[0] 
                    for event in mb_data.get('release_events', [])
                    if event.get('date')
                ]
                if release_dates:
                    try:
                        release_year = min(int(year) for year in release_dates if year.isdigit())
                    except ValueError:
                        pass
            
            # Get track length from MusicBrainz
            if mb_data:
                length = mb_data.get('length')
                # MusicBrainz returns length in milliseconds but sometimes as string
                if length:
                    try:
                        track_length = int(length)
                    except (ValueError, TypeError):
                        pass

            # Get relevant genre tags
            genres = []
            if mb_data and 'artist_data' in mb_data and mb_data['artist_data']:
                artist_tags = mb_data['artist_data'].get('tags', [])
                # Filter to most relevant music-related tags
                genres = [
                    tag for tag in artist_tags
                    if any(genre_term in tag.lower() 
                          for genre_term in ['metal', 'rock', 'punk', 'jazz', 'pop', 
                                           'electronic', 'hip hop', 'classical', 'blues'])
                ][:5]  # Limit to top 5 most relevant tags

            # Calculate play statistics
            play_count = len(track_df)
            total_ms_played = float(track_df['ms_played'].sum())
            avg_play_duration = float(track_df['ms_played'].mean())
            
            # Calculate completion statistics
            completion_stats = {
                'completion_rate': None,
                'full_plays': 0,
                'partial_plays': 0,
                'short_plays': 0
            }
            
            if track_length and track_length > 0:
                completion_rates = track_df['ms_played'] / track_length * 100
                completion_stats.update({
                    'completion_rate': float(avg_play_duration / track_length * 100),
                    'full_plays': int((completion_rates >= 85).sum()),  # 85% or more considered full play
                    'partial_plays': int(((completion_rates < 85) & (completion_rates >= 25)).sum()),
                    'short_plays': int((completion_rates < 25).sum())
                })

            # Calculate skip patterns
            skip_patterns = {
                'early_skips': 0,  # Skipped in first 25% of track
                'mid_skips': 0,    # Skipped between 25-75% of track
                'late_skips': 0    # Skipped in last 25% of track
            }
            
            if track_length:
                for _, row in track_df.iterrows():
                    if row.get('skipped', False):
                        play_position = (row['ms_played'] / track_length) * 100
                        if play_position < 25:
                            skip_patterns['early_skips'] += 1
                        elif play_position < 75:
                            skip_patterns['mid_skips'] += 1
                        else:
                            skip_patterns['late_skips'] += 1

            tracks[track_key] = {
                'metadata': {
                    'name': track_name,
                    'artist': artist_name,
                    'album': track_df.iloc[0].get('master_metadata_album_album_name'),
                    'release_year': release_year,
                    'spotify_uri': track_df.iloc[0].get('spotify_track_uri'),
                    'duration_ms': track_length
                },
                'play_stats': {
                    'play_count': play_count,
                    'total_ms_played': total_ms_played,
                    'avg_play_duration': avg_play_duration,
                    'first_played': track_df['ts'].min().isoformat(),
                    'last_played': track_df['ts'].max().isoformat(),
                    'completion_stats': completion_stats,
                    'skip_patterns': skip_patterns
                },
                'playback_behavior': {
                    'shuffle_plays': int(track_df['shuffle'].sum()) if 'shuffle' in track_df.columns else 0,
                    'skipped_count': int(track_df['skipped'].sum()) if 'skipped' in track_df.columns else 0,
                    'start_reasons': track_df['reason_start'].value_counts().to_dict() if 'reason_start' in track_df.columns else {},
                    'end_reasons': track_df['reason_end'].value_counts().to_dict() if 'reason_end' in track_df.columns else {},
                    'platform_stats': track_df['platform'].value_counts().to_dict() if 'platform' in track_df.columns else {}
                },
                'genres': genres,
                'hourly_distribution': track_df.groupby(track_df['ts'].dt.hour)['ms_played'].sum().to_dict()
            }
        
        return tracks

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal listening patterns"""
        try:
            df_time = df.copy()
            df_time['hour'] = df_time['ts'].dt.hour
            df_time['day'] = df_time['ts'].dt.day_name()
            df_time['month'] = df_time['ts'].dt.month
            df_time['year'] = df_time['ts'].dt.year

            # Aggregate listening time by different time periods
            analysis = {
                'distributions': {
                    'hourly': df_time.groupby('hour')['ms_played'].sum().to_dict(),
                    'daily': df_time.groupby('day')['ms_played'].sum().to_dict(),
                    'monthly': df_time.groupby('month')['ms_played'].sum().to_dict(),
                    'yearly': df_time.groupby('year')['ms_played'].sum().to_dict()
                },
                'peak_periods': {
                    'hours': sorted(
                        df_time.groupby('hour')['ms_played'].sum().items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3],
                    'days': sorted(
                        df_time.groupby('day')['ms_played'].sum().items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3],
                }
            }

            # Add time of day analysis
            time_of_day = {
                'morning': (6, 12),
                'afternoon': (12, 18),
                'evening': (18, 22),
                'night': (22, 6)
            }

            time_of_day_stats = {}
            for period, (start, end) in time_of_day.items():
                if start < end:
                    mask = (df_time['hour'] >= start) & (df_time['hour'] < end)
                else:  # Handle night crossing midnight
                    mask = (df_time['hour'] >= start) | (df_time['hour'] < end)
                    
                period_df = df_time[mask]
                time_of_day_stats[period] = {
                    'total_plays': len(period_df),
                    'total_time': period_df['ms_played'].sum(),
                    'unique_tracks': period_df['master_metadata_track_name'].nunique(),
                    'avg_completion_rate': (
                        (period_df['ms_played'] / period_df['master_metadata_track_duration_ms']).mean() * 100
                        if 'master_metadata_track_duration_ms' in period_df.columns
                        else None
                    )
                }

            analysis['time_of_day'] = time_of_day_stats
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {e}")
            return {}

    def _analyze_genres(self, tracks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced genre analysis with advanced metrics"""
        genre_stats = {
            'play_counts': defaultdict(int),
            'listening_time': defaultdict(float),
            'completion_rates': defaultdict(list),
            'skip_patterns': defaultdict(lambda: {'early': 0, 'mid': 0, 'late': 0}),
            'decades': defaultdict(set),
            'hourly_distribution': defaultdict(lambda: defaultdict(float))
        }
        
        # Analyze each track's contribution
        for track_data in tracks.values():
            play_count = track_data['play_stats']['play_count']
            listening_time = track_data['play_stats']['total_ms_played']
            completion_stats = track_data['play_stats'].get('completion_stats', {})
            release_year = track_data['metadata'].get('release_year')
            skip_patterns = track_data['play_stats'].get('skip_patterns', {})
            
            for genre in track_data['genres']:
                # Basic stats
                genre_stats['play_counts'][genre] += play_count
                genre_stats['listening_time'][genre] += listening_time
                
                # Completion rates
                if completion_stats.get('completion_rate') is not None:
                    genre_stats['completion_rates'][genre].append(completion_stats['completion_rate'])
                
                # Skip patterns
                for skip_type in ['early_skips', 'mid_skips', 'late_skips']:
                    genre_stats['skip_patterns'][genre][skip_type.replace('_skips', '')] += skip_patterns.get(skip_type, 0)
                
                # Track decades
                if release_year:
                    genre_stats['decades'][genre].add((release_year // 10) * 10)
                
                # Hourly distribution
                for hour, time in track_data.get('hourly_distribution', {}).items():
                    genre_stats['hourly_distribution'][genre][hour] += time

        # Calculate aggregated statistics
        total_plays = sum(genre_stats['play_counts'].values())
        total_time = sum(genre_stats['listening_time'].values())
        
        # Create sorted lists first, then convert to dictionaries
        by_plays = sorted(
            [(k, v) for k, v in genre_stats['play_counts'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        by_time = sorted(
            [(k, v/1000/60) for k, v in genre_stats['listening_time'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate genre relevance scores
        genre_relevance = {}
        max_plays = max(genre_stats['play_counts'].values()) if genre_stats['play_counts'] else 1
        max_time = max(genre_stats['listening_time'].values()) if genre_stats['listening_time'] else 1
        
        for genre in genre_stats['play_counts'].keys():
            play_score = genre_stats['play_counts'][genre] / max_plays
            time_score = genre_stats['listening_time'][genre] / max_time
            genre_relevance[genre] = (play_score + time_score) / 2 * 100
        
        return {
            'top_genres': {
                'by_plays': dict(by_plays),
                'by_time': dict(by_time),
                'by_relevance': dict(sorted(
                    genre_relevance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            'genre_engagement': {
                genre: {
                    'play_share': (plays / total_plays * 100) if total_plays > 0 else 0,
                    'time_share': (genre_stats['listening_time'][genre] / total_time * 100) if total_time > 0 else 0,
                    'relevance_score': genre_relevance[genre],
                    'avg_completion': (
                        float(np.mean(genre_stats['completion_rates'][genre]))
                        if genre_stats['completion_rates'][genre] else None
                    ),
                    'skip_patterns': genre_stats['skip_patterns'][genre],
                    'decades': sorted(genre_stats['decades'][genre]),
                    'peak_hours': self._get_peak_hours(genre_stats['hourly_distribution'][genre])
                }
                for genre, plays in genre_stats['play_counts'].items()
            }
        }

    def _get_peak_hours(self, hourly_dist: Dict[int, float]) -> List[Tuple[int, float]]:
        """Get the top 3 hours for a genre by listening time"""
        sorted_hours = sorted(
            hourly_dist.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_hours[:3]

    def _calculate_temporal_diversity(self, hourly_distribution: Dict[str, Dict[int, float]]) -> float:
        """Calculate how evenly genres are spread across different times of day"""
        total_diversity = 0
        for genre, hours in hourly_distribution.items():
            if not hours:
                continue
            total_time = sum(hours.values())
            if total_time == 0:
                continue
            # Calculate Shannon entropy for time distribution
            proportions = np.array(list(hours.values())) / total_time
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            total_diversity += entropy
        return total_diversity / len(hourly_distribution) if hourly_distribution else 0

    def _analyze_artists(self, tracks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced artist analysis with detailed insights"""
        artist_stats = defaultdict(lambda: {
            'tracks': set(),
            'albums': set(),
            'genres': set(),
            'play_stats': {
                'play_count': 0,
                'total_time': 0.0,
                'skipped_count': 0,
                'full_plays': 0
            },
            'completion_rates': [],
            'decades': set(),
            'hourly_distribution': defaultdict(float),
            'skip_patterns': {'early': 0, 'mid': 0, 'late': 0},
            'playback_platforms': defaultdict(int),
            'genre_affinity': defaultdict(float)
        })
        
        # Collect artist statistics
        for track_data in tracks.values():
            self._collect_artist_stats(track_data, artist_stats)
        
        # Calculate artist insights
        artist_insights = {
            'top_artists': {
                'by_plays': dict(sorted(
                    {artist: stats['play_stats']['play_count'] 
                     for artist, stats in artist_stats.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'by_time': dict(sorted(
                    {artist: stats['play_stats']['total_time'] / (1000 * 60)
                     for artist, stats in artist_stats.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'by_completion': dict(sorted(
                    {artist: np.mean(stats['completion_rates'])
                     for artist, stats in artist_stats.items()
                     if stats['completion_rates']}.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            'artist_details': {
                artist: self._generate_artist_insights(artist, stats, artist_stats)
                for artist, stats in artist_stats.items()
            },
            'cross_genre_insights': self._analyze_cross_genre_patterns(artist_stats),
            'temporal_patterns': self._analyze_artist_temporal_patterns(artist_stats),
            'platform_insights': self._analyze_platform_patterns(artist_stats)
        }
        
        return artist_insights

    def _collect_artist_stats(self, track_data: Dict, artist_stats: Dict) -> None:
        """Collect detailed statistics for an artist from track data"""
        artist = track_data['metadata']['artist']
        stats = artist_stats[artist]
        
        # Update basic stats
        stats['tracks'].add(track_data['metadata']['name'])
        stats['albums'].add(track_data['metadata']['album'])
        stats['genres'].update(track_data['genres'])
        
        # Update play statistics
        play_stats = track_data['play_stats']
        stats['play_stats']['play_count'] += play_stats['play_count']
        stats['play_stats']['total_time'] += play_stats['total_ms_played']
        stats['play_stats']['full_plays'] += play_stats['completion_stats']['full_plays']
        
        # Update skip patterns
        skip_patterns = play_stats['skip_patterns']
        stats['skip_patterns']['early'] += skip_patterns['early_skips']
        stats['skip_patterns']['mid'] += skip_patterns['mid_skips']
        stats['skip_patterns']['late'] += skip_patterns['late_skips']
        
        # Track completion rates
        if play_stats['completion_stats']['completion_rate'] is not None:
            stats['completion_rates'].append(play_stats['completion_stats']['completion_rate'])
        
        # Update platform stats
        for platform, count in track_data['playback_behavior'].get('platform_stats', {}).items():
            stats['playback_platforms'][platform] += count
        
        # Update hourly distribution
        for hour, time in track_data.get('hourly_distribution', {}).items():
            stats['hourly_distribution'][hour] += time
        
        # Update genre affinity
        for genre in track_data['genres']:
            stats['genre_affinity'][genre] += play_stats['total_ms_played']
        
        # Track decades
        if track_data['metadata'].get('release_year'):
            stats['decades'].add((track_data['metadata']['release_year'] // 10) * 10)

    def _generate_artist_insights(self, artist: str, stats: Dict, all_artist_stats: Dict) -> Dict:
        """Generate detailed insights for a specific artist"""
        total_artist_time = sum(s['play_stats']['total_time'] for s in all_artist_stats.values())
        
        # Calculate peak listening hours
        peak_hours = sorted(
            stats['hourly_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Calculate genre preferences
        genre_preferences = {}
        if stats['play_stats']['total_time'] > 0:
            genre_preferences = {
                genre: time / stats['play_stats']['total_time'] * 100
                for genre, time in stats['genre_affinity'].items()
            }
        
        return {
            'engagement': {
                'total_plays': stats['play_stats']['play_count'],
                'total_time_minutes': stats['play_stats']['total_time'] / (1000 * 60),
                'time_share': (stats['play_stats']['total_time'] / total_artist_time * 100) 
                             if total_artist_time > 0 else 0,
                'unique_tracks': len(stats['tracks']),
                'unique_albums': len(stats['albums']),
                'avg_completion_rate': np.mean(stats['completion_rates']) if stats['completion_rates'] else None,
                'full_plays': stats['play_stats']['full_plays']
            },
            'listening_patterns': {
                'peak_hours': [
                    {'hour': hour, 'time_minutes': time / (1000 * 60)} 
                    for hour, time in peak_hours
                ],
                'skip_patterns': stats['skip_patterns'],
                'platform_distribution': dict(stats['playback_platforms'])
            },
            'content_diversity': {
                'genres': list(stats['genres']),
                'genre_preferences': genre_preferences,
                'decades': sorted(stats['decades']),
                'genre_count': len(stats['genres']),
                'decade_span': max(stats['decades']) - min(stats['decades']) if stats['decades'] else 0
            }
        }

    def _analyze_artist_temporal_patterns(self, artist_stats: Dict) -> Dict:
        """Analyze temporal patterns across artists"""
        time_patterns = {
            'hourly_trends': defaultdict(list),
            'decade_trends': defaultdict(int),
            'era_transitions': []
        }
        
        for artist, stats in artist_stats.items():
            # Hourly distribution
            for hour, time in stats['hourly_distribution'].items():
                time_patterns['hourly_trends'][hour].append({
                    'artist': artist,
                    'time': time
                })
            
            # Decade distribution
            for decade in stats['decades']:
                time_patterns['decade_trends'][decade] += stats['play_stats']['play_count']
            
            # Track era transitions in listening
            if len(stats['decades']) > 1:
                time_patterns['era_transitions'].append({
                    'artist': artist,
                    'decades': sorted(stats['decades']),
                    'span': max(stats['decades']) - min(stats['decades'])
                })
        
        return time_patterns

    def _analyze_platform_patterns(self, artist_stats: Dict) -> Dict:
        """Analyze platform usage patterns"""
        platform_insights = {
            'platform_preferences': defaultdict(lambda: {
                'total_time': 0,
                'artist_count': 0,
                'top_artists': []
            })
        }
        
        for artist, stats in artist_stats.items():
            for platform, count in stats['playback_platforms'].items():
                platform_insights['platform_preferences'][platform]['total_time'] += stats['play_stats']['total_time']
                platform_insights['platform_preferences'][platform]['artist_count'] += 1
                platform_insights['platform_preferences'][platform]['top_artists'].append({
                    'artist': artist,
                    'play_count': count
                })
        
        # Sort top artists for each platform
        for platform_data in platform_insights['platform_preferences'].values():
            platform_data['top_artists'] = sorted(
                platform_data['top_artists'],
                key=lambda x: x['play_count'],
                reverse=True
            )[:5]
        
        return platform_insights

    def _analyze_cross_genre_patterns(self, artist_stats: Dict) -> Dict:
        """Analyze patterns in genre relationships across artists"""
        genre_pairs = defaultdict(float)
        genre_total_time = defaultdict(float)
        
        # Collect genre co-occurrence data
        for stats in artist_stats.values():
            for genre1 in stats['genres']:
                genre_total_time[genre1] += stats['play_stats']['total_time']
                for genre2 in stats['genres']:
                    if genre1 < genre2:  # Avoid counting pairs twice
                        genre_pairs[(genre1, genre2)] += stats['play_stats']['total_time']
        
        # Calculate similarity scores
        genre_similarities = {}
        for (genre1, genre2), pair_time in genre_pairs.items():
            similarity = pair_time / (genre_total_time[genre1] + genre_total_time[genre2] - pair_time)
            genre_similarities[f"{genre1} - {genre2}"] = similarity
        
        return {
            'genre_similarities': dict(sorted(
                genre_similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'genre_correlations': self._calculate_genre_correlations(artist_stats)
        }

    def _calculate_genre_correlations(self, artist_stats: Dict) -> Dict:
        """Calculate correlations between genres based on listening patterns"""
        genre_vectors = defaultdict(list)
        all_genres = set()
        
        # Collect all genres
        for stats in artist_stats.values():
            all_genres.update(stats['genres'])
        
        # Create listening vectors for each genre
        for hour in range(24):
            for genre in all_genres:
                total_time = sum(
                    stats['hourly_distribution'].get(hour, 0)
                    for stats in artist_stats.values()
                    if genre in stats['genres']
                )
                genre_vectors[genre].append(total_time)
        
        # Calculate correlations
        correlations = {}
        genres = list(all_genres)
        for i, genre1 in enumerate(genres):
            for genre2 in genres[i+1:]:
                vector1 = np.array(genre_vectors[genre1])
                vector2 = np.array(genre_vectors[genre2])
                
                if np.sum(vector1) > 0 and np.sum(vector2) > 0:
                    correlation = np.corrcoef(vector1, vector2)[0, 1]
                    correlations[f"{genre1} - {genre2}"] = float(correlation)
        
        return dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10])

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy for a series"""
        try:
            proportions = series / series.sum()
            proportions = proportions[proportions > 0]  # Remove zero values
            return -(proportions * np.log2(proportions)).sum()
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {e}")
            return 0.0

    def _get_top_items(self, items: List[str], limit: int = 20) -> Dict[str, int]:
        """Get top items with their counts"""
        try:
            counter = Counter(items)
            return dict(counter.most_common(limit))
        except Exception as e:
            self.logger.error(f"Error getting top items: {e}")
            return {}

    def _find_common_genre_combinations(self, genre_sets: pd.Series, min_support: float = 0.01) -> Dict[str, int]:
        """Find frequently occurring genre combinations"""
        from itertools import combinations
        
        # Count all 2-genre combinations
        combo_counts = defaultdict(int)
        total_sessions = len(genre_sets)
        
        for genres in genre_sets:
            if len(genres) >= 2:
                for combo in combinations(sorted(genres), 2):
                    combo_counts[combo] += 1
        
        # Filter by minimum support
        min_count = total_sessions * min_support
        return {
            ' + '.join(combo): count
            for combo, count in combo_counts.items()
            if count >= min_count
        }

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
        analyzer.export_results(results, 'analysis_results.json')
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
