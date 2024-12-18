from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import json
from pathlib import Path
import logging
from .comparison_insights import ComparisonInsightsAnalyzer

class UserComparisonAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.insights_analyzer = ComparisonInsightsAnalyzer()  # Add this line

    def compare_users(self, results1_path: str, results2_path: str, 
                     user1_name: str = "User 1", 
                     user2_name: str = "User 2") -> Dict[str, Any]:
        """Compare two users' listening histories from analysis result files"""
        try:
            # Load analysis results
            with open(results1_path, 'r', encoding='utf-8') as f:
                results1 = json.load(f)
            with open(results2_path, 'r', encoding='utf-8') as f:
                results2 = json.load(f)
            
            # Calculate various similarity metrics
            comparison_results = {
                'basic_comparison': self._compare_basic_stats(results1, results2),
                'artist_similarity': self._compare_artists(results1, results2),
                'temporal_similarity': self._compare_temporal_patterns(results1, results2),
                'genre_similarity': self._compare_genres(results1, results2),
                'listening_habits': self._compare_listening_habits(results1, results2),
                'overall_similarity': self._calculate_overall_similarity(results1, results2)
            }
            
            # Add insights analysis
            comparison_results['insights'] = self.insights_analyzer.analyze_comparison(
                comparison_results,
                user1_name,
                user2_name
            )
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing users: {e}")
            raise

    def _compare_basic_stats(self, results1: Dict, results2: Dict) -> Dict[str, Any]:
        """Compare basic listening statistics from analysis results"""
        stats1 = results1['user_metrics']['listening_stats']
        stats2 = results2['user_metrics']['listening_stats']

        def calculate_percentile(value1: float, value2: float) -> float:
            return (min(value1, value2) / max(value1, value2)) * 100 if max(value1, value2) > 0 else 0

        return {
            'listening_time_similarity': calculate_percentile(
                stats1['total_time_hours'],
                stats2['total_time_hours']
            ),
            'artist_variety_similarity': calculate_percentile(
                stats1['unique_artists'],
                stats2['unique_artists']
            ),
            'track_variety_similarity': calculate_percentile(
                stats1['unique_tracks'],
                stats2['unique_tracks']
            ),
            'comparisons': {
                'total_time_hours': {
                    'user1': stats1['total_time_hours'],
                    'user2': stats2['total_time_hours']
                },
                'unique_artists': {
                    'user1': stats1['unique_artists'],
                    'user2': stats2['unique_artists']
                },
                'unique_tracks': {
                    'user1': stats1['unique_tracks'],
                    'user2': stats2['unique_tracks']
                }
            }
        }

    def _compare_artists(self, results1: Dict, results2: Dict) -> Dict[str, Any]:
        """Compare artist preferences between users"""
        artists1 = results1['artists']['top_artists']
        artists2 = results2['artists']['top_artists']

        # Find common artists
        common_artists = set(artists1.keys()) & set(artists2.keys())
        
        # Calculate correlation for common artists
        if common_artists:
            artist_plays1 = [artists1[artist] for artist in common_artists]
            artist_plays2 = [artists2[artist] for artist in common_artists]
            correlation = pearsonr(artist_plays1, artist_plays2)[0]
        else:
            correlation = 0

        # Get artist profiles for detailed comparison
        profiles1 = results1['artists']['artist_profiles']
        profiles2 = results2['artists']['artist_profiles']

        return {
            'common_artists_count': len(common_artists),
            'common_artists_percentage': len(common_artists) / max(len(artists1), len(artists2)) * 100,
            'artist_preference_correlation': correlation if not np.isnan(correlation) else 0,
            'top_shared_artists': self._get_top_shared_artists(profiles1, profiles2, common_artists),
            'unique_preferences': {
                'user1_unique': list(set(artists1.keys()) - common_artists),
                'user2_unique': list(set(artists2.keys()) - common_artists)
            }
        }

    def _compare_temporal_patterns(self, results1: Dict, results2: Dict) -> Dict[str, float]:
        """Compare temporal listening patterns"""
        dist1 = results1['temporal_patterns']['distributions']
        dist2 = results2['temporal_patterns']['distributions']
        
        similarities = {}
        for timeframe in ['hourly', 'daily']:
            # Create vectors for comparison
            all_periods = set(str(k) for k in dist1[timeframe].keys()) | set(str(k) for k in dist2[timeframe].keys())
            vector1 = [dist1[timeframe].get(str(period), 0) for period in all_periods]
            vector2 = [dist2[timeframe].get(str(period), 0) for period in all_periods]
            
            # Normalize vectors
            sum1, sum2 = sum(vector1), sum(vector2)
            vector1_norm = np.array(vector1) / sum1 if sum1 > 0 else np.zeros_like(vector1)
            vector2_norm = np.array(vector2) / sum2 if sum2 > 0 else np.zeros_like(vector2)
            
            # Calculate cosine similarity
            if sum1 > 0 and sum2 > 0:
                similarity = 1 - cosine(vector1_norm, vector2_norm)
                similarities[f'{timeframe}_similarity'] = similarity if not np.isnan(similarity) else 0
            else:
                similarities[f'{timeframe}_similarity'] = 0
        
        return similarities

    def _compare_genres(self, results1: Dict, results2: Dict) -> Dict[str, Any]:
        """Compare genre preferences between users"""
        genres1 = results1['genres']['top_genres']
        genres2 = results2['genres']['top_genres']
        
        common_genres = set(genres1.keys()) & set(genres2.keys())
        
        # Calculate genre similarity using cosine similarity
        all_genres = list(set(genres1.keys()) | set(genres2.keys()))
        vector1 = [genres1.get(genre, 0) for genre in all_genres]
        vector2 = [genres2.get(genre, 0) for genre in all_genres]
        
        # Normalize vectors
        sum1, sum2 = sum(vector1), sum(vector2)
        vector1_norm = np.array(vector1) / sum1 if sum1 > 0 else np.zeros_like(vector1)
        vector2_norm = np.array(vector2) / sum2 if sum2 > 0 else np.zeros_like(vector2)
        
        similarity = 1 - cosine(vector1_norm, vector2_norm) if sum1 > 0 and sum2 > 0 else 0
        
        return {
            'genre_similarity_score': similarity if not np.isnan(similarity) else 0,
            'common_genres_count': len(common_genres),
            'top_shared_genres': sorted(
                [(genre, genres1.get(genre, 0), genres2.get(genre, 0)) for genre in common_genres],
                key=lambda x: x[1] + x[2],
                reverse=True
            )[:10],
            'unique_preferences': {
                'user1_unique': list(set(genres1.keys()) - common_genres),
                'user2_unique': list(set(genres2.keys()) - common_genres)
            }
        }

    def _compare_listening_habits(self, results1: Dict, results2: Dict) -> Dict[str, Any]:
        """Compare listening behaviors and habits"""
        metrics1 = results1['user_metrics']['engagement_metrics']
        metrics2 = results2['user_metrics']['engagement_metrics']
        
        sessions1 = results1['temporal_patterns']['listening_sessions']
        sessions2 = results2['temporal_patterns']['listening_sessions']

        def calculate_similarity(value1: float, value2: float) -> float:
            return (min(value1, value2) / max(value1, value2)) * 100 if max(value1, value2) > 0 else 100

        return {
            'engagement_similarity': {
                'skip_rate': calculate_similarity(
                    metrics1['skip_rate'],
                    metrics2['skip_rate']
                ),
                'shuffle_rate': calculate_similarity(
                    metrics1['shuffle_rate'],
                    metrics2['shuffle_rate']
                )
            },
            'session_patterns': {
                'avg_session_similarity': calculate_similarity(
                    sessions1['average_session_minutes'],
                    sessions2['average_session_minutes']
                ),
                'tracks_per_session_similarity': calculate_similarity(
                    sessions1['average_tracks_per_session'],
                    sessions2['average_tracks_per_session']
                )
            }
        }

    def _calculate_overall_similarity(self, results1: Dict, results2: Dict) -> float:
        """Calculate overall similarity score between users"""
        genre_sim = self._compare_genres(results1, results2)['genre_similarity_score']
        artist_sim = self._compare_artists(results1, results2)['artist_preference_correlation']
        temporal_sim = np.mean(list(self._compare_temporal_patterns(results1, results2).values()))
        
        # Weighted combination
        weights = {
            'genre': 0.35,
            'artist': 0.35,
            'temporal': 0.30
        }
        
        overall_score = (
            genre_sim * weights['genre'] +
            artist_sim * weights['artist'] +
            temporal_sim * weights['temporal']
        )
        
        return max(0, min(100, overall_score * 100))

    def _get_top_shared_artists(self, profiles1: Dict, profiles2: Dict, common_artists: set) -> List[Dict[str, Any]]:
        """Get detailed comparison of shared artists"""
        shared_artists = []
        
        for artist in common_artists:
            if artist in profiles1 and artist in profiles2:
                shared_artists.append({
                    'artist': artist,
                    'user1_plays': profiles1[artist]['play_count'],
                    'user2_plays': profiles2[artist]['play_count'],
                    'user1_time': profiles1[artist]['total_time_minutes'],
                    'user2_time': profiles2[artist]['total_time_minutes'],
                    'genres': list(set(profiles1[artist]['genres']) & set(profiles2[artist]['genres']))
                })
        
        return sorted(shared_artists, key=lambda x: x['user1_plays'] + x['user2_plays'], reverse=True)[:10]
