from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import json
from pathlib import Path

class UserComparisonAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_users(self, user1_dir: str, user2_dir: str) -> Dict[str, Any]:
        """Compare two users' listening histories and calculate similarity metrics"""
        try:
            # Load data for both users
            user1_data = self._load_user_data(user1_dir)
            user2_data = self._load_user_data(user2_dir)
            
            # Calculate various similarity metrics
            return {
                'basic_comparison': self._compare_basic_stats(user1_data, user2_data),
                'artist_similarity': self._compare_artists(user1_data, user2_data),
                'temporal_similarity': self._compare_temporal_patterns(user1_data, user2_data),
                'genre_similarity': self._compare_genres(user1_data, user2_data),
                'platform_similarity': self._compare_platforms(user1_data, user2_data),
                'listening_behavior': self._compare_listening_behavior(user1_data, user2_data),
                'overall_similarity': self._calculate_overall_similarity(user1_data, user2_data)
            }
        except Exception as e:
            self.logger.error(f"Error comparing users: {e}")
            raise

    def _load_user_data(self, directory: str) -> Dict[str, Any]:
        """Load and process user's listening history"""
        try:
            # Get all JSON files in directory
            json_files = list(Path(directory).glob('*.json'))
            
            if not json_files:
                raise ValueError(f"No JSON files found in {directory}")
            
            all_data = []
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['ts'] = pd.to_datetime(df['ts'])
            
            return {
                'raw_data': df,
                'artist_counts': df['master_metadata_album_artist_name'].value_counts(),
                'track_counts': df['master_metadata_track_name'].value_counts(),
                'platform_counts': df['platform'].value_counts(),
                'total_time': df['ms_played'].sum(),
                'temporal_distribution': self._get_temporal_distribution(df)
            }
        except Exception as e:
            self.logger.error(f"Error loading user data from {directory}: {e}")
            raise

    def _get_temporal_distribution(self, df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
        """Calculate temporal listening distributions"""
        return {
            'hourly': df.groupby(df['ts'].dt.hour)['ms_played'].sum().to_dict(),
            'daily': df.groupby(df['ts'].dt.dayofweek)['ms_played'].sum().to_dict(),
            'monthly': df.groupby(df['ts'].dt.month)['ms_played'].sum().to_dict()
        }

    def _compare_basic_stats(self, user1_data: Dict, user2_data: Dict) -> Dict[str, Any]:
        """Compare basic listening statistics"""
        def calculate_percentile(value1: float, value2: float) -> float:
            return (min(value1, value2) / max(value1, value2)) * 100

        total_time1 = user1_data['total_time']
        total_time2 = user2_data['total_time']
        
        unique_artists1 = len(user1_data['artist_counts'])
        unique_artists2 = len(user2_data['artist_counts'])
        
        unique_tracks1 = len(user1_data['track_counts'])
        unique_tracks2 = len(user2_data['track_counts'])
        
        return {
            'listening_time_similarity': calculate_percentile(total_time1, total_time2),
            'artist_variety_similarity': calculate_percentile(unique_artists1, unique_artists2),
            'track_variety_similarity': calculate_percentile(unique_tracks1, unique_tracks2),
            'comparisons': {
                'total_time_hours': {
                    'user1': total_time1 / (1000 * 60 * 60),
                    'user2': total_time2 / (1000 * 60 * 60)
                },
                'unique_artists': {
                    'user1': unique_artists1,
                    'user2': unique_artists2
                },
                'unique_tracks': {
                    'user1': unique_tracks1,
                    'user2': unique_tracks2
                }
            }
        }

    def _compare_artists(self, user1_data: Dict, user2_data: Dict) -> Dict[str, Any]:
        """Compare artist preferences between users"""
        # Normalize artist counts to percentages
        def normalize_counts(counts: pd.Series) -> pd.Series:
            return counts / counts.sum()

        artists1 = normalize_counts(user1_data['artist_counts'])
        artists2 = normalize_counts(user2_data['artist_counts'])

        # Find common artists
        common_artists = set(artists1.index) & set(artists2.index)
        
        # Calculate Pearson correlation for common artists
        if common_artists:
            correlation = pearsonr(
                [artists1[artist] for artist in common_artists],
                [artists2[artist] for artist in common_artists]
            )[0]
        else:
            correlation = 0

        return {
            'common_artists_count': len(common_artists),
            'common_artists_percentage': len(common_artists) / max(len(artists1), len(artists2)) * 100,
            'artist_preference_correlation': correlation if not np.isnan(correlation) else 0,
            'top_shared_artists': self._get_top_shared_artists(artists1, artists2, 10),
            'unique_preferences': {
                'user1_unique': list(set(artists1.nlargest(10).index) - common_artists),
                'user2_unique': list(set(artists2.nlargest(10).index) - common_artists)
            }
        }

    def _get_top_shared_artists(self, artists1: pd.Series, artists2: pd.Series, n: int) -> List[Dict[str, Any]]:
        """Get top shared artists with their relative rankings"""
        common_artists = set(artists1.index) & set(artists2.index)
        shared_artists = []
        
        for artist in common_artists:
            rank1 = artists1.rank(ascending=False)[artist]
            rank2 = artists2.rank(ascending=False)[artist]
            avg_rank = (rank1 + rank2) / 2
            shared_artists.append({
                'artist': artist,
                'user1_rank': int(rank1),
                'user2_rank': int(rank2),
                'user1_percentage': artists1[artist] * 100,
                'user2_percentage': artists2[artist] * 100
            })
        
        return sorted(shared_artists, key=lambda x: x['user1_percentage'] + x['user2_percentage'], reverse=True)[:n]

    def _compare_temporal_patterns(self, user1_data: Dict, user2_data: Dict) -> Dict[str, float]:
        """Compare temporal listening patterns"""
        dist1 = user1_data['temporal_distribution']
        dist2 = user2_data['temporal_distribution']
        
        similarities = {}
        for timeframe in ['hourly', 'daily', 'monthly']:
            # Create vectors for comparison
            all_periods = set(dist1[timeframe].keys()) | set(dist2[timeframe].keys())
            vector1 = [dist1[timeframe].get(period, 0) for period in all_periods]
            vector2 = [dist2[timeframe].get(period, 0) for period in all_periods]
            
            # Normalize vectors
            vector1 = np.array(vector1) / np.sum(vector1)
            vector2 = np.array(vector2) / np.sum(vector2)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(vector1, vector2)
            similarities[f'{timeframe}_similarity'] = similarity if not np.isnan(similarity) else 0
            
        return similarities

    def _compare_genres(self, user1_data: Dict, user2_data: Dict) -> Dict[str, Any]:
        """Compare genre preferences between users"""
        def extract_genres(df: pd.DataFrame) -> Dict[str, int]:
            genres = defaultdict(int)
            for _, row in df.iterrows():
                if 'api_data' in row and isinstance(row['api_data'], dict):
                    if 'musicbrainz' in row['api_data']:
                        for tag in row['api_data']['musicbrainz'].get('tags', []):
                            genres[tag] += 1
            return dict(genres)

        genres1 = extract_genres(user1_data['raw_data'])
        genres2 = extract_genres(user2_data['raw_data'])
        
        common_genres = set(genres1.keys()) & set(genres2.keys())
        
        # Calculate genre similarity using cosine similarity
        all_genres = list(set(genres1.keys()) | set(genres2.keys()))
        vector1 = [genres1.get(genre, 0) for genre in all_genres]
        vector2 = [genres2.get(genre, 0) for genre in all_genres]
        
        # Normalize vectors
        vector1 = np.array(vector1) / np.sum(vector1) if np.sum(vector1) > 0 else np.zeros_like(vector1)
        vector2 = np.array(vector2) / np.sum(vector2) if np.sum(vector2) > 0 else np.zeros_like(vector2)
        
        similarity = 1 - cosine(vector1, vector2) if np.sum(vector1) > 0 and np.sum(vector2) > 0 else 0
        
        return {
            'genre_similarity_score': similarity if not np.isnan(similarity) else 0,
            'common_genres_count': len(common_genres),
            'top_shared_genres': sorted(
                [(genre, genres1[genre], genres2[genre]) for genre in common_genres],
                key=lambda x: x[1] + x[2],
                reverse=True
            )[:10],
            'unique_preferences': {
                'user1_unique': list(set(genres1.keys()) - common_genres),
                'user2_unique': list(set(genres2.keys()) - common_genres)
            }
        }

    def _compare_platforms(self, user1_data: Dict, user2_data: Dict) -> Dict[str, Any]:
        """Compare platform usage patterns"""
        platforms1 = user1_data['platform_counts']
        platforms2 = user2_data['platform_counts']
        
        # Calculate platform usage similarities
        common_platforms = set(platforms1.index) & set(platforms2.index)
        
        # Normalize platform usage
        platforms1_norm = platforms1 / platforms1.sum()
        platforms2_norm = platforms2 / platforms2.sum()
        
        return {
            'common_platforms': list(common_platforms),
            'platform_similarity': {
                platform: {
                    'user1_percentage': platforms1_norm.get(platform, 0) * 100,
                    'user2_percentage': platforms2_norm.get(platform, 0) * 100
                }
                for platform in common_platforms
            },
            'unique_platforms': {
                'user1_unique': list(set(platforms1.index) - common_platforms),
                'user2_unique': list(set(platforms2.index) - common_platforms)
            }
        }

    def _compare_listening_behavior(self, user1_data: Dict, user2_data: Dict) -> Dict[str, Any]:
        """Compare listening behaviors like skipping, shuffling, and session patterns"""
        def calculate_behavior_stats(df: pd.DataFrame) -> Dict[str, float]:
            return {
                'skip_rate': (df['skipped'].sum() / len(df)) * 100,
                'shuffle_rate': (df['shuffle'].sum() / len(df)) * 100,
                'offline_rate': (df['offline'].sum() / len(df)) * 100,
                'avg_track_completion': df['ms_played'].mean(),
                'completion_distribution': {
                    'short_plays': len(df[df['ms_played'] < 30000]),  # < 30 seconds
                    'medium_plays': len(df[(df['ms_played'] >= 30000) & (df['ms_played'] < 240000)]),
                    'long_plays': len(df[df['ms_played'] >= 240000])  # > 4 minutes
                }
            }

        behavior1 = calculate_behavior_stats(user1_data['raw_data'])
        behavior2 = calculate_behavior_stats(user2_data['raw_data'])

        return {
            'behavior_patterns': {
                'user1': behavior1,
                'user2': behavior2
            },
            'behavior_similarity': {
                metric: self._calculate_similarity(behavior1[metric], behavior2[metric])
                for metric in ['skip_rate', 'shuffle_rate', 'offline_rate', 'avg_track_completion']
            }
        }

    def _calculate_overall_similarity(self, user1_data: Dict, user2_data: Dict) -> float:
        """Calculate overall similarity score between users"""
        # Weighted combination of various similarity metrics
        artist_sim = self._compare_artists(user1_data, user2_data)
        temporal_sim = self._compare_temporal_patterns(user1_data, user2_data)
        genre_sim = self._compare_genres(user1_data, user2_data)
        
        weights = {
            'artist_preference': 0.4,
            'genre': 0.3,
            'temporal': 0.3
        }
        
        similarity_scores = {
            'artist_preference': artist_sim['artist_preference_correlation'],
            'genre': genre_sim['genre_similarity_score'],
            'temporal': np.mean([
                temporal_sim['hourly_similarity'],
                temporal_sim['daily_similarity'],
                temporal_sim['monthly_similarity']
            ])
        }
        
        overall_similarity = sum(
            score * weights[metric]
            for metric, score in similarity_scores.items()
        )
        
        return max(0, min(100, overall_similarity * 100))  # Convert to percentage

    @staticmethod
    def _calculate_similarity(value1: float, value2: float) -> float:
        """Calculate similarity percentage between two values"""
        if value1 == value2:
            return 100
        max_val = max(value1, value2)
        min_val = min(value1, value2)
        return (min_val / max_val * 100) if max_val > 0 else 0
