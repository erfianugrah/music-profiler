from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import percentileofscore
import json
from pathlib import Path
import logging

class UserComparisonAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define baseline statistics for realistic probability calculation
        self.baseline_stats = {
            'genre_overlap_distribution': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'artist_overlap_distribution': [2, 5, 8, 12, 15, 20, 25, 30, 35, 40],
            'max_realistic_score': 85.0
        }
        
        # Weights for different comparison aspects
        self.weight_factors = {
            'genre_overlap': 0.35,
            'artist_overlap': 0.25,
            'era_preference': 0.20,
            'listening_patterns': 0.20
        }

    def compare_users(self, user1_results: Dict, user2_results: Dict) -> Dict[str, Any]:
        """Compare two users' listening histories and calculate similarity metrics"""
        try:
            # Calculate detailed comparison metrics
            detailed_comparison = {
                'listening_habits': self._compare_listening_habits(
                    user1_results['user_metrics'],
                    user2_results['user_metrics']
                ),
                'music_taste': self._compare_music_taste(user1_results, user2_results),
                'temporal_patterns': self._compare_temporal_patterns(
                    user1_results['temporal_patterns'],
                    user2_results['temporal_patterns']
                ),
                'shared_music': self._analyze_shared_music(user1_results, user2_results)
            }

            # Calculate realistic match probability
            match_probability = self._calculate_realistic_probability(user1_results, user2_results)
            
            return {
                'match_probability': match_probability,
                'detailed_comparison': detailed_comparison
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing users: {e}")
            raise

    def _calculate_realistic_probability(self, user1_results: Dict, user2_results: Dict) -> float:
        """Calculate realistic probability of finding this level of music taste similarity"""
        try:
            # Calculate raw similarities
            genre_sim = self._calculate_genre_similarity(
                user1_results['genres'],
                user2_results['genres']
            )
            
            artist_sim = self._calculate_artist_similarity(
                user1_results['artists'],
                user2_results['artists']
            )
            
            era_sim = self._calculate_era_similarity(
                user1_results['tracks'],
                user2_results['tracks']
            )
            
            pattern_sim = self._calculate_temporal_similarity(
                user1_results['temporal_patterns'],
                user2_results['temporal_patterns']
            )
            
            # Calculate percentile ranks
            genre_percentile = percentileofscore(self.baseline_stats['genre_overlap_distribution'], genre_sim)
            artist_percentile = percentileofscore(self.baseline_stats['artist_overlap_distribution'], artist_sim)
            
            # Combine scores with weights
            weighted_score = (
                genre_percentile * self.weight_factors['genre_overlap'] +
                artist_percentile * self.weight_factors['artist_overlap'] +
                era_sim * self.weight_factors['era_preference'] +
                pattern_sim * self.weight_factors['listening_patterns']
            )
            
            # Normalize to realistic maximum
            final_score = min(weighted_score, self.baseline_stats['max_realistic_score'])
            
            # Convert to probability
            probability = self._convert_to_probability(final_score)
            
            return round(probability, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating probability: {e}")
            return 0.0

    def _convert_to_probability(self, similarity_score: float) -> float:
        """Convert similarity score to probability of finding such a match"""
        if similarity_score >= self.baseline_stats['max_realistic_score']:
            return 0.1  # 0.1% chance of finding nearly identical taste
        
        # Exponential decay function for realistic distribution
        probability = 100 * np.exp(-0.05 * similarity_score)
        
        return min(100, max(0, probability))

    # [Previous methods remain the same: _compare_listening_habits, _compare_music_taste, etc.]
    
    def _calculate_genre_similarity(self, genres1: Dict, genres2: Dict) -> float:
        """Calculate genre similarity percentage"""
        try:
            genres1_set = set(genres1['top_genres'].keys())
            genres2_set = set(genres2['top_genres'].keys())
            
            if not genres1_set or not genres2_set:
                return 0.0
            
            intersection = len(genres1_set & genres2_set)
            union = len(genres1_set | genres2_set)
            
            return (intersection / union) * 100
            
        except (KeyError, AttributeError):
            return 0.0

    def _calculate_artist_similarity(self, artists1: Dict, artists2: Dict) -> float:
        """Calculate artist similarity percentage"""
        try:
            artists1_set = set(artists1['top_artists'].keys())
            artists2_set = set(artists2['top_artists'].keys())
            
            if not artists1_set or not artists2_set:
                return 0.0
            
            intersection = len(artists1_set & artists2_set)
            union = len(artists1_set | artists2_set)
            
            return (intersection / union) * 100
            
        except (KeyError, AttributeError):
            return 0.0

    def _calculate_era_similarity(self, tracks1: Dict, tracks2: Dict) -> float:
        """Calculate similarity in era preferences (0-100)"""
        try:
            # Extract release years
            years1 = [t['metadata'].get('release_year', 0) for t in tracks1.values()]
            years2 = [t['metadata'].get('release_year', 0) for t in tracks2.values()]
            
            years1 = [y for y in years1 if y > 0]
            years2 = [y for y in years2 if y > 0]
            
            if not years1 or not years2:
                return 0.0
            
            # Calculate median years
            median1 = np.median(years1)
            median2 = np.median(years2)
            
            # Calculate similarity based on decade difference
            decade_diff = abs(median1 - median2) / 10
            return max(0, 100 - (decade_diff * 20))
            
        except (KeyError, AttributeError):
            return 0.0

    def _calculate_temporal_similarity(self, temporal1: Dict, temporal2: Dict) -> float:
        """Calculate similarity in listening patterns (0-100)"""
        try:
            dist1 = temporal1['distributions']['hourly']
            dist2 = temporal2['distributions']['hourly']
            
            if not dist1 or not dist2:
                return 0.0
            
            # Normalize distributions
            norm1 = self._normalize_distribution(dist1)
            norm2 = self._normalize_distribution(dist2)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(
                list(norm1.values()),
                list(norm2.values())
            )
            
            return similarity * 100
            
        except (KeyError, AttributeError):
            return 0.0

    @staticmethod
    def _normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
        """Normalize a distribution to sum to 1"""
        total = sum(dist.values())
        return {k: v/total for k, v in dist.items()} if total > 0 else dist
