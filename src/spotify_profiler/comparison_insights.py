from typing import Dict, Any, List, Tuple
import logging
import numpy as np
from datetime import datetime

class ComparisonInsightsAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_comparison(self, comparison_results: Dict[str, Any], 
                         user1_name: str = "User 1", 
                         user2_name: str = "User 2") -> Dict[str, Any]:
        """Generate comprehensive insights from comparison results"""
        try:
            return {
                'overview': self._generate_comparison_overview(
                    comparison_results, user1_name, user2_name
                ),
                'taste_analysis': self._analyze_taste_compatibility(
                    comparison_results, user1_name, user2_name
                ),
                'engagement_comparison': self._compare_engagement_patterns(
                    comparison_results, user1_name, user2_name
                ),
                'temporal_comparison': self._compare_temporal_patterns(
                    comparison_results, user1_name, user2_name
                ),
                'listening_habits': self._compare_listening_habits(
                    comparison_results, user1_name, user2_name
                )
            }
        except Exception as e:
            self.logger.error(f"Error generating comparison insights: {e}")
            raise

    def _generate_comparison_overview(self, data: Dict[str, Any], 
                                    user1: str, user2: str) -> Dict[str, Any]:
        """Generate high-level overview of listening similarities and differences"""
        basic = data.get('basic_comparison', {})
        overall = data.get('overall_similarity', 0)

        comparisons = basic.get('comparisons', {})
        time_diff = self._calculate_time_difference(
            comparisons.get('total_time_hours', {})
        )

        return {
            'overall_compatibility': {
                'score': f"{overall:.1f}%",
                'interpretation': self._interpret_similarity(overall),
                'explanation': self._explain_similarity(overall, basic)
            },
            'time_investment': {
                'difference': time_diff,
                'explanation': self._explain_time_difference(time_diff, user1, user2)
            },
            'variety_comparison': self._compare_variety(comparisons, user1, user2)
        }

    def _analyze_taste_compatibility(self, data: Dict[str, Any], 
                                   user1: str, user2: str) -> Dict[str, Any]:
        """Analyze music taste compatibility between users"""
        artist_sim = data.get('artist_similarity', {})
        genre_sim = data.get('genre_similarity', {})

        return {
            'artist_overlap': {
                'score': artist_sim.get('artist_preference_correlation', 0),
                'common_artists': len(artist_sim.get('top_shared_artists', [])),
                'unique_preferences': self._analyze_unique_preferences(
                    artist_sim.get('unique_preferences', {}), user1, user2
                )
            },
            'genre_compatibility': {
                'score': genre_sim.get('genre_similarity_score', 0),
                'common_genres': genre_sim.get('common_genres_count', 0),
                'explanation': self._explain_genre_compatibility(genre_sim, user1, user2)
            },
            'taste_summary': self._generate_taste_summary(artist_sim, genre_sim)
        }

    def _compare_engagement_patterns(self, data: Dict[str, Any], 
                                   user1: str, user2: str) -> Dict[str, Any]:
        """Compare how users engage with music"""
        habits = data.get('listening_habits', {})
        engagement = habits.get('engagement_similarity', {})

        return {
            'listening_style': {
                'skip_similarity': engagement.get('skip_rate', 0),
                'shuffle_similarity': engagement.get('shuffle_rate', 0),
                'explanation': self._explain_engagement_patterns(engagement, user1, user2)
            },
            'session_comparison': self._compare_sessions(
                habits.get('session_patterns', {}), user1, user2
            )
        }

    def _compare_temporal_patterns(self, data: Dict[str, Any], 
                                 user1: str, user2: str) -> Dict[str, Any]:
        """Compare temporal listening patterns"""
        temporal = data.get('temporal_similarity', {})

        return {
            'rhythm_compatibility': {
                'daily_similarity': temporal.get('daily_similarity', 0),
                'hourly_similarity': temporal.get('hourly_similarity', 0),
                'explanation': self._explain_temporal_compatibility(temporal)
            },
            'peak_periods': self._analyze_peak_periods(temporal, user1, user2)
        }

    def _compare_listening_habits(self, data: Dict[str, Any], 
                                user1: str, user2: str) -> Dict[str, Any]:
        """Compare detailed listening habits"""
        habits = data.get('listening_habits', {})
        
        return {
            'session_behavior': {
                'similarity': self._calculate_session_similarity(habits),
                'differences': self._explain_session_differences(habits, user1, user2)
            },
            'engagement_style': self._compare_engagement_styles(habits, user1, user2)
        }

    def _analyze_unique_preferences(self, unique_prefs: Dict[str, List], 
                                  user1: str, user2: str) -> Dict[str, Any]:
        """Analyze unique preferences between users"""
        user1_unique = unique_prefs.get('user1_unique', [])
        user2_unique = unique_prefs.get('user2_unique', [])
        
        return {
            'counts': {
                user1: len(user1_unique),
                user2: len(user2_unique)
            },
            'unique_items': {
                user1: user1_unique[:5],
                user2: user2_unique[:5]
            },
            'analysis': self._generate_uniqueness_explanation(
                user1, user2, len(user1_unique), len(user2_unique)
            )
        }

    def _calculate_time_difference(self, time_data: Dict[str, float]) -> float:
        """Calculate absolute time difference between users"""
        return abs(time_data.get('user1', 0) - time_data.get('user2', 0))

    def _interpret_similarity(self, score: float) -> str:
        """Interpret overall similarity score"""
        if score >= 80:
            return "Very High Compatibility"
        elif score >= 60:
            return "High Compatibility"
        elif score >= 40:
            return "Moderate Compatibility"
        elif score >= 20:
            return "Low Compatibility"
        return "Very Low Compatibility"

    def _explain_similarity(self, overall: float, basic: Dict) -> str:
        """Generate detailed explanation of similarity"""
        factors = []
        if basic.get('listening_time_similarity', 0) > 70:
            factors.append("similar listening time investments")
        if basic.get('artist_variety_similarity', 0) > 70:
            factors.append("similar variety in artist selection")
        if basic.get('track_variety_similarity', 0) > 70:
            factors.append("similar track exploration patterns")

        if factors:
            return f"The compatibility is driven by {', '.join(factors)}"
        return "The listening patterns show significant differences"

    def _explain_time_difference(self, diff: float, user1: str, user2: str) -> str:
        """Explain the difference in listening time"""
        if diff < 0.1:
            return "Both users show very similar time investment in music"
        elif diff < 1:
            return f"{user1} and {user2} have slightly different listening durations"
        return f"There's a significant difference in listening time between {user1} and {user2}"

    def _compare_variety(self, comparisons: Dict, user1: str, user2: str) -> Dict[str, str]:
        """Compare and explain variety in listening"""
        artists1 = comparisons.get('unique_artists', {}).get('user1', 0)
        artists2 = comparisons.get('unique_artists', {}).get('user2', 0)
        
        tracks1 = comparisons.get('unique_tracks', {}).get('user1', 0)
        tracks2 = comparisons.get('unique_tracks', {}).get('user2', 0)

        return {
            'artist_variety': self._explain_variety_difference(
                artists1, artists2, "artists", user1, user2
            ),
            'track_variety': self._explain_variety_difference(
                tracks1, tracks2, "tracks", user1, user2
            )
        }

    def _explain_variety_difference(self, count1: int, count2: int, 
                                  item_type: str, user1: str, user2: str) -> str:
        """Explain the difference in variety between users"""
        diff = abs(count1 - count2)
        if diff == 0:
            return f"Both users explored the same number of {item_type}"
        elif diff <= 2:
            return f"Similar exploration of {item_type} with slight differences"
        else:
            more_variety = user1 if count1 > count2 else user2
            return f"{more_variety} explored a wider variety of {item_type}"

    def _generate_uniqueness_explanation(self, user1: str, user2: str, 
                                       count1: int, count2: int) -> str:
        """Generate explanation for unique preferences"""
        if count1 == 0 and count2 == 0:
            return "Both users share all their preferences"
        if count1 > count2:
            return f"{user1} shows more unique preferences ({count1} items) compared to {user2} ({count2} items)"
        elif count2 > count1:
            return f"{user2} shows more unique preferences ({count2} items) compared to {user1} ({count1} items)"
        return f"Both users have an equal number of unique preferences ({count1} items each)"

    def _explain_genre_compatibility(self, genre_sim: Dict, user1: str, user2: str) -> str:
        """Explain genre compatibility between users"""
        score = genre_sim.get('genre_similarity_score', 0)
        common = genre_sim.get('common_genres_count', 0)
        
        if score > 80:
            return f"{user1} and {user2} have very similar genre preferences"
        elif score > 50:
            return f"Moderate genre overlap with {common} genres in common"
        return "Distinct genre preferences with limited overlap"

    def _generate_taste_summary(self, artist_sim: Dict, genre_sim: Dict) -> str:
        """Generate overall taste compatibility summary"""
        artist_score = artist_sim.get('artist_preference_correlation', 0)
        genre_score = genre_sim.get('genre_similarity_score', 0)
        avg_score = (artist_score + genre_score) / 2
        
        if avg_score > 75:
            return "Very similar music taste"
        elif avg_score > 50:
            return "Moderately similar music preferences"
        return "Distinct musical preferences"

    def _explain_engagement_patterns(self, engagement: Dict, user1: str, user2: str) -> str:
        """Explain similarities and differences in engagement patterns"""
        skip_sim = engagement.get('skip_rate', 0)
        shuffle_sim = engagement.get('shuffle_rate', 0)
        
        patterns = []
        if skip_sim > 80:
            patterns.append("very similar skipping behavior")
        if shuffle_sim > 80:
            patterns.append("similar shuffle preferences")
            
        if patterns:
            return f"Users show {' and '.join(patterns)}"
        return "Distinct listening interaction patterns"

    def _compare_sessions(self, session_patterns: Dict, user1: str, user2: str) -> Dict[str, Any]:
        """Compare and analyze session patterns"""
        avg_sim = session_patterns.get('avg_session_similarity', 0)
        tracks_sim = session_patterns.get('tracks_per_session_similarity', 0)
        
        return {
            'duration_similarity': avg_sim,
            'content_similarity': tracks_sim,
            'explanation': self._explain_session_patterns(avg_sim, tracks_sim, user1, user2)
        }

    def _explain_session_patterns(self, duration_sim: float, content_sim: float, 
                                user1: str, user2: str) -> str:
        """Generate explanation for session pattern comparison"""
        if duration_sim > 80 and content_sim > 80:
            return "Very similar listening session patterns"
        elif duration_sim > 50 and content_sim > 50:
            return "Moderately similar listening sessions"
        return "Different approaches to listening sessions"

    def _explain_temporal_compatibility(self, temporal: Dict) -> str:
        """Explain temporal listening pattern compatibility"""
        daily_sim = temporal.get('daily_similarity', 0)
        hourly_sim = temporal.get('hourly_similarity', 0)
        
        if daily_sim > 80 and hourly_sim > 80:
            return "Very similar listening schedules"
        elif daily_sim > 50 and hourly_sim > 50:
            return "Somewhat similar listening times"
        return "Different listening schedules"

    def _analyze_peak_periods(self, temporal: Dict, user1: str, user2: str) -> Dict[str, Any]:
        """Analyze peak listening periods"""
        return {
            'similarity': (temporal.get('daily_similarity', 0) + 
                         temporal.get('hourly_similarity', 0)) / 2,
            'explanation': self._explain_peak_periods(temporal, user1, user2)
        }

    def _explain_peak_periods(self, temporal: Dict, user1: str, user2: str) -> str:
        """Generate explanation for peak period comparison"""
        if temporal.get('daily_similarity', 0) > 80:
            return "Users tend to listen to music at similar times"
        return "Users have different preferred listening times"

    def _calculate_session_similarity(self, habits: Dict) -> float:
        """Calculate overall session similarity"""
        patterns = habits.get('session_patterns', {})
        return (patterns.get('avg_session_similarity', 0) + 
                patterns.get('tracks_per_session_similarity', 0)) / 2

    def _explain_session_differences(self, habits: Dict, user1: str, user2: str) -> str:
        """Explain differences in session behavior"""
        similarity = self._calculate_session_similarity(habits)
        if similarity > 80:
            return f"{user1} and {user2} have very similar listening session patterns"
        elif similarity > 50:
            return "Moderate similarities in how listening sessions are structured"
        return "Different approaches to structuring listening sessions"

    def _compare_engagement_styles(self, habits: Dict, user1: str, user2: str) -> Dict[str, Any]:
        """Compare and explain engagement styles"""
        engagement = habits.get('engagement_similarity', {})
        total_similarity = (engagement.get('skip_rate', 0) + 
                          engagement.get('shuffle_rate', 0)) / 2
        
        return {
            'style_similarity': f"{total_similarity:.1f}%",
            'explanation': self._explain_style_differences(engagement, user1, user2)
        }

    def _explain_style_differences(self, engagement: Dict, user1: str, user2: str) -> str:
        """Generate explanation for engagement style differences"""
        skip_sim = engagement.get('skip_rate', 0)
        shuffle_sim = engagement.get('shuffle_rate', 0)
        
        if skip_sim > 80 and shuffle_sim > 80:
            return f"{user1} and {user2} interact with music very similarly"
        elif skip_sim < 20 and shuffle_sim < 20:
            return f"{user1} and {user2} have very different music interaction styles"
        
        differences = []
        if skip_sim < 50:
            differences.append("skipping behavior")
        if shuffle_sim < 50:
            differences.append("shuffle preferences")
            
        if differences:
            return f"Users differ mainly in their {' and '.join(differences)}"
        return "Mixed similarity in interaction patterns"

    def _analyze_engagement_correlation(self, data: Dict[str, Any], user1: str, user2: str) -> Dict[str, Any]:
        """Analyze correlation between users' engagement patterns"""
        habits = data.get('listening_habits', {})
        engagement = habits.get('engagement_similarity', {})
        
        skip_correlation = self._calculate_behavior_correlation(
            engagement.get('skip_rate', 0),
            'skipping behavior'
        )
        
        shuffle_correlation = self._calculate_behavior_correlation(
            engagement.get('shuffle_rate', 0),
            'playlist navigation'
        )
        
        return {
            'correlations': {
                'skip_behavior': skip_correlation,
                'shuffle_behavior': shuffle_correlation
            },
            'explanation': self._explain_engagement_correlation(
                skip_correlation,
                shuffle_correlation,
                user1,
                user2
            )
        }

    def _calculate_behavior_correlation(self, similarity: float, behavior_type: str) -> Dict[str, Any]:
        """Calculate and categorize behavior correlation"""
        correlation_level = "strong" if similarity > 75 else \
                          "moderate" if similarity > 50 else \
                          "weak"
        
        return {
            'score': similarity,
            'level': correlation_level,
            'behavior': behavior_type
        }

    def _explain_engagement_correlation(self, skip_corr: Dict, shuffle_corr: Dict,
                                     user1: str, user2: str) -> str:
        """Generate explanation for engagement correlation"""
        correlations = []
        
        if skip_corr['level'] == "strong":
            correlations.append(f"very similar {skip_corr['behavior']}")
        elif skip_corr['level'] == "moderate":
            correlations.append(f"somewhat similar {skip_corr['behavior']}")
            
        if shuffle_corr['level'] == "strong":
            correlations.append(f"very similar {shuffle_corr['behavior']}")
        elif shuffle_corr['level'] == "moderate":
            correlations.append(f"somewhat similar {shuffle_corr['behavior']}")
            
        if correlations:
            return f"{user1} and {user2} show {' and '.join(correlations)}"
        return f"{user1} and {user2} have distinct music interaction patterns"
