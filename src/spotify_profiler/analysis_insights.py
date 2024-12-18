from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import numpy as np
from collections import defaultdict

class AnalysisInsights:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive insights from analysis results"""
        try:
            return {
                'summary': self._generate_summary(analysis_results),
                'behavior_insights': self._analyze_behavior(analysis_results),
                'genre_insights': self._analyze_genre_preferences(analysis_results),
                'artist_insights': self._analyze_artist_engagement(analysis_results),
                'temporal_insights': self._analyze_temporal_behavior(analysis_results)
            }
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            raise

    def _generate_summary(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate high-level summary of listening behavior"""
        metrics = data.get('user_metrics', {})
        stats = metrics.get('listening_stats', {})
        engagement = metrics.get('engagement_metrics', {})

        listening_type = self._determine_listening_type(
            skip_rate=engagement.get('skip_rate', 0),
            completion_rate=stats.get('completion_rate')
        )

        time_investment = self._categorize_time_investment(
            stats.get('total_time_hours', 0),
            stats.get('average_daily_time', 0)
        )

        return {
            'listening_style': f"This session shows {listening_type} behavior, with {engagement.get('skip_rate')}% skip rate",
            'time_investment': time_investment,
            'genre_preference': self._summarize_genre_preference(data.get('genres', {}))
        }

    def _determine_listening_type(self, skip_rate: float, completion_rate: Optional[float]) -> str:
        """Determine the type of listener based on behavior"""
        if skip_rate > 75:
            return "exploratory"
        elif completion_rate and completion_rate > 80:
            return "focused"
        else:
            return "casual"

    def _categorize_time_investment(self, total_hours: float, daily_average: float) -> str:
        """Categorize time investment in music listening"""
        if total_hours < 0.1:
            return "This represents a very brief listening session"
        elif total_hours < 1:
            return "This shows a short listening session"
        else:
            return "This indicates an extended listening session"

    def _summarize_genre_preference(self, genres: Dict[str, Any]) -> str:
        """Summarize genre preferences into a readable insight"""
        if not genres:
            return "No genre data available"

        top_genres = genres.get('top_genres', {})
        if not top_genres:
            return "No genre preferences detected"

        sorted_genres = sorted(top_genres.items(), key=lambda x: x[1], reverse=True)
        primary_genres = [genre for genre, count in sorted_genres if count > 1]
        
        if not primary_genres:
            return f"Explored various genres with focus on {sorted_genres[0][0]}"
        
        if len(primary_genres) == 1:
            return f"Strong preference for {primary_genres[0]}"
        
        if len(primary_genres) == 2:
            return f"Primary focus on {primary_genres[0]} and {primary_genres[1]}"
        
        return f"Diverse listening across multiple genres, particularly {', '.join(primary_genres[:3])}"

    def _analyze_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze listening behavior patterns"""
        tracks = data.get('tracks', {})
        metrics = data.get('user_metrics', {})
        
        completion_rates = []
        for track_info in tracks.values():
            if 'listening_data' in track_info:
                completion = track_info['listening_data'].get('completion_rate')
                if completion is not None:
                    completion_rates.append(completion)

        avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0

        return {
            'listening_pattern': {
                'type': 'song_sampler' if avg_completion < 25 else 'complete_listener',
                'avg_completion': f"{avg_completion:.1f}%",
                'explanation': self._explain_listening_pattern(avg_completion)
            },
            'engagement_level': self._analyze_engagement(metrics.get('engagement_metrics', {}))
        }

    def _analyze_engagement(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user engagement patterns based on metrics"""
        skip_rate = metrics.get('skip_rate', 0)
        shuffle_rate = metrics.get('shuffle_rate', 0)
        
        engagement_type = self._determine_engagement_type(skip_rate)
        playback_style = self._determine_playback_style(shuffle_rate)
        engagement_score = self._calculate_engagement_score(skip_rate, shuffle_rate)

        return {
            'type': engagement_type,
            'playback_style': playback_style,
            'engagement_score': engagement_score,
            'metrics': {
                'skip_rate': f"{skip_rate:.1f}%",
                'shuffle_rate': f"{shuffle_rate:.1f}%"
            },
            'analysis': self._explain_engagement_pattern(engagement_type, playback_style, skip_rate, shuffle_rate)
        }

    def _determine_engagement_type(self, skip_rate: float) -> str:
        """Determine engagement type based on skip rate"""
        if skip_rate > 75:
            return "explorer"
        elif skip_rate < 25:
            return "focused_listener"
        return "balanced_listener"

    def _determine_playback_style(self, shuffle_rate: float) -> str:
        """Determine playback style based on shuffle rate"""
        if shuffle_rate > 75:
            return "variety_seeker"
        elif shuffle_rate < 25:
            return "sequential_listener"
        return "mixed_playback"

    def _calculate_engagement_score(self, skip_rate: float, shuffle_rate: float) -> float:
        """Calculate overall engagement score"""
        return (100 - skip_rate) * 0.7 + (shuffle_rate if shuffle_rate < 50 else (100 - shuffle_rate)) * 0.3

    def _explain_listening_pattern(self, avg_completion: float) -> str:
        """Generate explanation for listening pattern"""
        if avg_completion < 25:
            return "The low completion rate suggests a song sampling behavior"
        elif avg_completion < 50:
            return "Moderate completion rates indicate casual listening"
        else:
            return "High completion rates show engaged listening"

    def _explain_engagement_pattern(self, engagement_type: str, playback_style: str, 
                                  skip_rate: float, shuffle_rate: float) -> str:
        """Generate detailed explanation of engagement pattern"""
        explanations = []
        
        if engagement_type == "explorer":
            explanations.append("Shows exploratory listening behavior with frequent track skipping")
        elif engagement_type == "focused_listener":
            explanations.append("Demonstrates focused listening with minimal track skipping")
        else:
            explanations.append("Maintains a balance between exploration and focused listening")

        if playback_style == "variety_seeker":
            explanations.append("Prefers shuffled playback, suggesting preference for variety")
        elif playback_style == "sequential_listener":
            explanations.append("Mostly listens to tracks in sequence")
        else:
            explanations.append("Alternates between sequential and shuffled playback")

        return " and ".join(explanations)

    def _analyze_genre_preferences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze genre preferences and patterns"""
        genres = data.get('genres', {})
        preferences = genres.get('genre_preferences', {})

        sorted_genres = sorted(
            preferences.items(),
            key=lambda x: x[1].get('play_share', 0),
            reverse=True
        )

        primary_genres = [g for g, stats in sorted_genres if stats.get('play_share', 0) > 50]
        
        return {
            'primary_genres': primary_genres,
            'genre_variety': len(genres.get('top_genres', {})),
            'genre_focus': self._determine_genre_focus(preferences),
            'explanation': self._explain_genre_distribution(sorted_genres)
        }

    def _determine_genre_focus(self, preferences: Dict) -> str:
        """Determine if listening is genre-focused or diverse"""
        high_share_genres = sum(1 for stats in preferences.values() 
                              if stats.get('play_share', 0) > 50)
        return "genre_focused" if high_share_genres > 0 else "genre_diverse"

    def _explain_genre_distribution(self, sorted_genres: List) -> str:
        """Explain the genre distribution pattern"""
        if not sorted_genres:
            return "No genre data available"
            
        top_genre = sorted_genres[0]
        if top_genre[1].get('play_share', 0) > 75:
            return f"Strong preference for {top_genre[0]}"
        elif len(sorted_genres) > 3:
            return "Diverse genre preferences across multiple categories"
        else:
            return "Moderate genre exploration"

    def _analyze_artist_engagement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze artist engagement patterns"""
        artists = data.get('artists', {}).get('artist_profiles', {})
        
        artist_completion = {}
        for artist, profile in artists.items():
            avg_completion = profile.get('avg_completion_rate')
            if avg_completion is not None:
                artist_completion[artist] = avg_completion

        most_completed = max(artist_completion.items(), key=lambda x: x[1]) if artist_completion else (None, 0)

        return {
            'engagement_pattern': {
                'most_engaged_artist': most_completed[0],
                'engagement_level': f"{most_completed[1]:.1f}% completion" if most_completed[0] else "N/A"
            },
            'artist_variety': len(artists),
            'explanation': self._explain_artist_engagement(artists)
        }

    def _explain_artist_engagement(self, artists: Dict) -> str:
        """Generate explanation for artist engagement"""
        if not artists:
            return "No artist data available"
            
        artist_count = len(artists)
        if artist_count == 1:
            return "Single artist focus"
        elif artist_count <= 3:
            return "Exploring a small selection of artists"
        else:
            return "Broad artist exploration"

    def _analyze_temporal_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal listening patterns"""
        temporal = data.get('temporal_patterns', {})
        distributions = temporal.get('distributions', {})
        
        peak_hour = max(
            distributions.get('hourly', {}).items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]

        return {
            'peak_listening': {
                'hour': peak_hour,
                'period': self._categorize_time_period(peak_hour)
            },
            'session_behavior': self._analyze_session_behavior(temporal.get('listening_sessions', {})),
            'explanation': self._explain_temporal_patterns(distributions)
        }

    def _categorize_time_period(self, hour: Optional[int]) -> str:
        """Categorize the time period of listening"""
        if hour is None:
            return "unknown"
        if 5 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 17:
            return "afternoon"
        elif 18 <= hour <= 22:
            return "evening"
        else:
            return "night"

    def _analyze_session_behavior(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze listening session behavior"""
        avg_duration = session_data.get('average_session_minutes', 0)
        tracks_per_session = session_data.get('average_tracks_per_session', 0)

        if avg_duration < 1:
            session_type = "brief_sampling"
        elif avg_duration < 5:
            session_type = "quick_session"
        else:
            session_type = "extended_session"

        return {
            'session_type': session_type,
            'avg_duration': f"{avg_duration:.1f} minutes",
            'tracks_per_session': f"{tracks_per_session:.1f} tracks"
        }

    def _explain_temporal_patterns(self, distributions: Dict) -> str:
        """Generate explanation for temporal patterns"""
        hourly = distributions.get('hourly', {})
        if not hourly:
            return "No temporal pattern data available"

        peak_hour = max(hourly.items(), key=lambda x: x[1])[0]
        period = self._categorize_time_period(peak_hour)
        return f"Peak listening activity during {period} hours"
