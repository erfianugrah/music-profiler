from typing import Dict, Any, List
import pandas as pd
from collections import defaultdict

class DataReconciler:
    @staticmethod
    def merge_track_data(spotify_data: Dict[str, Any], mb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge and reconcile data from both sources"""
        merged = {
            'basic_info': {
                'name': spotify_data.get('name', mb_data.get('title')),
                'duration': spotify_data.get('duration_ms') or mb_data.get('length'),
                'popularity': spotify_data.get('popularity', 0)
            },
            'release_info': {
                'date': spotify_data.get('release_date') or mb_data.get('date'),
                'album': spotify_data.get('album', {}).get('name'),
                'labels': mb_data.get('label-info-list', [])
            },
            'artist_info': {
                'name': spotify_data.get('artists', [{}])[0].get('name') or 
                       mb_data.get('artist-credit', [{}])[0].get('name'),
                'genres': list(set(
                    spotify_data.get('genres', []) + 
                    mb_data.get('artist_data', {}).get('genres', [])
                )),
                'tags': mb_data.get('artist_data', {}).get('tags', [])
            }
        }
        
        return merged

    @staticmethod
    def reconcile_genres(spotify_genres: List[str], mb_genres: List[str]) -> Dict[str, List[str]]:
        """Reconcile and categorize genres from both sources"""
        # Combine genres from both sources
        all_genres = set(spotify_genres + mb_genres)
        
        # Create genre mapping based on actual data
        categorized = defaultdict(list)
        for genre in all_genres:
            categorized['all'].append(genre)
            
            # Track source
            if genre in spotify_genres:
                categorized['spotify_source'].append(genre)
            if genre in mb_genres:
                categorized['musicbrainz_source'].append(genre)
        
        return dict(categorized)
