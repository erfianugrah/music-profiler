import concurrent.futures
import os
import time
from typing import Optional, Dict, Any, List
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import musicbrainzngs
from datetime import datetime, timedelta
import logging
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict, deque
from threading import Lock
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class MusicAPIClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self.cache = defaultdict(dict)
        self.error_count = 0
        self.success_count = 0
        self.spotify_available = False
        self.musicbrainz_available = False
        
        # Rate limiting setup
        self.spotify_requests = deque(maxlen=30)
        self.spotify_rate_limit = Lock()
        
        self._setup_clients()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        )
        # Suppress musicbrainzngs warnings
        logging.getLogger('musicbrainzngs').setLevel(logging.ERROR)

    def _setup_clients(self):
        """Initialize APIs with minimal console output"""
        env_path = find_dotenv()
        self.logger.debug(f"Looking for .env file at: {env_path}")
        
        if env_path:
            self.logger.debug("Found .env file, loading variables...")
            load_dotenv(env_path)
        else:
            self.logger.warning("No .env file found")

        # Try setting up each client independently
        self.spotify_available = False
        self.musicbrainz_available = False

        try:
            self._setup_spotify()
            self.spotify_available = True
            self.logger.debug("Successfully connected to Spotify API")
        except Exception as e:
            self.logger.warning(f"Spotify client initialization failed: {str(e)}")

        try:
            self._setup_musicbrainz()
            self.musicbrainz_available = True
            self.logger.debug("Successfully connected to MusicBrainz server")
        except Exception as e:
            self.logger.warning(f"MusicBrainz client initialization failed: {str(e)}")

        if not self.musicbrainz_available:
            raise RuntimeError("Failed to initialize MusicBrainz API")

        self.logger.debug(
            f"Services available - Spotify: {self.spotify_available}, "
            f"MusicBrainz: {self.musicbrainz_available}"
        )

    def _setup_connection_pools(self):
        """Setup connection pools with retry logic"""
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        
        if hasattr(self, 'sp'):
            self.sp._session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=100,
                    pool_maxsize=100
                )
            )

    def get_track_data(self, track_name: str, artist_name: str) -> Dict[str, Any]:
        """Get track data from MusicBrainz"""
        cache_key = f"{artist_name}-{track_name}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Get MusicBrainz data
            mb_data = self._get_musicbrainz_track(track_name, artist_name)
            
            # Create response with just MusicBrainz data
            data = {'musicbrainz': mb_data} if mb_data else {}
            
            if data:
                self.cache[cache_key] = data
                self.success_count += 1
            else:
                self.error_count += 1
                
            return data
            
        except Exception as e:
            self.logger.debug(f"Error getting track data: {e}")
            self.error_count += 1
            return {}

    def _setup_spotify(self):
        """Set up Spotify API client with OAuth for user data access"""
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback')
            
            if not client_id or not client_secret:
                raise ValueError("Missing Spotify credentials")

            # Set up OAuth with required scopes
            scope = 'user-read-recently-played user-read-playback-state'
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
                cache_path='.spotify_cache'
            )
            
            # Create Spotify client with OAuth
            self.sp = spotipy.Spotify(
                auth_manager=auth_manager,
                requests_timeout=10
            )
            
            # Test connection with user data endpoint
            self.sp.current_user()
            
        except Exception as e:
            self.logger.error(f"Error setting up Spotify client: {str(e)}")
            raise

    def _setup_musicbrainz(self):
        """Set up MusicBrainz API client"""
        try:
            # Configure MusicBrainz for local server
            musicbrainzngs.set_rate_limit(False)  # Disable rate limiting for local server
            musicbrainzngs.set_hostname("localhost:5000")  # Point to local server
            musicbrainzngs.set_useragent(
                "MusicProfiler",
                "0.1.0"
            )
            
            # Test connection
            try:
                musicbrainzngs.get_artist_by_id("f27ec8db-af05-4f36-916e-3d57f91ecf5e")
                self.logger.info("Successfully connected to local MusicBrainz server")
            except Exception as e:
                self.logger.warning(f"MusicBrainz connection test failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error setting up MusicBrainz client: {str(e)}")
            raise

    def get_streaming_history(self, after_timestamp: datetime) -> List[Dict]:
        """Get complete streaming history from Spotify after the given timestamp"""
        if not self.spotify_available:
            self.logger.warning("Spotify not available for fetching streams")
            return []

        try:
            all_tracks = []
            after_ms = int(after_timestamp.timestamp() * 1000)
            last_processed_ts = None
            
            while True:
                try:
                    # Apply rate limiting
                    with self.spotify_rate_limit:
                        current_time = time.time()
                        
                        # Remove old requests from tracking
                        while self.spotify_requests and current_time - self.spotify_requests[0] > 30:
                            self.spotify_requests.popleft()
                        
                        # Check if we need to wait
                        if len(self.spotify_requests) >= 25:
                            wait_time = 30 - (current_time - self.spotify_requests[0])
                            if wait_time > 0:
                                time.sleep(wait_time + 0.1)
                        
                        # Add current request timestamp
                        self.spotify_requests.append(current_time)
                    
                    # Get recently played tracks
                    response = self.sp.current_user_recently_played(
                        limit=50,
                        after=after_ms
                    )
                    
                    if not response or 'items' not in response or not response['items']:
                        break
                        
                    for item in response['items']:
                        played_at = datetime.fromtimestamp(int(item['played_at']) / 1000)
                        
                        # Check for duplicate timestamps
                        if last_processed_ts and played_at <= last_processed_ts:
                            continue
                        
                        track_data = {
                            'ts': played_at.isoformat(),
                            'master_metadata_track_name': item['track']['name'],
                            'master_metadata_album_artist_name': item['track']['artists'][0]['name'],
                            'master_metadata_album_album_name': item['track']['album']['name'],
                            'master_metadata_track_duration_ms': item['track']['duration_ms'],
                            'spotify_track_uri': item['track']['id'],
                            'ms_played': item['track']['duration_ms'],  # Actual played duration not provided by API
                            'platform': 'spotify_api',
                            'conn_country': 'unknown',
                            'reason_start': 'trackstart',
                            'reason_end': 'trackdone',
                            'shuffle': False,
                            'skipped': False,
                            'offline': False,
                            'offline_timestamp': None,
                            'incognito_mode': False
                        }
                        
                        # Get MusicBrainz data
                        mb_data = self._get_musicbrainz_track(
                            track_data['master_metadata_track_name'],
                            track_data['master_metadata_album_artist_name']
                        )
                        if mb_data:
                            track_data['api_data'] = {'musicbrainz': mb_data}
                        
                        all_tracks.append(track_data)
                        last_processed_ts = played_at
                    
                    if len(response['items']) < 50:
                        break
                        
                    # Update after_ms for next page
                    after_ms = int(last_processed_ts.timestamp() * 1000)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching streaming history page: {str(e)}")
                    break

            self.logger.info(f"Retrieved {len(all_tracks)} tracks from Spotify API")
            return all_tracks
                
        except Exception as e:
            self.logger.error(f"Error getting streaming history: {str(e)}")
            return []

    def combine_history(self, historical_df: pd.DataFrame, recent_tracks: List[Dict]) -> pd.DataFrame:
        """Combine historical data with recently retrieved tracks"""
        try:
            if not recent_tracks:
                return historical_df
                
            # Convert recent tracks to DataFrame
            recent_df = pd.DataFrame(recent_tracks)
            
            # Ensure timestamps are datetime
            historical_df['ts'] = pd.to_datetime(historical_df['ts'])
            recent_df['ts'] = pd.to_datetime(recent_df['ts'])
            
            # Sort both dataframes by timestamp
            historical_df = historical_df.sort_values('ts')
            recent_df = recent_df.sort_values('ts')
            
            # Check for overlap
            last_historical_ts = historical_df['ts'].max()
            overlap_df = recent_df[recent_df['ts'] <= last_historical_ts]
            
            if not overlap_df.empty:
                self.logger.info(f"Found {len(overlap_df)} overlapping tracks")
                # Keep only newer tracks
                recent_df = recent_df[recent_df['ts'] > last_historical_ts]
            
            # Combine and sort final dataset
            combined_df = pd.concat([historical_df, recent_df], ignore_index=True)
            combined_df = combined_df.sort_values('ts')
            
            # Remove any exact duplicates
            combined_df = combined_df.drop_duplicates(
                subset=['ts', 'master_metadata_track_name', 'master_metadata_album_artist_name'],
                keep='first'
            )
            
            self.logger.info(
                f"Combined {len(historical_df)} historical tracks with {len(recent_df)} new tracks. "
                f"Final dataset contains {len(combined_df)} tracks"
            )
            
            return combined_df
                
        except Exception as e:
            self.logger.error(f"Error combining history: {str(e)}")
            return historical_df

    def _get_musicbrainz_track(self, track_name: str, artist_name: str) -> Optional[Dict]:
        """Get track data from MusicBrainz with proper error handling"""
        try:
            # Clean input parameters
            track_name = track_name.strip()
            artist_name = artist_name.strip()
            
            # Search parameters for the recording search
            search_params = {
                'recording': track_name,
                'artist': artist_name,
                'limit': 1,
                'strict': False
            }
            
            # Search with more lenient matching
            result = musicbrainzngs.search_recordings(**search_params)
            
            # Safely check for results
            recordings = result.get('recording-list', []) if result else []
            if not recordings:
                self.logger.debug(f"No MusicBrainz results found for '{track_name}' by '{artist_name}'")
                return None
                
            recording = recordings[0]
                
            # Get additional artist data if available
            artist_data = None
            if recording.get('artist-credit'):
                try:
                    artist = recording['artist-credit'][0].get('artist', {})
                    if artist and 'id' in artist:
                        artist_data = self._get_musicbrainz_artist(artist['id'])
                except (KeyError, IndexError) as e:
                    self.logger.debug(f"Error getting artist data: {str(e)}")
            
            # Build response with safe gets
            return {
                'recording_id': recording.get('id'),
                'score': recording.get('ext:score'),
                'length': recording.get('length'),
                'title': recording.get('title'),
                'artist_credit': recording.get('artist-credit'),
                'artist_data': artist_data,
                'release_events': recording.get('release-list', []),
                'tags': [tag['name'] for tag in recording.get('tag-list', [])
                        if isinstance(tag, dict) and 'name' in tag],
                'isrcs': recording.get('isrc-list', [])
            }
                    
        except Exception as e:
            self.logger.debug(f"MusicBrainz API error: {str(e)}")
            return None

    def _get_musicbrainz_artist(self, artist_id: str) -> Optional[Dict]:
        """Get artist data from MusicBrainz"""
        try:
            artist = musicbrainzngs.get_artist_by_id(
                artist_id,
                includes=['tags', 'aliases', 'artist-rels', 'url-rels']
            )
            
            if not artist or 'artist' not in artist:
                return None
                
            artist_data = artist['artist']
            
            # Extract relationships
            relationships = defaultdict(list)
            if 'relation-list' in artist_data:
                for rel_list in artist_data['relation-list']:
                    rel_type = rel_list.get('target-type', 'unknown')
                    for rel in rel_list.get('relation', []):
                        if isinstance(rel, dict):
                            relationships[rel_type].append({
                                'type': rel.get('type'),
                                'target': rel.get('target'),
                                'begin': rel.get('begin'),
                                'end': rel.get('end'),
                                'ended': rel.get('ended', False)
                            })
            
            return {
                'id': artist_data.get('id'),
                'name': artist_data.get('name', 'Unknown'),
                'sort_name': artist_data.get('sort-name'),
                'type': artist_data.get('type'),
                'gender': artist_data.get('gender'),
                'area': artist_data.get('area', {}).get('name'),
                'begin_area': artist_data.get('begin-area', {}).get('name'),
                'life_span': artist_data.get('life-span', {}),
                'tags': [tag['name'] for tag in artist_data.get('tag-list', []) 
                        if isinstance(tag, dict) and 'name' in tag],
                'aliases': [alias.get('name') for alias in artist_data.get('alias-list', [])
                          if isinstance(alias, dict) and 'name' in alias],
                'relationships': dict(relationships),
                'isnis': artist_data.get('isni-list', []),
                'ipis': artist_data.get('ipi-list', [])
            }
            
        except Exception as e:
            self.logger.debug(f"MusicBrainz artist error for ID {artist_id}: {str(e)}")
            return None

    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def refresh_spotify_token(self) -> bool:
        """Refresh Spotify access token if needed"""
        try:
            if hasattr(self, 'sp') and hasattr(self.sp, 'auth_manager'):
                self.sp.auth_manager.refresh_access_token()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error refreshing Spotify token: {str(e)}")
            return False
