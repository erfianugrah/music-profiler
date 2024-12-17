from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import time
import logging
import numpy as np
import concurrent.futures
import pickle
from pathlib import Path
import json
from datetime import datetime
from collections import deque
from threading import Lock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import sys
import os

class BatchProcessor:
    def __init__(self, api_client, batch_size=200, max_workers=100, state_file="processor_state.pkl"):
        self.api_client = api_client
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.state_file = Path(state_file)
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.success_count = 0
        self.is_paused = False
        
        # Rate limiting setup - adjust for Spotify's limits
        self.spotify_requests = deque(maxlen=50)  # Track last 50 requests
        self.spotify_rate_limit = Lock()
        
        self._init_state()
        self.setup_connection_pools()

    def _init_state(self):
        """Initialize or load existing state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                self.processed_indices = state['processed_indices']
                self.results = state['results']
                self.error_count = state.get('error_count', 0)
                self.success_count = state.get('success_count', 0)
            except Exception as e:
                self.logger.error(f"Error loading state: {e}")
                self._reset_state()
        else:
            self._reset_state()

    def setup_connection_pools(self):
        """Setup connection pools with retry logic"""
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        
        if hasattr(self.api_client, 'sp'):
            self.api_client.sp._session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=100,
                    pool_maxsize=100
                )
            )

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data with high-precision percentage display"""
        total_items = len(df)
        remaining_indices = list(set(range(total_items)) - self.processed_indices)
        
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 120

        try:
            with tqdm(
                total=total_items,
                initial=len(self.processed_indices),
                ncols=terminal_width,
                bar_format='{n_fmt}/{total_fmt} [{bar:20}] {percentage:3.4f}% {rate_fmt} {remaining}{postfix}',
                file=sys.stdout
            ) as pbar:
                last_time = time.time()
                
                while remaining_indices and not self.is_paused:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        batch_indices = remaining_indices[:self.batch_size]
                        futures = {
                            executor.submit(
                                self._process_track_with_timeout, 
                                (idx, df.iloc[idx])
                            ): idx for idx in batch_indices
                        }
                        
                        for future in concurrent.futures.as_completed(futures):
                            if self.is_paused:
                                break
                                
                            idx = futures[future]
                            try:
                                current_time = time.time()
                                processing_time = current_time - last_time
                                last_time = current_time
                                
                                result = future.result()
                                if result:
                                    track_name = result.get('master_metadata_track_name', '')
                                    artist_name = result.get('master_metadata_album_artist_name', '')
                                    track_info = f"{track_name} - {artist_name}"
                                    
                                    # Calculate available width for track info
                                    base_length = len(f"{total_items}/{total_items} [====================] 100.0000% 0.00it/s 00:00")
                                    available_width = terminal_width - base_length - 10
                                    
                                    if len(track_info) > available_width:
                                        track_info = track_info[:available_width-3] + "..."
                                    
                                    # Update with track info and timing
                                    postfix = f", {track_info} ({processing_time:.2f}s/track)"
                                    pbar.set_postfix_str(postfix)
                                    
                                    self.results.append(result)
                                    self.processed_indices.add(idx)
                                    remaining_indices.remove(idx)
                                    pbar.update(1)
                                    
                            except Exception as e:
                                self.logger.error(f"Error processing track at index {idx}: {e}")
                                self.error_count += 1
                            
                            if len(self.processed_indices) % 1000 == 0:
                                self._save_state()
                
                self._save_state()
                
                if self.results:
                    result_df = pd.DataFrame(self.results)
                    for col in df.columns:
                        if col not in result_df.columns:
                            result_df[col] = df[col]
                    return result_df
                return df
                
        except KeyboardInterrupt:
            self.logger.info("\nProcessing paused by user")
            self.is_paused = True
            self._save_state()
            if self.results:
                return pd.DataFrame(self.results)
            return df

    def _process_track_with_timeout(self, args: tuple, timeout: int = 30) -> Optional[Dict]:
        """Process a single track with timeout protection"""
        idx, row = args
        if idx in self.processed_indices:
            return None

        track_name = row['master_metadata_track_name']
        artist_name = row['master_metadata_album_artist_name']
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.api_client.get_track_data,
                    track_name,
                    artist_name
                )
                
                try:
                    api_data = future.result(timeout=timeout)
                    result = {col: row[col] for col in row.index}
                    result['api_data'] = api_data if api_data else {}
                    
                    if api_data:
                        self.success_count += 1
                    else:
                        self.error_count += 1
                        
                    return result
                    
                except concurrent.futures.TimeoutError:
                    self.logger.warning(
                        f"Timeout processing track: {track_name} - {artist_name}"
                    )
                    self.error_count += 1
                    return None
                    
        except Exception as e:
            self.logger.error(
                f"Error processing {track_name} by {artist_name}: {str(e)}"
            )
            self.error_count += 1
            return None

    def reset(self) -> None:
        """Reset all processor state"""
        self.logger.info("Resetting batch processor state")
        self._reset_state()
        self.is_paused = False
        self.spotify_requests.clear()  # Clear any stored rate limit data
        
        # Clear any existing state file
        if self.state_file.exists():
            try:
                self.state_file.unlink()  # Delete the state file
                self.logger.debug("Deleted existing state file")
            except Exception as e:
                self.logger.warning(f"Failed to delete state file: {e}")
        
        self.logger.info("Batch processor reset complete")

    def resume(self) -> None:
        """Resume processing from saved state"""
        self.logger.info("Attempting to resume from saved state")
        if not self.state_file.exists():
            self.logger.warning("No state file found, starting fresh")
            self._reset_state()
            return
            
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
                
            self.processed_indices = state['processed_indices']
            self.results = state['results']
            self.error_count = state.get('error_count', 0)
            self.success_count = state.get('success_count', 0)
            
            self.logger.info(f"Successfully resumed from state. "
                           f"Processed: {len(self.processed_indices)}, "
                           f"Results: {len(self.results)}, "
                           f"Errors: {self.error_count}")
            
        except Exception as e:
            self.logger.error(f"Error resuming from state file: {e}")
            self.logger.warning("Starting fresh due to state load failure")
            self._reset_state()

    # State management methods
    def _reset_state(self):
        """Reset processing state"""
        self.processed_indices = set()
        self.results = []
        self.error_count = 0
        self.success_count = 0
        self._save_state()

    def _save_state(self):
        """Save current processing state"""
        state = {
            'processed_indices': self.processed_indices,
            'results': self.results,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    # Processing methods
    def _process_track(self, args: tuple) -> Dict[str, Any]:
        """Process a single track with rate limiting"""
        idx, row = args
        if idx in self.processed_indices:
            return None

        track_name = row['master_metadata_track_name']
        artist_name = row['master_metadata_album_artist_name']
        
        try:
            # Prepare result dictionary
            result = {col: self._convert_to_native_type(row[col]) for col in row.index}
            result['api_data'] = {}
            result['index'] = idx
            
            # Get track data from APIs
            try:
                # Check rate limits before Spotify API call
                if self.api_client.spotify_available:
                    self._check_rate_limit()
                    
                api_data = self.api_client.get_track_data(track_name, artist_name)
                if api_data:
                    result['api_data'] = api_data
                    
                    # Add additional fields from enhanced API responses
                    if 'spotify' in api_data:
                        result['spotify_genres'] = api_data['spotify'].get('genres', [])
                        result['spotify_popularity'] = api_data['spotify'].get('popularity')
                        
                    if 'musicbrainz' in api_data:
                        result['mb_genres'] = api_data['musicbrainz'].get('genres', [])
                        result['mb_tags'] = api_data['musicbrainz'].get('tags', [])
                        
            except Exception as e:
                self.logger.debug(f"API error for {track_name}: {e}")

            result['processed_successfully'] = bool(result['api_data'])
            
            if result['processed_successfully']:
                self.success_count += 1
            else:
                self.error_count += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {track_name} by {artist_name}: {str(e)}")
            self.error_count += 1
            return {col: self._convert_to_native_type(row[col]) for col in row.index}

    def _check_rate_limit(self) -> None:
        """Check Spotify rate limits with stricter control"""
        with self.spotify_rate_limit:
            current_time = time.time()
            
            # Remove requests older than 30 seconds
            while self.spotify_requests and current_time - self.spotify_requests[0] > 30:
                self.spotify_requests.popleft()
            
            # Spotify limit is about 30 requests per 30 seconds
            # Use a lower limit to be safe
            if len(self.spotify_requests) >= 25:  # Leave some buffer
                wait_time = 30 - (current_time - self.spotify_requests[0])
                if wait_time > 0:
                    self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    time.sleep(wait_time + 0.1)  # Add small buffer
            
            # Add current request timestamp
            self.spotify_requests.append(current_time)
            
            # Add a small delay between requests
            time.sleep(0.1)  # 100ms between requests


    @staticmethod
    def _convert_to_native_type(value: Any) -> Any:
        """Convert numpy/pandas types to Python native types"""
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            return value.isoformat()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
