import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "logs") -> None:
    """Configure logging with strict console output control"""
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamped log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_log = log_path / f'music_profiler_debug_{timestamp}.log'
    error_log = log_path / f'music_profiler_error_{timestamp}.log'
    api_log = log_path / f'api_{timestamp}.log'
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    class ConsoleFilter(logging.Filter):
        def __init__(self, allowed_messages):
            super().__init__()
            self.allowed_messages = allowed_messages
            
        def filter(self, record):
            # Only allow specific progress messages
            return any(msg in record.msg for msg in self.allowed_messages)
    
    # Set up handlers
    # Console handler - strictly filtered
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleFilter([
        "Loading",
        "Found",
        "Loaded",
        "Processing",
        "Analysis complete",
        "Results saved",
        "Quick Summary",
        "Total tracks",
        "Unique artists",
        "Total listening time",
        "You can find your results in"
    ]))
    
    # Debug file handler - all messages
    debug_handler = logging.FileHandler(debug_log)
    debug_handler.setFormatter(file_formatter)
    debug_handler.setLevel(logging.DEBUG)
    
    # API file handler
    api_handler = logging.FileHandler(api_log)
    api_handler.setFormatter(file_formatter)
    api_handler.setLevel(logging.DEBUG)
    
    # Error file handler
    error_handler = logging.FileHandler(error_log)
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(api_handler)
    root_logger.addHandler(error_handler)
    
    # Configure third-party loggers to be silent on console
    for logger_name in ['spotipy', 'urllib3', 'musicbrainzngs', 'requests']:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logging.WARNING)
        third_party_logger.propagate = False
        third_party_logger.addHandler(api_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)
