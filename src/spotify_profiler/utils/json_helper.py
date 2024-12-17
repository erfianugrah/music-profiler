import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_results(results: dict, filename: str) -> None:
    """Save results to JSON file with proper type conversion"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)
    except Exception as e:
        raise Exception(f"Error saving results: {str(e)}")
