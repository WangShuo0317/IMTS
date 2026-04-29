"""
Utility for handling flexible state parameter parsing in data optimization tools.

The LLM may pass state as a dict (correct) or as a string (pipeline placeholder or JSON).
This module provides a helper to normalize state input, with fallback dataset path injection.
"""

import json
import logging
import os

logger = logging.getLogger("data_opt_agent.tools")

# Output directory for all runtime-generated files
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "temp_datasets")


def get_output_dir() -> str:
    """Return the shared output directory for runtime-generated files, creating it if needed."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR

# Global fallback dataset path — set by graph_engine before agent invocation
_FALLBACK_DATASET_PATH: str = ""


def set_fallback_dataset_path(path: str) -> None:
    """Set the global fallback dataset path for when state is a placeholder."""
    global _FALLBACK_DATASET_PATH
    _FALLBACK_DATASET_PATH = path


def get_fallback_dataset_path() -> str:
    """Get the current fallback dataset path."""
    return _FALLBACK_DATASET_PATH


def parse_state(state) -> dict:
    """Normalize state parameter to dict.

    Handles four cases:
    1. state is already a dict with dataset_path/path -> return directly
    2. state is a dict without path -> inject fallback dataset_path
    3. state is a JSON string -> parse and return
    4. state is a pipeline placeholder (#RETURNED_DATA_FROM_...) -> return dict with fallback dataset_path

    Args:
        state: The state parameter from LLM tool call.

    Returns:
        dict: Normalized state dictionary with at least a dataset_path or path key.
    """
    if isinstance(state, dict):
        has_path = state.get("dataset_path") or state.get("path")
        if has_path:
            return state
        fallback = get_fallback_dataset_path()
        if fallback:
            logger.warning(f"State dict has no path, injecting fallback: {fallback}")
            return {"dataset_path": fallback, "path": fallback}
        return state

    if isinstance(state, str):
        # Check for pipeline placeholder — inject fallback path instead of empty dict
        if state.startswith("#RETURNED_DATA_FROM_"):
            fallback = get_fallback_dataset_path()
            if fallback and os.path.exists(fallback):
                logger.warning(f"Pipeline placeholder detected: {state}. "
                               f"Injecting fallback dataset_path: {fallback}")
                return {"dataset_path": fallback, "path": fallback}
            else:
                logger.warning(f"Pipeline placeholder detected: {state}. "
                               f"No fallback path available, returning empty dict.")
                return {}

        # Try JSON parse
        try:
            parsed = json.loads(state)
            if isinstance(parsed, dict):
                has_path = parsed.get("dataset_path") or parsed.get("path")
                if not has_path:
                    fallback = get_fallback_dataset_path()
                    if fallback:
                        parsed["dataset_path"] = fallback
                        parsed["path"] = fallback
                return parsed
        except json.JSONDecodeError:
            logger.warning(f"State string is not valid JSON: {state[:200]}")

    # Fallback: return dict with fallback path
    fallback = get_fallback_dataset_path()
    if fallback and os.path.exists(fallback):
        logger.warning(f"Unexpected state type: {type(state)}, injecting fallback: {fallback}")
        return {"dataset_path": fallback, "path": fallback}

    logger.warning(f"Unexpected state type: {type(state)}, returning empty dict")
    return {}