"""Random seed utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Set global random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not available


def get_ci_safe_seed() -> int:
    """Get a deterministic seed for CI environments."""
    # Use fixed seed in CI for reproducibility
    return 42


def should_use_ci_mode() -> bool:
    """Check if we should use CI-safe configurations."""
    ci_indicators = [
        os.getenv("CI"),
        os.getenv("GITHUB_ACTIONS"), 
        os.getenv("PYTEST_CURRENT_TEST"),
    ]
    
    return any(ci_indicators)


def get_ci_safe_config_overrides() -> dict:
    """Get configuration overrides for CI-safe execution."""
    if should_use_ci_mode():
        return {
            # Forecasting overrides
            "forecasting": {
                "max_epochs": 1,
                "batch_size": 4,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "patch_len": 8,
                "forecast_horizon": 8,
            },
            # Anomaly detection overrides
            "anomaly": {
                "max_epochs": 1,
                "batch_size": 4,
                "window_size": 16,
                "hidden_sizes": [8, 4],
            }
        }
    else:
        return {}
