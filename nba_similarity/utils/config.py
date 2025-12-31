"""Configuration and constants."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Random seeds for reproducibility
RANDOM_SEED = 42

# Minimum minutes threshold for filtering players
MIN_MINUTES_THRESHOLD = 500  # per season

# PCA dimensions
PCA_DIM_20 = 20
PCA_DIM_2 = 2

# Default number of clusters
DEFAULT_N_CLUSTERS = 8

# Default top-K for similarity search
DEFAULT_TOP_K = 10

