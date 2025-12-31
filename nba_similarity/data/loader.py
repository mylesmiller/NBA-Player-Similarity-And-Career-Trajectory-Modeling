"""Data loading and fetching module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NBADataLoader:
    """Loads NBA player-season statistics from CSV or fetches from API."""
    
    # Expected schema for NBA player-season stats
    REQUIRED_COLUMNS = [
        'player_name', 'season', 'team', 'age', 'games', 'games_started',
        'minutes', 'field_goals', 'field_goal_attempts', 'field_goal_pct',
        'three_pointers', 'three_point_attempts', 'three_point_pct',
        'two_pointers', 'two_point_attempts', 'two_point_pct',
        'free_throws', 'free_throw_attempts', 'free_throw_pct',
        'offensive_rebounds', 'defensive_rebounds', 'total_rebounds',
        'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls', 'points'
    ]
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV files. If None, uses default.
        """
        from nba_similarity.utils.config import DATA_RAW
        
        self.data_dir = data_dir or DATA_RAW
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_csv(self, filepath: Optional[Path] = None, auto_transform: bool = True) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filepath: Path to CSV file. If None, looks for 'nba_stats.csv' in data_dir.
            auto_transform: If True, attempts to auto-detect and transform draft format.
            
        Returns:
            DataFrame with player-season statistics.
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist.
            ValueError: If schema validation fails.
        """
        if filepath is None:
            filepath = self.data_dir / "nba_stats.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Try to auto-detect draft format and transform
        if auto_transform:
            draft_columns = ['player', 'year', 'minutes_played', 'field_goal_percentage']
            if all(col in df.columns for col in draft_columns):
                logger.info("Detected draft data format. Transforming...")
                from nba_similarity.data.adapter import DraftDataAdapter
                adapter = DraftDataAdapter()
                df = adapter.transform_draft_data(df)
            else:
                # Try to validate standard format
                try:
                    self._validate_schema(df)
                except ValueError:
                    logger.warning("Schema validation failed. Attempting to continue with available columns.")
        else:
            # Validate schema
            self._validate_schema(df)
        
        logger.info(f"Loaded {len(df)} player-season records")
        return df
    
    def fetch_data(self, seasons: Optional[list] = None, save: bool = True) -> pd.DataFrame:
        """Fetch NBA data from API (placeholder - can be extended with nba_api).
        
        Args:
            seasons: List of seasons to fetch (e.g., ['2022-23', '2023-24']). 
                     If None, fetches current season.
            save: Whether to save fetched data to CSV.
            
        Returns:
            DataFrame with player-season statistics.
        """
        logger.warning("fetch_data is a placeholder. Implement with nba_api or other source.")
        logger.info("For now, please provide a CSV file with NBA statistics.")
        
        # Placeholder: return empty DataFrame with correct schema
        df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate.
            
        Raises:
            ValueError: If required columns are missing.
        """
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        min_minutes: int = 500,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """Preprocess data: filter low-minute players and handle missing values.
        
        Args:
            df: Raw DataFrame.
            min_minutes: Minimum total minutes per season to include player.
            fill_missing: Whether to fill missing values.
            
        Returns:
            Preprocessed DataFrame.
        """
        logger.info(f"Preprocessing {len(df)} records")
        
        # Filter by minimum minutes
        if 'minutes' in df.columns:
            initial_count = len(df)
            df = df[df['minutes'] >= min_minutes].copy()
            logger.info(f"Filtered {initial_count - len(df)} records below {min_minutes} minutes")
        
        # Handle missing values
        if fill_missing:
            # Fill percentage columns with 0 if attempts are 0
            pct_cols = [col for col in df.columns if col.endswith('_pct')]
            for col in pct_cols:
                df[col] = df[col].fillna(0.0)
            
            # Fill count columns with 0
            count_cols = [
                'field_goals', 'field_goal_attempts', 'three_pointers',
                'three_point_attempts', 'two_pointers', 'two_point_attempts',
                'free_throws', 'free_throw_attempts', 'offensive_rebounds',
                'defensive_rebounds', 'assists', 'steals', 'blocks',
                'turnovers', 'personal_fouls', 'points'
            ]
            for col in count_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            
            # Fill other numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        logger.info(f"Preprocessed data: {len(df)} records remaining")
        return df

