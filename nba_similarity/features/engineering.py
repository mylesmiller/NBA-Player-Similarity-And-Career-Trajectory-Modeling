"""Feature engineering module."""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers playstyle features from raw NBA statistics."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        self.scaler = None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features from raw stats.
        
        Args:
            df: DataFrame with raw NBA statistics.
            
        Returns:
            DataFrame with engineered features.
        """
        logger.info("Engineering features...")
        
        # Create a copy to avoid modifying original
        features_df = df[['player_name', 'season']].copy()
        
        # Per-36 minute stats
        per36_features = self._compute_per36(df)
        features_df = pd.concat([features_df, per36_features], axis=1)
        
        # Shot profile rates
        shot_profile = self._compute_shot_profile(df)
        features_df = pd.concat([features_df, shot_profile], axis=1)
        
        # Efficiency proxies
        efficiency = self._compute_efficiency_proxies(df)
        features_df = pd.concat([features_df, efficiency], axis=1)
        
        # Store feature names (excluding player_name and season)
        self.feature_names = [col for col in features_df.columns 
                             if col not in ['player_name', 'season']]
        
        logger.info(f"Engineered {len(self.feature_names)} features")
        return features_df
    
    def _compute_per36(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-36 minute statistics.
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            DataFrame with per-36 features.
        """
        per36 = pd.DataFrame()
        
        # Calculate per-36 rates
        minutes = df['minutes'].values
        # Avoid division by zero
        minutes = np.maximum(minutes, 1)
        
        stats_to_normalize = [
            'points', 'field_goals', 'field_goal_attempts',
            'three_pointers', 'three_point_attempts',
            'two_pointers', 'two_point_attempts',
            'free_throws', 'free_throw_attempts',
            'total_rebounds', 'offensive_rebounds', 'defensive_rebounds',
            'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls'
        ]
        
        for stat in stats_to_normalize:
            if stat in df.columns:
                per36[f'{stat}_per36'] = (df[stat].values / minutes) * 36
        
        return per36
    
    def _compute_shot_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute shot profile rates.
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            DataFrame with shot profile features.
        """
        shot_profile = pd.DataFrame()
        
        # Shot distribution rates
        fga = df['field_goal_attempts'].values
        fga = np.maximum(fga, 1)  # Avoid division by zero
        
        if 'three_point_attempts' in df.columns:
            shot_profile['three_point_rate'] = (
                df['three_point_attempts'].values / fga
            )
        
        if 'two_point_attempts' in df.columns:
            shot_profile['two_point_rate'] = (
                df['two_point_attempts'].values / fga
            )
        
        # Free throw rate
        if 'free_throw_attempts' in df.columns:
            shot_profile['ft_rate'] = (
                df['free_throw_attempts'].values / fga
            )
        
        # Shot selection: mid-range vs paint (approximation)
        if 'two_point_attempts' in df.columns:
            # Approximate paint shots as high-percentage 2PA
            two_pct = df['two_point_pct'].values
            shot_profile['paint_shot_rate'] = np.where(
                two_pct > 0.55,  # High 2P% suggests paint shots
                df['two_point_attempts'].values / fga * 0.6,
                0.0
            )
            shot_profile['mid_range_rate'] = (
                df['two_point_attempts'].values / fga - 
                shot_profile['paint_shot_rate']
            )
        
        return shot_profile
    
    def _compute_efficiency_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute efficiency proxy metrics.
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            DataFrame with efficiency features.
        """
        efficiency = pd.DataFrame()
        
        # True shooting percentage (approximation)
        fga = df['field_goal_attempts'].values
        fta = df['free_throw_attempts'].values
        points = df['points'].values
        
        ts_denominator = fga + 0.44 * fta
        ts_denominator = np.maximum(ts_denominator, 1)
        efficiency['ts_pct'] = points / (2 * ts_denominator)
        
        # Effective field goal percentage
        fgm = df['field_goals'].values
        three_made = df['three_pointers'].values if 'three_pointers' in df.columns else 0
        efg_denominator = np.maximum(fga, 1)
        efficiency['efg_pct'] = (fgm + 0.5 * three_made) / efg_denominator
        
        # Usage rate approximation (points + assists + turnovers per possession)
        assists = df['assists'].values
        turnovers = df['turnovers'].values
        possessions = fga + turnovers + 0.44 * fta
        possessions = np.maximum(possessions, 1)
        efficiency['usage_rate'] = (points + assists + turnovers) / possessions
        
        # Assist-to-turnover ratio
        turnovers_safe = np.maximum(turnovers, 1)
        efficiency['ast_to_ratio'] = assists / turnovers_safe
        
        # Rebound rate approximation (rebounds per game)
        if 'total_rebounds' in df.columns:
            games = df['games'].values
            games = np.maximum(games, 1)
            efficiency['reb_rate'] = df['total_rebounds'].values / games
        
        # Block and steal rates
        if 'blocks' in df.columns:
            efficiency['block_rate'] = df['blocks'].values / np.maximum(fga, 1)
        if 'steals' in df.columns:
            efficiency['steal_rate'] = df['steals'].values / np.maximum(fga, 1)
        
        return efficiency
    
    def standardize_features(
        self, 
        features_df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, object]:
        """Standardize features using StandardScaler.
        
        Args:
            features_df: DataFrame with engineered features.
            fit: Whether to fit the scaler (True) or use existing (False).
            
        Returns:
            Tuple of (standardized DataFrame, scaler object).
        """
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        
        if fit:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(features_df[feature_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            scaled_data = self.scaler.transform(features_df[feature_cols])
        
        scaled_df = pd.DataFrame(
            scaled_data,
            columns=feature_cols,
            index=features_df.index
        )
        
        # Add back metadata columns
        scaled_df = pd.concat([
            features_df[['player_name', 'season']],
            scaled_df
        ], axis=1)
        
        return scaled_df, self.scaler

