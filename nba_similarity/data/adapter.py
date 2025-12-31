"""Data adapter for converting different CSV formats to standard format."""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DraftDataAdapter:
    """Adapter for NBA draft data CSV format."""
    
    def __init__(self):
        """Initialize adapter."""
        pass
    
    def transform_draft_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform draft data CSV to standard format.
        
        This adapter handles the nbaplayersdraft.csv format which contains
        career totals. It converts them to per-season estimates.
        
        Args:
            df: DataFrame with draft data format.
            
        Returns:
            DataFrame in standard format.
        """
        logger.info("Transforming draft data to standard format...")
        
        # Create new DataFrame with standard columns
        transformed = pd.DataFrame()
        
        # Map player name
        transformed['player_name'] = df['player']
        
        # Map season (use year as season, format as YYYY-YY)
        transformed['season'] = df['year'].apply(
            lambda y: f"{int(y)}-{str(int(y)+1)[-2:]}" if pd.notna(y) else None
        )
        
        # Map team
        transformed['team'] = df['team']
        
        # Age (estimate from draft year - assume 20-22 age range)
        # We'll use a default since we don't have exact age
        transformed['age'] = 21  # Default age
        
        # Games
        transformed['games'] = df['games'].fillna(0)
        transformed['games_started'] = (df['games'] * 0.6).fillna(0).astype(int)  # Estimate
        
        # Minutes
        transformed['minutes'] = df['minutes_played'].fillna(0)
        
        # Calculate field goals from percentage and attempts
        # We need to estimate attempts from points and percentages
        points = df['points'].fillna(0)
        fg_pct = df['field_goal_percentage'].fillna(0)
        three_pct = df['3_point_percentage'].fillna(0)
        ft_pct = df['free_throw_percentage'].fillna(0)
        
        # Estimate field goal attempts from points per game and percentages
        # This is an approximation
        ppg = df['points_per_game'].fillna(0)
        avg_min = df['average_minutes_played'].fillna(0)
        
        # Use averages to estimate totals per season
        games = transformed['games']
        games = games.replace(0, 1)  # Avoid division by zero
        
        # Estimate field goal attempts (rough approximation)
        # Points = 2*2PA*2P% + 3*3PA*3P% + FTA*FT%
        # We'll use a simplified approach
        fg_pct_safe = fg_pct.replace(0, 0.4)  # Default to 40% if 0
        fga_estimate = (points / (fg_pct_safe + 0.5)).fillna(0)
        fga_estimate = np.maximum(fga_estimate, points / 2)  # At least points/2
        fga_estimate = fga_estimate.replace([np.inf, -np.inf], 0).fillna(0)
        
        transformed['field_goal_attempts'] = fga_estimate.astype(int)
        transformed['field_goals'] = (fga_estimate * fg_pct).fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        transformed['field_goal_pct'] = fg_pct.fillna(0)
        
        # Three-pointers
        # Estimate 3PA as fraction of FGA
        three_rate = 0.3  # Assume 30% of shots are 3s
        three_attempts = (fga_estimate * three_rate).fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        transformed['three_point_attempts'] = three_attempts
        transformed['three_pointers'] = (three_attempts * three_pct).fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        transformed['three_point_pct'] = three_pct.fillna(0)
        
        # Two-pointers
        two_attempts = (fga_estimate - three_attempts).fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        transformed['two_point_attempts'] = two_attempts
        two_made = transformed['field_goals'] - transformed['three_pointers']
        transformed['two_pointers'] = np.maximum(two_made, 0).astype(int)
        two_pct_calc = np.where(
            two_attempts > 0,
            transformed['two_pointers'] / two_attempts.replace(0, 1),
            0.0
        )
        transformed['two_point_pct'] = pd.Series(two_pct_calc, index=transformed.index).fillna(0)
        
        # Free throws
        # Estimate FTA from points and percentages
        ft_attempts_estimate = ((points - transformed['field_goals'] * 2 - transformed['three_pointers']) / ft_pct.replace(0, 1)).fillna(0)
        ft_attempts_estimate = np.maximum(ft_attempts_estimate, 0)
        ft_attempts_estimate = ft_attempts_estimate.replace([np.inf, -np.inf], 0).fillna(0)
        transformed['free_throw_attempts'] = ft_attempts_estimate.astype(int)
        transformed['free_throws'] = (ft_attempts_estimate * ft_pct).fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        transformed['free_throw_pct'] = ft_pct.fillna(0)
        
        # Rebounds
        total_reb = df['total_rebounds'].fillna(0)
        # Estimate offensive/defensive split (typically 25% offensive, 75% defensive)
        transformed['offensive_rebounds'] = (total_reb * 0.25).fillna(0).astype(int)
        transformed['defensive_rebounds'] = (total_reb * 0.75).fillna(0).astype(int)
        transformed['total_rebounds'] = total_reb.astype(int)
        
        # Assists, steals, blocks
        transformed['assists'] = df['assists'].fillna(0).astype(int)
        
        # Estimate steals and blocks from available data
        # Use league averages as estimates
        transformed['steals'] = (games * 1.0).fillna(0).astype(int)  # ~1 steal per game
        transformed['blocks'] = (games * 0.5).fillna(0).astype(int)  # ~0.5 blocks per game
        
        # Turnovers and fouls (estimates)
        transformed['turnovers'] = (games * 2.0).fillna(0).astype(int)  # ~2 TO per game
        transformed['personal_fouls'] = (games * 2.5).fillna(0).astype(int)  # ~2.5 fouls per game
        
        # Points
        transformed['points'] = points.astype(int)
        
        # Filter out rows with no minutes or games
        transformed = transformed[
            (transformed['minutes'] > 0) & (transformed['games'] > 0)
        ].copy()
        
        logger.info(f"Transformed {len(transformed)} player records")
        
        return transformed


def load_and_transform_draft_csv(filepath: str) -> pd.DataFrame:
    """Load draft CSV and transform to standard format.
    
    Args:
        filepath: Path to draft CSV file.
        
    Returns:
        Transformed DataFrame in standard format.
    """
    logger.info(f"Loading draft data from {filepath}")
    df = pd.read_csv(filepath)
    
    adapter = DraftDataAdapter()
    transformed = adapter.transform_draft_data(df)
    
    return transformed

