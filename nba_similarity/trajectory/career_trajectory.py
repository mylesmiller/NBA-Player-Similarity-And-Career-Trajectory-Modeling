"""Career trajectory analysis module."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class CareerTrajectoryAnalyzer:
    """Analyzes player career trajectories and finds historical comps."""
    
    def __init__(self, early_career_years: int = 3):
        """Initialize trajectory analyzer.
        
        Args:
            early_career_years: Number of early career years to use for signature.
        """
        self.early_career_years = early_career_years
        self.trajectory_signatures = {}
        self.predictive_model = None
    
    def build_trajectory_signature(
        self, 
        player_stats: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """Build early-career trajectory signature from year-to-year deltas.
        
        Args:
            player_stats: DataFrame with player statistics across seasons.
            feature_cols: List of feature columns to use.
            
        Returns:
            Trajectory signature vector (deltas between consecutive years).
        """
        # Sort by season
        player_stats = player_stats.sort_values('season').reset_index(drop=True)
        
        # Take early career years
        early_career = player_stats.head(self.early_career_years)
        
        if len(early_career) < 2:
            # Not enough data, return zeros
            n_features = len(feature_cols)
            return np.zeros((self.early_career_years - 1) * n_features)
        
        # Compute year-to-year deltas
        deltas = []
        for i in range(1, len(early_career)):
            prev_features = early_career.iloc[i-1][feature_cols].values
            curr_features = early_career.iloc[i][feature_cols].values
            delta = curr_features - prev_features
            deltas.append(delta)
        
        # Pad if needed
        while len(deltas) < self.early_career_years - 1:
            n_features = len(feature_cols)
            deltas.append(np.zeros(n_features))
        
        # Flatten into signature
        signature = np.concatenate(deltas)
        
        return signature
    
    def build_all_signatures(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        """Build trajectory signatures for all players.
        
        Args:
            features_df: DataFrame with player features across seasons.
            feature_cols: List of feature columns to use.
            
        Returns:
            Dictionary mapping player_name to signature vector.
        """
        signatures = {}
        
        for player_name in features_df['player_name'].unique():
            player_stats = features_df[features_df['player_name'] == player_name]
            signature = self.build_trajectory_signature(player_stats, feature_cols)
            signatures[player_name] = signature
        
        self.trajectory_signatures = signatures
        logger.info(f"Built trajectory signatures for {len(signatures)} players")
        
        return signatures
    
    def find_trajectory_comps(
        self,
        player_name: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> pd.DataFrame:
        """Find historical trajectory comparisons for a player.
        
        Args:
            player_name: Name of player to find comps for.
            top_k: Number of comps to return.
            exclude_self: Whether to exclude the query player.
            
        Returns:
            DataFrame with similar trajectory players and similarity scores.
        """
        if not self.trajectory_signatures:
            raise ValueError("Trajectory signatures not built. Call build_all_signatures first.")
        
        if player_name not in self.trajectory_signatures:
            raise ValueError(f"Player '{player_name}' not found in trajectory signatures.")
        
        query_signature = self.trajectory_signatures[player_name]
        
        # Compute cosine similarity with all other signatures
        results = []
        for other_player, other_signature in self.trajectory_signatures.items():
            if exclude_self and other_player == player_name:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_signature, other_signature) / (
                np.linalg.norm(query_signature) * np.linalg.norm(other_signature) + 1e-8
            )
            
            results.append({
                'player_name': other_player,
                'trajectory_similarity': similarity
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('trajectory_similarity', ascending=False)
        results_df = results_df.head(top_k)
        results_df = results_df.reset_index(drop=True)
        
        return results_df
    
    def build_predictive_model(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'points_per36',
        years_ahead: int = 1
    ) -> Dict:
        """Build a simple predictive model for future performance.
        
        Args:
            features_df: DataFrame with player features.
            feature_cols: List of feature columns to use as inputs.
            target_col: Target column to predict.
            years_ahead: Number of years ahead to predict.
            
        Returns:
            Dictionary with model and evaluation metrics.
        """
        logger.info(f"Building predictive model for {target_col} {years_ahead} year(s) ahead")
        
        # Prepare training data: for each player-season, predict target_col N years ahead
        X_train = []
        y_train = []
        
        for player_name in features_df['player_name'].unique():
            player_stats = features_df[features_df['player_name'] == player_name].sort_values('season')
            
            for i in range(len(player_stats) - years_ahead):
                current_features = player_stats.iloc[i][feature_cols].values
                future_target = player_stats.iloc[i + years_ahead][target_col]
                
                if not np.isnan(future_target):
                    X_train.append(current_features)
                    y_train.append(future_target)
        
        if len(X_train) == 0:
            logger.warning("No training data available for predictive model")
            return {'model': None, 'mae': None, 'rmse': None}
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        self.predictive_model = model
        
        logger.info(f"Predictive model MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'years_ahead': years_ahead
        }
    
    def predict_future(
        self,
        player_features: np.ndarray,
        feature_cols: List[str]
    ) -> float:
        """Predict future performance for a player.
        
        Args:
            player_features: Feature vector for current season.
            feature_cols: List of feature columns (must match model).
            
        Returns:
            Predicted target value.
        """
        if self.predictive_model is None:
            raise ValueError("Predictive model not built. Call build_predictive_model first.")
        
        # Ensure correct order
        feature_vector = np.array([player_features[col] for col in feature_cols])
        prediction = self.predictive_model.predict([feature_vector])[0]
        
        return prediction

