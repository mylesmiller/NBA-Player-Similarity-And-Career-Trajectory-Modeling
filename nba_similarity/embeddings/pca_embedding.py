"""PCA embedding generation module."""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class PCAEmbedder:
    """Generates PCA embeddings for player similarity and visualization."""
    
    def __init__(self, n_components_20d: int = 20, n_components_2d: int = 2, random_seed: int = 42):
        """Initialize PCA embedder.
        
        Args:
            n_components_20d: Number of components for similarity embedding.
            n_components_2d: Number of components for visualization embedding.
            random_seed: Random seed for reproducibility.
        """
        self.n_components_20d = n_components_20d
        self.n_components_2d = n_components_2d
        self.random_seed = random_seed
        
        self.pca_20d = PCA(n_components=n_components_20d, random_state=random_seed)
        self.pca_2d = PCA(n_components=n_components_2d, random_state=random_seed)
        
        self.fitted_20d = False
        self.fitted_2d = False
    
    def fit_transform(
        self, 
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit PCA models and transform features to embeddings.
        
        Args:
            features_df: Standardized features DataFrame.
            
        Returns:
            Tuple of (20D embedding DataFrame, 2D embedding DataFrame).
        """
        # Extract feature columns (exclude metadata)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        X = features_df[feature_cols].values
        
        logger.info(f"Fitting PCA models on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Fit and transform to 20D
        embedding_20d = self.pca_20d.fit_transform(X)
        self.fitted_20d = True
        
        # Fit and transform to 2D
        embedding_2d = self.pca_2d.fit_transform(X)
        self.fitted_2d = True
        
        # Create DataFrames
        embedding_20d_df = pd.DataFrame(
            embedding_20d,
            columns=[f'pc_{i+1}' for i in range(self.n_components_20d)],
            index=features_df.index
        )
        embedding_20d_df = pd.concat([
            features_df[['player_name', 'season']],
            embedding_20d_df
        ], axis=1)
        
        embedding_2d_df = pd.DataFrame(
            embedding_2d,
            columns=['pc1', 'pc2'],
            index=features_df.index
        )
        embedding_2d_df = pd.concat([
            features_df[['player_name', 'season']],
            embedding_2d_df
        ], axis=1)
        
        logger.info(
            f"20D PCA explained variance: {self.pca_20d.explained_variance_ratio_.sum():.4f}"
        )
        logger.info(
            f"2D PCA explained variance: {self.pca_2d.explained_variance_ratio_.sum():.4f}"
        )
        
        return embedding_20d_df, embedding_2d_df
    
    def transform(
        self, 
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform features to embeddings using fitted models.
        
        Args:
            features_df: Standardized features DataFrame.
            
        Returns:
            Tuple of (20D embedding DataFrame, 2D embedding DataFrame).
        """
        if not (self.fitted_20d and self.fitted_2d):
            raise ValueError("PCA models not fitted. Call fit_transform first.")
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        X = features_df[feature_cols].values
        
        # Transform to 20D
        embedding_20d = self.pca_20d.transform(X)
        
        # Transform to 2D
        embedding_2d = self.pca_2d.transform(X)
        
        # Create DataFrames
        embedding_20d_df = pd.DataFrame(
            embedding_20d,
            columns=[f'pc_{i+1}' for i in range(self.n_components_20d)],
            index=features_df.index
        )
        embedding_20d_df = pd.concat([
            features_df[['player_name', 'season']],
            embedding_20d_df
        ], axis=1)
        
        embedding_2d_df = pd.DataFrame(
            embedding_2d,
            columns=['pc1', 'pc2'],
            index=features_df.index
        )
        embedding_2d_df = pd.concat([
            features_df[['player_name', 'season']],
            embedding_2d_df
        ], axis=1)
        
        return embedding_20d_df, embedding_2d_df
    
    def save(self, filepath: Path) -> None:
        """Save PCA models to disk.
        
        Args:
            filepath: Path to save models.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pca_20d': self.pca_20d,
                'pca_2d': self.pca_2d,
                'fitted_20d': self.fitted_20d,
                'fitted_2d': self.fitted_2d,
                'n_components_20d': self.n_components_20d,
                'n_components_2d': self.n_components_2d,
                'random_seed': self.random_seed
            }, f)
        logger.info(f"Saved PCA models to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'PCAEmbedder':
        """Load PCA models from disk.
        
        Args:
            filepath: Path to load models from.
            
        Returns:
            PCAEmbedder instance with loaded models.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        embedder = cls(
            n_components_20d=data['n_components_20d'],
            n_components_2d=data['n_components_2d'],
            random_seed=data['random_seed']
        )
        embedder.pca_20d = data['pca_20d']
        embedder.pca_2d = data['pca_2d']
        embedder.fitted_20d = data['fitted_20d']
        embedder.fitted_2d = data['fitted_2d']
        
        logger.info(f"Loaded PCA models from {filepath}")
        return embedder

