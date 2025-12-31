"""Evaluation metrics module."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Computes evaluation metrics for the similarity and clustering models."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_similarity(
        self,
        similarity_searcher,
        features_df: pd.DataFrame,
        n_samples: int = 100
    ) -> Dict:
        """Evaluate similarity search quality.
        
        Args:
            similarity_searcher: SimilaritySearcher instance.
            features_df: Original features DataFrame.
            n_samples: Number of random samples for comparison.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating similarity search quality...")
        
        # Sample random players
        all_players = similarity_searcher.embedding_df['player_name'].unique()
        if len(all_players) < n_samples:
            sample_players = all_players
        else:
            np.random.seed(42)
            sample_players = np.random.choice(all_players, n_samples, replace=False)
        
        # Compute average feature distance for similar players vs random
        similar_distances = []
        random_distances = []
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        
        for player_name in sample_players:
            try:
                # Get similar players
                similar = similarity_searcher.find_similar_players(
                    player_name, top_k=5, exclude_self=True
                )
                
                # Get query player features
                query_features = features_df[
                    features_df['player_name'] == player_name
                ][feature_cols].iloc[0].values
                
                # Compute distances to similar players
                for similar_player in similar['player_name'].head(3):  # Top 3
                    similar_features = features_df[
                        features_df['player_name'] == similar_player
                    ][feature_cols].iloc[0].values
                    distance = np.linalg.norm(query_features - similar_features)
                    similar_distances.append(distance)
                
                # Compute distances to random players
                other_players = [p for p in all_players if p != player_name]
                np.random.seed(42)
                random_players = np.random.choice(other_players, 3, replace=False)
                
                for random_player in random_players:
                    random_features = features_df[
                        features_df['player_name'] == random_player
                    ][feature_cols].iloc[0].values
                    distance = np.linalg.norm(query_features - random_features)
                    random_distances.append(distance)
                    
            except Exception as e:
                logger.warning(f"Error evaluating {player_name}: {e}")
                continue
        
        avg_similar_distance = np.mean(similar_distances) if similar_distances else None
        avg_random_distance = np.mean(random_distances) if random_distances else None
        
        # Improvement ratio (lower is better for distances, so we want similar < random)
        improvement_ratio = (
            (avg_random_distance / avg_similar_distance) 
            if (avg_similar_distance and avg_random_distance and avg_similar_distance > 0)
            else None
        )
        
        results = {
            'avg_similar_feature_distance': avg_similar_distance,
            'avg_random_feature_distance': avg_random_distance,
            'improvement_ratio': improvement_ratio,
            'n_samples': len(sample_players)
        }
        
        logger.info(f"Similarity evaluation: improvement_ratio = {improvement_ratio:.4f}")
        
        return results
    
    def evaluate_clustering(
        self,
        features_df: pd.DataFrame,
        cluster_assignments: pd.DataFrame
    ) -> Dict:
        """Evaluate clustering quality.
        
        Args:
            features_df: Features DataFrame.
            cluster_assignments: DataFrame with cluster assignments.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating clustering quality...")
        
        # Merge features with cluster assignments
        merged = cluster_assignments.merge(
            features_df,
            on=['player_name', 'season'],
            how='inner'
        )
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        X = merged[feature_cols].values
        labels = merged['cluster'].values
        
        # Compute silhouette score
        silhouette = silhouette_score(X, labels)
        
        # Cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        
        results = {
            'silhouette_score': silhouette,
            'n_clusters': len(cluster_sizes),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes.values()) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0
        }
        
        logger.info(f"Clustering evaluation: silhouette_score = {silhouette:.4f}")
        
        return results
    
    def evaluate_pca(
        self,
        pca_embedder,
        which: str = '20d'
    ) -> Dict:
        """Evaluate PCA embedding quality.
        
        Args:
            pca_embedder: PCAEmbedder instance.
            which: Which PCA to evaluate ('20d' or '2d').
            
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info(f"Evaluating PCA {which} quality...")
        
        if which == '20d':
            pca = pca_embedder.pca_20d
            if not pca_embedder.fitted_20d:
                raise ValueError("20D PCA not fitted.")
        else:
            pca = pca_embedder.pca_2d
            if not pca_embedder.fitted_2d:
                raise ValueError("2D PCA not fitted.")
        
        explained_variance_ratio = pca.explained_variance_ratio_
        
        results = {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_explained_variance': np.cumsum(explained_variance_ratio).tolist(),
            'total_explained_variance': explained_variance_ratio.sum(),
            'n_components': pca.n_components
        }
        
        logger.info(f"PCA {which} evaluation: total_explained_variance = {results['total_explained_variance']:.4f}")
        
        return results

