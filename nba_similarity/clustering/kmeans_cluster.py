"""KMeans clustering module with auto-labeling."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """Clusters players using KMeans and auto-labels clusters."""
    
    def __init__(self, n_clusters: int = 8, random_seed: int = 42):
        """Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters.
            random_seed: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        self.fitted = False
        self.feature_names = []
        self.scaler = None
    
    def fit_predict(
        self, 
        features_df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        scaler: Optional[object] = None
    ) -> pd.DataFrame:
        """Fit KMeans and predict cluster assignments.
        
        Args:
            features_df: Standardized features DataFrame.
            feature_names: List of feature names (for labeling).
            scaler: Scaler object (for inverse transform of centroids).
            
        Returns:
            DataFrame with cluster assignments.
        """
        # Extract feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        X = features_df[feature_cols].values
        
        # Store for labeling
        self.feature_names = feature_names or feature_cols
        self.scaler = scaler
        
        logger.info(f"Fitting KMeans with {self.n_clusters} clusters on {X.shape[0]} samples")
        
        # Fit and predict
        cluster_labels = self.kmeans.fit_predict(X)
        self.fitted = True
        
        # Create results DataFrame
        results_df = features_df[['player_name', 'season']].copy()
        results_df['cluster'] = cluster_labels
        
        logger.info(f"Clustering complete. Cluster sizes: {pd.Series(cluster_labels).value_counts().to_dict()}")
        
        return results_df
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict cluster assignments for new data.
        
        Args:
            features_df: Standardized features DataFrame.
            
        Returns:
            DataFrame with cluster assignments.
        """
        if not self.fitted:
            raise ValueError("KMeans not fitted. Call fit_predict first.")
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        X = features_df[feature_cols].values
        
        cluster_labels = self.kmeans.predict(X)
        
        results_df = features_df[['player_name', 'season']].copy()
        results_df['cluster'] = cluster_labels
        
        return results_df
    
    def get_cluster_labels(self) -> Dict[int, str]:
        """Auto-label clusters based on centroid feature z-scores.
        
        Returns:
            Dictionary mapping cluster ID to label string.
        """
        if not self.fitted:
            raise ValueError("KMeans not fitted. Call fit_predict first.")
        
        # Get centroids
        centroids = self.kmeans.cluster_centers_
        
        # Centroids are in standardized space, use them directly for z-score analysis
        # (z-scores are already computed in standardized space)
        centroids_original = centroids
        
        cluster_labels = {}
        
        for cluster_id in range(self.n_clusters):
            centroid = centroids_original[cluster_id]
            
            # Compute z-scores (centroids are already in standardized space)
            # Find top features (highest absolute values)
            abs_centroid = np.abs(centroid)
            top_indices = np.argsort(abs_centroid)[-5:][::-1]  # Top 5 features
            
            # Build label from top features
            label_parts = []
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                value = centroid[idx]
                
                # Simplify feature names for labels
                feature_short = feature_name.replace('_per36', '').replace('_pct', '').replace('_rate', '')
                
                if abs(value) > 0.5:  # Significant feature
                    direction = "high" if value > 0 else "low"
                    label_parts.append(f"{direction}-{feature_short}")
            
            # Create label
            if label_parts:
                label = " / ".join(label_parts[:3])  # Top 3 features
            else:
                label = f"Cluster {cluster_id}"
            
            cluster_labels[cluster_id] = label
        
        return cluster_labels
    
    def get_cluster_summary(self, features_df: pd.DataFrame, cluster_assignments: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for each cluster.
        
        Args:
            features_df: Original features DataFrame.
            cluster_assignments: DataFrame with cluster assignments.
            
        Returns:
            DataFrame with cluster summaries.
        """
        # Merge features with cluster assignments
        merged = cluster_assignments.merge(
            features_df, 
            on=['player_name', 'season'], 
            how='left'
        )
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_name', 'season']]
        
        summaries = []
        for cluster_id in range(self.n_clusters):
            cluster_data = merged[merged['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            summary = {
                'cluster': cluster_id,
                'n_players': len(cluster_data),
                'label': self.get_cluster_labels().get(cluster_id, f"Cluster {cluster_id}")
            }
            
            # Add mean values for key features
            for col in feature_cols[:10]:  # Top 10 features
                summary[f'mean_{col}'] = cluster_data[col].mean()
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)

