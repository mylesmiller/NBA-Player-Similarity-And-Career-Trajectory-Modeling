"""Main pipeline for NBA similarity analysis."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import json

from nba_similarity.data.loader import NBADataLoader
from nba_similarity.features.engineering import FeatureEngineer
from nba_similarity.embeddings.pca_embedding import PCAEmbedder
from nba_similarity.similarity.search import SimilaritySearcher
from nba_similarity.clustering.kmeans_cluster import KMeansClusterer
from nba_similarity.trajectory.career_trajectory import CareerTrajectoryAnalyzer
from nba_similarity.evaluation.metrics import Evaluator
from nba_similarity.utils.config import (
    DATA_RAW, DATA_PROCESSED, ARTIFACTS_DIR, RANDOM_SEED,
    MIN_MINUTES_THRESHOLD, PCA_DIM_20, PCA_DIM_2, DEFAULT_N_CLUSTERS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBASimilarityPipeline:
    """End-to-end pipeline for NBA player similarity analysis."""
    
    def __init__(self, data_dir: Path = None, artifacts_dir: Path = None):
        """Initialize pipeline.
        
        Args:
            data_dir: Directory for data files.
            artifacts_dir: Directory for saved artifacts.
        """
        self.data_dir = data_dir or DATA_RAW
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        
        # Initialize components
        self.loader = NBADataLoader(data_dir=self.data_dir)
        self.feature_engineer = FeatureEngineer()
        self.pca_embedder = PCAEmbedder(
            n_components_20d=PCA_DIM_20,
            n_components_2d=PCA_DIM_2,
            random_seed=RANDOM_SEED
        )
        self.clusterer = None
        self.similarity_searcher = None
        self.trajectory_analyzer = CareerTrajectoryAnalyzer()
        self.evaluator = Evaluator()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features_df = None
        self.scaled_features_df = None
        self.embedding_20d = None
        self.embedding_2d = None
        self.cluster_assignments = None
    
    def preprocess(self, csv_file: Path = None, fetch_data: bool = False) -> pd.DataFrame:
        """Preprocess data: load, validate, and clean.
        
        Args:
            csv_file: Path to CSV file (if None, uses default).
            fetch_data: Whether to fetch data from API (placeholder).
            
        Returns:
            Preprocessed DataFrame.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Data Preprocessing")
        logger.info("=" * 60)
        
        if fetch_data:
            logger.info("Fetching data from API...")
            self.raw_data = self.loader.fetch_data()
        else:
            logger.info("Loading data from CSV...")
            self.raw_data = self.loader.load_from_csv(csv_file)
        
        # Preprocess
        self.processed_data = self.loader.preprocess(
            self.raw_data,
            min_minutes=MIN_MINUTES_THRESHOLD,
            fill_missing=True
        )
        
        # Save processed data
        processed_path = DATA_PROCESSED / "processed_stats.csv"
        self.processed_data.to_csv(processed_path, index=False)
        logger.info(f"Saved processed data to {processed_path}")
        
        return self.processed_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features from processed data.
        
        Returns:
            DataFrame with engineered features.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 60)
        
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        # Engineer features
        self.features_df = self.feature_engineer.engineer_features(self.processed_data)
        
        # Standardize
        self.scaled_features_df, scaler = self.feature_engineer.standardize_features(
            self.features_df,
            fit=True
        )
        
        # Save features
        features_path = DATA_PROCESSED / "engineered_features.csv"
        self.features_df.to_csv(features_path, index=False)
        logger.info(f"Saved engineered features to {features_path}")
        
        # Save scaled features
        scaled_features_path = DATA_PROCESSED / "scaled_features.csv"
        self.scaled_features_df.to_csv(scaled_features_path, index=False)
        logger.info(f"Saved scaled features to {scaled_features_path}")
        
        # Save scaler
        scaler_path = self.artifacts_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
        
        return self.scaled_features_df
    
    def train_embeddings(self) -> tuple:
        """Train PCA embeddings.
        
        Returns:
            Tuple of (20D embedding, 2D embedding).
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Training Embeddings")
        logger.info("=" * 60)
        
        if self.scaled_features_df is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        # Fit and transform
        self.embedding_20d, self.embedding_2d = self.pca_embedder.fit_transform(
            self.scaled_features_df
        )
        
        # Save embeddings
        embedding_20d_path = DATA_PROCESSED / "embedding_20d.csv"
        embedding_2d_path = DATA_PROCESSED / "embedding_2d.csv"
        self.embedding_20d.to_csv(embedding_20d_path, index=False)
        self.embedding_2d.to_csv(embedding_2d_path, index=False)
        logger.info(f"Saved embeddings to {embedding_20d_path} and {embedding_2d_path}")
        
        # Save PCA models
        pca_path = self.artifacts_dir / "pca_models.pkl"
        self.pca_embedder.save(pca_path)
        
        # Initialize similarity searcher
        self.similarity_searcher = SimilaritySearcher(
            self.embedding_20d,
            random_seed=RANDOM_SEED
        )
        
        return self.embedding_20d, self.embedding_2d
    
    def cluster_players(self, n_clusters: int = None) -> pd.DataFrame:
        """Cluster players using KMeans.
        
        Args:
            n_clusters: Number of clusters (uses default if None).
            
        Returns:
            DataFrame with cluster assignments.
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Clustering Players")
        logger.info("=" * 60)
        
        if self.scaled_features_df is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        n_clusters = n_clusters or DEFAULT_N_CLUSTERS
        
        # Initialize clusterer
        self.clusterer = KMeansClusterer(
            n_clusters=n_clusters,
            random_seed=RANDOM_SEED
        )
        
        # Fit and predict
        self.cluster_assignments = self.clusterer.fit_predict(
            self.scaled_features_df,
            feature_names=self.feature_engineer.feature_names,
            scaler=self.feature_engineer.scaler
        )
        
        # Save cluster assignments
        cluster_path = DATA_PROCESSED / "cluster_assignments.csv"
        self.cluster_assignments.to_csv(cluster_path, index=False)
        logger.info(f"Saved cluster assignments to {cluster_path}")
        
        # Get cluster labels
        cluster_labels = self.clusterer.get_cluster_labels()
        logger.info("Cluster labels:")
        for cluster_id, label in cluster_labels.items():
            logger.info(f"  Cluster {cluster_id}: {label}")
        
        return self.cluster_assignments
    
    def build_trajectories(self) -> dict:
        """Build career trajectory signatures.
        
        Returns:
            Dictionary of trajectory signatures.
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Building Career Trajectories")
        logger.info("=" * 60)
        
        if self.features_df is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        feature_cols = self.feature_engineer.feature_names
        
        # Build all signatures
        signatures = self.trajectory_analyzer.build_all_signatures(
            self.features_df,
            feature_cols
        )
        
        # Save signatures
        signatures_path = self.artifacts_dir / "trajectory_signatures.pkl"
        with open(signatures_path, 'wb') as f:
            pickle.dump(signatures, f)
        logger.info(f"Saved trajectory signatures to {signatures_path}")
        
        return signatures
    
    def evaluate(self) -> dict:
        """Run evaluation metrics.
        
        Returns:
            Dictionary with all evaluation results.
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Evaluation")
        logger.info("=" * 60)
        
        results = {}
        
        # Evaluate similarity
        if self.similarity_searcher is not None and self.features_df is not None:
            similarity_results = self.evaluator.evaluate_similarity(
                self.similarity_searcher,
                self.features_df
            )
            results['similarity'] = similarity_results
        
        # Evaluate clustering
        if self.cluster_assignments is not None and self.scaled_features_df is not None:
            clustering_results = self.evaluator.evaluate_clustering(
                self.scaled_features_df,
                self.cluster_assignments
            )
            results['clustering'] = clustering_results
        
        # Evaluate PCA
        if self.pca_embedder.fitted_20d:
            pca_20d_results = self.evaluator.evaluate_pca(self.pca_embedder, '20d')
            results['pca_20d'] = pca_20d_results
        
        if self.pca_embedder.fitted_2d:
            pca_2d_results = self.evaluator.evaluate_pca(self.pca_embedder, '2d')
            results['pca_2d'] = pca_2d_results
        
        # Save evaluation results
        eval_path = self.artifacts_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved evaluation results to {eval_path}")
        
        return results
    
    def run_full_pipeline(
        self,
        csv_file: Path = None,
        fetch_data: bool = False,
        n_clusters: int = None
    ) -> dict:
        """Run the complete pipeline.
        
        Args:
            csv_file: Path to CSV file.
            fetch_data: Whether to fetch data.
            n_clusters: Number of clusters.
            
        Returns:
            Dictionary with all results.
        """
        logger.info("=" * 60)
        logger.info("NBA SIMILARITY PIPELINE - FULL RUN")
        logger.info("=" * 60)
        
        # Run all steps
        self.preprocess(csv_file=csv_file, fetch_data=fetch_data)
        self.engineer_features()
        self.train_embeddings()
        self.cluster_players(n_clusters=n_clusters)
        self.build_trajectories()
        results = self.evaluate()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return results

