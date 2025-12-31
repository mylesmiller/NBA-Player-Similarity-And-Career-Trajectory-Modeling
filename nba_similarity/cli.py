"""Command-line interface for NBA similarity pipeline."""

import argparse
import logging
from pathlib import Path
import sys

from nba_similarity.pipeline import NBASimilarityPipeline
from nba_similarity.utils.config import DATA_RAW, ARTIFACTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_command(args):
    """Preprocess data command."""
    pipeline = NBASimilarityPipeline()
    
    csv_file = Path(args.csv_file) if args.csv_file else None
    pipeline.preprocess(csv_file=csv_file, fetch_data=args.fetch)
    
    logger.info("Preprocessing complete!")


def train_command(args):
    """Train embeddings command."""
    pipeline = NBASimilarityPipeline()
    
    # Load processed data if exists, otherwise preprocess
    processed_path = Path("data/processed/processed_stats.csv")
    if not processed_path.exists():
        logger.info("Processed data not found. Running preprocessing...")
        csv_file = Path(args.csv_file) if args.csv_file else None
        pipeline.preprocess(csv_file=csv_file, fetch_data=False)
    else:
        # Load processed data
        logger.info("Loading processed data...")
        import pandas as pd
        pipeline.processed_data = pd.read_csv(processed_path)
        logger.info(f"Loaded {len(pipeline.processed_data)} processed records")
    
    # Engineer features
    pipeline.engineer_features()
    
    # Train embeddings
    pipeline.train_embeddings()
    
    # Cluster players
    n_clusters = args.n_clusters if args.n_clusters else None
    pipeline.cluster_players(n_clusters=n_clusters)
    
    # Build trajectories
    pipeline.build_trajectories()
    
    logger.info("Training complete!")


def evaluate_command(args):
    """Evaluate models command."""
    pipeline = NBASimilarityPipeline()
    
    # Load artifacts
    try:
        # Load embeddings
        embedding_20d_path = Path("data/processed/embedding_20d.csv")
        embedding_2d_path = Path("data/processed/embedding_2d.csv")
        
        if not embedding_20d_path.exists():
            logger.error("Embeddings not found. Run 'train' command first.")
            sys.exit(1)
        
        import pandas as pd
        from nba_similarity.embeddings.pca_embedding import PCAEmbedder
        from nba_similarity.similarity.search import SimilaritySearcher
        from nba_similarity.utils.config import RANDOM_SEED
        
        embedding_20d = pd.read_csv(embedding_20d_path)
        embedding_2d = pd.read_csv(embedding_2d_path)
        
        # Load PCA models
        pca_path = ARTIFACTS_DIR / "pca_models.pkl"
        if pca_path.exists():
            pipeline.pca_embedder = PCAEmbedder.load(pca_path)
        else:
            logger.warning("PCA models not found. Some metrics may be unavailable.")
        
        # Initialize similarity searcher
        pipeline.similarity_searcher = SimilaritySearcher(
            embedding_20d,
            random_seed=RANDOM_SEED
        )
        
        # Load features
        features_path = Path("data/processed/engineered_features.csv")
        if features_path.exists():
            pipeline.features_df = pd.read_csv(features_path)
        
        # Load scaled features
        scaled_features_path = Path("data/processed/scaled_features.csv")
        if scaled_features_path.exists():
            pipeline.scaled_features_df = pd.read_csv(scaled_features_path)
        else:
            pipeline.scaled_features_df = pipeline.features_df  # Fallback
        
        # Load cluster assignments
        cluster_path = Path("data/processed/cluster_assignments.csv")
        if cluster_path.exists():
            pipeline.cluster_assignments = pd.read_csv(cluster_path)
        
        # Run evaluation
        results = pipeline.evaluate()
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        if 'similarity' in results:
            sim = results['similarity']
            print(f"\nSimilarity Search:")
            print(f"  Improvement Ratio: {sim.get('improvement_ratio', 'N/A'):.4f}")
        
        if 'clustering' in results:
            clust = results['clustering']
            print(f"\nClustering:")
            print(f"  Silhouette Score: {clust.get('silhouette_score', 'N/A'):.4f}")
            print(f"  Number of Clusters: {clust.get('n_clusters', 'N/A')}")
        
        if 'pca_20d' in results:
            pca = results['pca_20d']
            print(f"\nPCA 20D:")
            print(f"  Explained Variance: {pca.get('total_explained_variance', 'N/A'):.4f}")
        
        if 'pca_2d' in results:
            pca = results['pca_2d']
            print(f"\nPCA 2D:")
            print(f"  Explained Variance: {pca.get('total_explained_variance', 'N/A'):.4f}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def app_command(args):
    """Run Streamlit app command."""
    import subprocess
    import sys
    import webbrowser
    import time
    import threading
    
    app_path = Path(__file__).parent.parent / "app.py"
    
    if not app_path.exists():
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)
    
    logger.info("Starting Streamlit app...")
    logger.info("The app will open automatically in your browser...")
    
    # Function to open browser after a short delay
    def open_browser():
        time.sleep(2)  # Wait for server to start
        webbrowser.open("http://localhost:8501")
    
    # Start browser opening in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NBA Player Similarity and Career Trajectory Modeling"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess NBA data')
    preprocess_parser.add_argument(
        '--csv-file',
        type=str,
        help='Path to CSV file with NBA statistics'
    )
    preprocess_parser.add_argument(
        '--fetch',
        action='store_true',
        help='Fetch data from API (placeholder)'
    )
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train embeddings and models')
    train_parser.add_argument(
        '--csv-file',
        type=str,
        help='Path to CSV file (if preprocessing needed)'
    )
    train_parser.add_argument(
        '--n-clusters',
        type=int,
        help='Number of clusters for KMeans'
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.set_defaults(func=evaluate_command)
    
    # App command
    app_parser = subparsers.add_parser('app', help='Run Streamlit app')
    app_parser.set_defaults(func=app_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()

