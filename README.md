# NBA Player Similarity and Career Trajectory Modeling

An end-to-end machine learning pipeline for analyzing NBA player similarities, clustering players into archetypes, and modeling career trajectories.

## Features

- **Data Pipeline**: Load NBA player-season statistics from CSV with schema validation and missing value handling
- **Feature Engineering**: Compute per-36 minute stats, shot profile rates, and efficiency proxies
- **Embeddings**: Generate 20D PCA embeddings for similarity search and 2D embeddings for visualization
- **Similarity Search**: Cosine similarity-based top-K neighbor search with deterministic outputs
- **Clustering**: KMeans clustering with automatic cluster labeling based on centroid feature z-scores
- **Career Trajectories**: Build early-career trajectory signatures and find historical trajectory comparisons
- **Evaluation Metrics**: Comprehensive evaluation including neighbor distance vs random, silhouette score, PCA explained variance, and MAE/RMSE
- **Streamlit App**: Interactive web application for exploring player similarities, clusters, and trajectories

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NBA-Player-Similarity-And-Career-Trajectory-Modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
nba-similarity/
├── nba_similarity/          # Main package
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature engineering
│   ├── embeddings/          # PCA embedding generation
│   ├── similarity/          # Similarity search
│   ├── clustering/          # KMeans clustering
│   ├── trajectory/          # Career trajectory analysis
│   ├── evaluation/          # Evaluation metrics
│   ├── utils/               # Utilities and config
│   ├── pipeline.py          # Main pipeline
│   └── cli.py               # Command-line interface
├── data/
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
├── artifacts/               # Saved models and artifacts
├── tests/                   # Unit tests
├── app.py                   # Streamlit application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Data Format

The pipeline supports two data formats:

### Standard Format

A CSV file with NBA player-season statistics. Required columns:

- `player_name`, `season`, `team`, `age`, `games`, `games_started`
- `minutes`, `field_goals`, `field_goal_attempts`, `field_goal_pct`
- `three_pointers`, `three_point_attempts`, `three_point_pct`
- `two_pointers`, `two_point_attempts`, `two_point_pct`
- `free_throws`, `free_throw_attempts`, `free_throw_pct`
- `offensive_rebounds`, `defensive_rebounds`, `total_rebounds`
- `assists`, `steals`, `blocks`, `turnovers`, `personal_fouls`, `points`

### Draft Data Format (Auto-Detected)

The pipeline can automatically detect and transform NBA draft data format (like `nbaplayersdraft.csv`) which contains:
- `player`, `year`, `team`, `games`, `minutes_played`, `points`
- `total_rebounds`, `assists`, `field_goal_percentage`, `3_point_percentage`, `free_throw_percentage`
- And other draft-related columns

The adapter will automatically transform this format to the standard format.

**Place your CSV file in `data/raw/nba_stats.csv` or specify the path when running commands.**

## Usage

### Command-Line Interface

The project provides a CLI with four main commands:

#### 1. Preprocess Data

Load and preprocess NBA statistics:

```bash
python -m nba_similarity.cli preprocess --csv-file data/raw/nba_stats.csv
```

Options:
- `--csv-file`: Path to CSV file (default: `data/raw/nba_stats.csv`)
- `--fetch`: Fetch data from API (placeholder, not implemented)

#### 2. Train Models

Engineer features, train embeddings, and cluster players:

```bash
python -m nba_similarity.cli train --n-clusters 8
```

Options:
- `--csv-file`: Path to CSV file (if preprocessing needed)
- `--n-clusters`: Number of clusters for KMeans (default: 8)

#### 3. Evaluate Models

Run evaluation metrics:

```bash
python -m nba_similarity.cli evaluate
```

This will compute:
- Similarity search quality (neighbor distance vs random)
- Clustering quality (silhouette score)
- PCA explained variance

#### 4. Run Streamlit App

Launch the interactive web application (opens automatically in browser):

**Simple launcher (recommended):**
```bash
# Windows
launch_app.bat

# Linux/Mac
chmod +x launch_app.sh
./launch_app.sh

# Or directly with Python
python launch_app.py
```

**Using CLI:**
```bash
python -m nba_similarity.cli app
```

**Direct Streamlit:**
```bash
streamlit run app.py
```

### Python API

You can also use the pipeline programmatically:

```python
from nba_similarity.pipeline import NBASimilarityPipeline
from pathlib import Path

# Initialize pipeline
pipeline = NBASimilarityPipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    csv_file=Path("data/raw/nba_stats.csv"),
    n_clusters=8
)

# Access individual components
similar_players = pipeline.similarity_searcher.find_similar_players(
    "LeBron James", 
    top_k=10
)
```

## Streamlit App Features

The Streamlit app provides five main pages:

1. **Player Search**: Search for a player and find similar players with feature comparisons
2. **Similarity Results**: Compare multiple players and visualize similarity matrices
3. **Embedding Visualization**: Interactive 2D PCA plot of player embeddings
4. **Cluster Analysis**: Explore player clusters and their characteristics
5. **Career Trajectories**: Analyze career trajectories and find historical comparisons

## Testing

Run unit tests:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=nba_similarity --cov-report=html
```

## Configuration

Key configuration parameters can be adjusted in `nba_similarity/utils/config.py`:

- `MIN_MINUTES_THRESHOLD`: Minimum minutes per season (default: 500)
- `PCA_DIM_20`: Dimensions for similarity embedding (default: 20)
- `PCA_DIM_2`: Dimensions for visualization embedding (default: 2)
- `DEFAULT_N_CLUSTERS`: Default number of clusters (default: 8)
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)

## Output Files

The pipeline generates several output files:

- `data/processed/processed_stats.csv`: Preprocessed statistics
- `data/processed/engineered_features.csv`: Engineered features
- `data/processed/scaled_features.csv`: Standardized features
- `data/processed/embedding_20d.csv`: 20D PCA embeddings
- `data/processed/embedding_2d.csv`: 2D PCA embeddings
- `data/processed/cluster_assignments.csv`: Cluster assignments
- `artifacts/pca_models.pkl`: Saved PCA models
- `artifacts/scaler.pkl`: Saved feature scaler
- `artifacts/trajectory_signatures.pkl`: Trajectory signatures
- `artifacts/evaluation_results.json`: Evaluation metrics

## Notes

- The pipeline filters out players with fewer than 500 minutes per season
- All random operations use a fixed seed (42) for reproducibility
- Missing values are handled by filling with 0 for counts and median for other numeric columns
- The fetch_data option is a placeholder and requires implementation with an NBA API

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]

