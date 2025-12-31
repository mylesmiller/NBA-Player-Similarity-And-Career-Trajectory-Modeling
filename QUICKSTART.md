# Quick Start Guide

This guide will help you get started with the NBA Player Similarity project quickly.

## Step 1: Set Up Virtual Environment (Recommended)

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Or use the setup scripts:**
- Windows: `setup_env.bat`
- Linux/Mac: `chmod +x setup_env.sh && ./setup_env.sh`

> **Note:** Always activate your virtual environment before running commands. You'll see `(venv)` in your terminal prompt when it's active.

See [VENV_GUIDE.md](VENV_GUIDE.md) for more details.

## Step 2: Prepare Your Data

> **Make sure your virtual environment is activated!** You should see `(venv)` in your terminal prompt.

You have two options:

### Option A: Use Your Own Data

If you have a CSV file with NBA statistics (like `nbaplayersdraft.csv`), you can use it directly:

```bash
python -m nba_similarity.cli preprocess --csv-file nbaplayersdraft.csv
```

The pipeline will automatically detect the format and transform it if needed.

### Option B: Generate Sample Data (Optional)

If you don't have NBA statistics data, you can generate sample data for testing:

```bash
python scripts/generate_sample_data.py --n-players 50 --n-seasons 3
```

This will create a sample CSV file at `data/raw/nba_stats.csv` with 50 players across 3 seasons.

## Step 3: Preprocess Data

If you have your own CSV file (like `nbaplayersdraft.csv`):

```bash
python -m nba_similarity.cli preprocess --csv-file nbaplayersdraft.csv
```

Or if you used the sample data generator or have `data/raw/nba_stats.csv`:

```bash
python -m nba_similarity.cli preprocess
```

This will:
- Load and validate the CSV file
- Filter players with <500 minutes
- Handle missing values
- Save processed data to `data/processed/processed_stats.csv`

## Step 4: Train Models

```bash
python -m nba_similarity.cli train --n-clusters 8
```

This will:
- Engineer features (per-36, shot profiles, efficiency metrics)
- Standardize features
- Generate 20D and 2D PCA embeddings
- Cluster players using KMeans
- Build career trajectory signatures
- Save all artifacts

## Step 5: Evaluate Models

```bash
python -m nba_similarity.cli evaluate
```

This will compute and display:
- Similarity search quality metrics
- Clustering quality (silhouette score)
- PCA explained variance

## Step 6: Run the Streamlit App

The app will automatically open in your browser:

**Windows:**
```bash
launch_app.bat
```

**Linux/Mac:**
```bash
chmod +x launch_app.sh
./launch_app.sh
```

**Or using Python directly:**
```bash
python launch_app.py
```

**Or using CLI:**
```bash
python -m nba_similarity.cli app
```

The app will automatically open at `http://localhost:8501` in your default browser.

The app will open in your browser. You can:
- Search for players and find similar players
- Visualize player embeddings in 2D
- Explore player clusters
- Analyze career trajectories

## Using Your Own Data

To use your own NBA statistics data:

1. Prepare a CSV file with the required columns (see README.md for schema)
2. Place it in `data/raw/nba_stats.csv` or specify the path
3. Run the preprocessing and training steps above

## Troubleshooting

### "CSV file not found"
- Make sure your CSV file is in `data/raw/` or specify the full path with `--csv-file`

### "Missing required columns"
- Check that your CSV has all required columns (see README.md)
- Column names must match exactly (case-sensitive)

### "No data loaded" in Streamlit app
- Make sure you've run `train` command first
- Check that files exist in `data/processed/`

### Import errors
- Make sure you've installed all dependencies: `pip install -r requirements.txt`
- Make sure you're running commands from the project root directory

## Next Steps

- Explore the code in `nba_similarity/` to understand the implementation
- Modify configuration in `nba_similarity/utils/config.py`
- Add your own features or models
- Extend the Streamlit app with new visualizations

## Example: Finding Similar Players

```python
from nba_similarity.pipeline import NBASimilarityPipeline
from pathlib import Path

# Initialize and run pipeline
pipeline = NBASimilarityPipeline()
pipeline.run_full_pipeline(csv_file=Path("data/raw/nba_stats.csv"))

# Find similar players
similar = pipeline.similarity_searcher.find_similar_players(
    "Player_0", 
    top_k=10
)
print(similar)
```

## Example: Getting Cluster Information

```python
# After running the pipeline
cluster_labels = pipeline.clusterer.get_cluster_labels()
for cluster_id, label in cluster_labels.items():
    print(f"Cluster {cluster_id}: {label}")
```

