# GUI Guide - Streamlit Application

## Launching the App

### Method 1: Simple Launcher (Recommended) ⭐
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

**Or directly:**
```bash
python launch_app.py
```

### Method 2: Using the CLI Command
```bash
python -m nba_similarity.cli app
```

### Method 3: Direct Streamlit Command
```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## App Features

The Streamlit GUI provides 5 main pages accessible via the sidebar:

### 1. **Player Search** 🏀
- **Purpose**: Find players similar to a selected player
- **Features**:
  - Dropdown to select any player from your dataset
  - Slider to choose number of similar players (5-20)
  - Table showing similar players with similarity scores
  - Feature comparison table showing top 15 features for the selected player vs. their most similar player
- **How to Use**:
  1. Select a player from the dropdown
  2. Adjust the "Number of similar players" slider
  3. View the results table
  4. Scroll down to see detailed feature comparison

### 2. **Similarity Results** 📊
- **Purpose**: Compare multiple players at once
- **Features**:
  - Multi-select dropdown to choose up to 5 players
  - Slider to set number of similar players per query
  - Results table showing all similar players for each query
  - Interactive heatmap visualization showing similarity matrix
- **How to Use**:
  1. Select multiple players (up to 5)
  2. Set the number of similar players per query
  3. View the results table
  4. See the similarity heatmap below

### 3. **Embedding Visualization** 📈
- **Purpose**: Visualize players in 2D space using PCA
- **Features**:
  - Interactive scatter plot of all players
  - Color-coded by cluster (if available)
  - Hover to see player name and season
  - Player search tool to find specific players in the plot
- **How to Use**:
  1. View the 2D PCA plot (players are positioned by similarity)
  2. Hover over points to see player names
  3. Use the search box to find a specific player's location
  4. Points closer together = more similar players

### 4. **Cluster Analysis** 🎯
- **Purpose**: Explore player archetypes/clusters
- **Features**:
  - Bar chart showing cluster sizes
  - Dropdown to select a cluster
  - Cluster label (auto-generated description)
  - List of all players in the selected cluster
- **How to Use**:
  1. View cluster size distribution
  2. Select a cluster from the dropdown
  3. See the cluster's label (e.g., "high-three_pointers / low-paint_shot")
  4. Browse the list of players in that cluster

### 5. **Career Trajectories** 📉
- **Purpose**: Analyze career paths and find historical comparisons
- **Features**:
  - Select a player to analyze
  - Table of historical trajectory comparisons
  - Career trajectory plot (points per 36 minutes over time)
- **How to Use**:
  1. Select a player from the dropdown
  2. Adjust number of trajectory comps (5-15)
  3. View players with similar early-career trajectories
  4. See the player's career trajectory chart

## Navigation

- **Sidebar**: Use the radio buttons in the left sidebar to switch between pages
- **Auto-refresh**: The app automatically loads the latest processed data
- **Responsive**: Works on desktop and tablet screens

## Prerequisites

Before using the app, make sure you've run:

```bash
# 1. Preprocess your data
python -m nba_similarity.cli preprocess --csv-file nbaplayersdraft.csv

# 2. Train the models
python -m nba_similarity.cli train --n-clusters 8
```

The app needs these files to exist:
- `data/processed/embedding_20d.csv`
- `data/processed/embedding_2d.csv`
- `data/processed/engineered_features.csv`
- `data/processed/cluster_assignments.csv`

## Troubleshooting

### "Data not found" Error
- Make sure you've run the preprocessing and training steps first
- Check that files exist in `data/processed/` directory

### App Won't Start
- Make sure Streamlit is installed: `pip install streamlit`
- Check that port 8501 is not already in use
- Try a different port: `streamlit run app.py --server.port 8502`

### No Players Showing
- Verify your data was processed correctly
- Check the preprocessing logs for errors
- Ensure at least some players passed the minimum minutes filter (500 minutes)

## Tips

1. **Start with Player Search**: This is the most intuitive feature - pick a player you know and see who's similar
2. **Explore Clusters**: Use cluster analysis to understand player archetypes
3. **Visualize Similarity**: The embedding plot shows the "shape" of player similarity in 2D
4. **Compare Trajectories**: Use career trajectories to find players with similar career paths

## Example Workflow

1. **Launch the app**: `python -m nba_similarity.cli app`
2. **Go to Player Search**: Select "LeBron James" (or any player in your data)
3. **View Similar Players**: See who has a similar playstyle
4. **Check Embedding Plot**: See where players are positioned in similarity space
5. **Explore Clusters**: See what archetype your selected player belongs to
6. **Analyze Trajectory**: See their career path and historical comparisons

Enjoy exploring NBA player similarities! 🏀

