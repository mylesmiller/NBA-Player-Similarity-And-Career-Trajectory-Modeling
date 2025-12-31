"""Streamlit app for NBA player similarity analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle

# Set page config
st.set_page_config(
    page_title="NBA Player Similarity",
    page_icon="🏀",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


@st.cache_data
def load_data():
    """Load all required data."""
    data_dir = Path("data/processed")
    artifacts_dir = Path("artifacts")
    
    data = {}
    
    try:
        # Load embeddings
        if (data_dir / "embedding_20d.csv").exists():
            data['embedding_20d'] = pd.read_csv(data_dir / "embedding_20d.csv")
        if (data_dir / "embedding_2d.csv").exists():
            data['embedding_2d'] = pd.read_csv(data_dir / "embedding_2d.csv")
        
        # Load features
        if (data_dir / "engineered_features.csv").exists():
            data['features'] = pd.read_csv(data_dir / "engineered_features.csv")
        
        # Load cluster assignments
        if (data_dir / "cluster_assignments.csv").exists():
            data['clusters'] = pd.read_csv(data_dir / "cluster_assignments.csv")
        
        # Load PCA models for cluster labels
        if (artifacts_dir / "pca_models.pkl").exists():
            from nba_similarity.embeddings.pca_embedding import PCAEmbedder
            data['pca_embedder'] = PCAEmbedder.load(artifacts_dir / "pca_models.pkl")
        
        # Load trajectory signatures
        if (artifacts_dir / "trajectory_signatures.pkl").exists():
            with open(artifacts_dir / "trajectory_signatures.pkl", 'rb') as f:
                data['trajectory_signatures'] = pickle.load(f)
        
        data['loaded'] = True
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data['loaded'] = False
        data['error'] = str(e)
    
    return data


def main():
    """Main app function."""
    st.title("🏀 NBA Player Similarity and Career Trajectory Analysis")
    st.markdown("---")
    
    # Load data
    data = load_data()
    
    if not data.get('loaded', False):
        st.error("Data not found. Please run the pipeline first:")
        st.code("python -m nba_similarity.cli preprocess --csv-file <your_file.csv>")
        st.code("python -m nba_similarity.cli train")
        return
    
    # Check required data
    required = ['embedding_20d', 'embedding_2d', 'features', 'clusters']
    missing = [r for r in required if r not in data]
    
    if missing:
        st.error(f"Missing required data: {missing}. Please run the training pipeline.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Player Search", "Similarity Results", "Embedding Visualization", 
         "Cluster Analysis", "Career Trajectories"]
    )
    
    if page == "Player Search":
        show_player_search(data)
    elif page == "Similarity Results":
        show_similarity_results(data)
    elif page == "Embedding Visualization":
        show_embedding_plot(data)
    elif page == "Cluster Analysis":
        show_cluster_analysis(data)
    elif page == "Career Trajectories":
        show_trajectories(data)


def show_player_search(data):
    """Player search and similarity page."""
    st.header("Player Search and Similarity")
    
    # Get unique players
    players = sorted(data['embedding_20d']['player_name'].unique())
    
    # Player selection
    selected_player = st.selectbox("Select a player:", players)
    
    if selected_player:
        # Find similar players
        from nba_similarity.similarity.search import SimilaritySearcher
        from nba_similarity.utils.config import RANDOM_SEED
        
        searcher = SimilaritySearcher(data['embedding_20d'], random_seed=RANDOM_SEED)
        
        top_k = st.slider("Number of similar players:", 5, 20, 10)
        
        try:
            similar = searcher.find_similar_players(selected_player, top_k=top_k)
            
            st.subheader(f"Players Similar to {selected_player}")
            
            # Display results
            st.dataframe(
                similar[['player_name', 'similarity']].style.format({
                    'similarity': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Show feature comparison
            if 'features' in data:
                st.subheader("Feature Comparison")
                
                # Get features for selected and top similar player
                selected_features = data['features'][
                    data['features']['player_name'] == selected_player
                ].iloc[0]
                
                top_similar = similar.iloc[0]['player_name']
                similar_features = data['features'][
                    data['features']['player_name'] == top_similar
                ].iloc[0]
                
                # Select numeric features for comparison
                numeric_cols = data['features'].select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['season']]
                
                # Create comparison DataFrame
                comparison = pd.DataFrame({
                    'Feature': numeric_cols[:15],  # Top 15 features
                    selected_player: [selected_features[col] for col in numeric_cols[:15]],
                    top_similar: [similar_features[col] for col in numeric_cols[:15]]
                })
                
                st.dataframe(comparison, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error finding similar players: {e}")


def show_similarity_results(data):
    """Show similarity search results."""
    st.header("Similarity Search Results")
    
    # Get unique players
    players = sorted(data['embedding_20d']['player_name'].unique())
    
    # Multi-select players
    selected_players = st.multiselect("Select players to compare:", players, max_selections=5)
    
    if selected_players:
        from nba_similarity.similarity.search import SimilaritySearcher
        from nba_similarity.utils.config import RANDOM_SEED
        
        searcher = SimilaritySearcher(data['embedding_20d'], random_seed=RANDOM_SEED)
        
        top_k = st.slider("Number of similar players per query:", 5, 15, 10)
        
        try:
            batch_results = searcher.find_similar_players_batch(
                selected_players,
                top_k=top_k
            )
            
            if not batch_results.empty:
                st.dataframe(batch_results, use_container_width=True)
                
                # Visualization
                pivot = batch_results.pivot_table(
                    index='player_name',
                    columns='query_player',
                    values='similarity',
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    pivot,
                    labels=dict(x="Query Player", y="Similar Player", color="Similarity"),
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results found.")
                
        except Exception as e:
            st.error(f"Error: {e}")


def show_embedding_plot(data):
    """Show 2D embedding visualization."""
    st.header("Player Embedding Visualization")
    
    # Merge with cluster assignments if available
    plot_df = data['embedding_2d'].copy()
    
    if 'clusters' in data:
        plot_df = plot_df.merge(
            data['clusters'][['player_name', 'season', 'cluster']],
            on=['player_name', 'season'],
            how='left'
        )
        color_col = 'cluster'
    else:
        color_col = None
    
    # Create plot
    fig = px.scatter(
        plot_df,
        x='pc1',
        y='pc2',
        color=color_col,
        hover_data=['player_name', 'season'],
        title="2D PCA Embedding of Players",
        labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'}
    )
    
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)
    
    # Player search in plot
    st.subheader("Find Player in Plot")
    players = sorted(plot_df['player_name'].unique())
    search_player = st.selectbox("Search for player:", [""] + players)
    
    if search_player:
        player_data = plot_df[plot_df['player_name'] == search_player]
        if not player_data.empty:
            st.info(f"{search_player} is at PC1={player_data.iloc[0]['pc1']:.4f}, "
                   f"PC2={player_data.iloc[0]['pc2']:.4f}")


def show_cluster_analysis(data):
    """Show cluster analysis."""
    st.header("Player Cluster Analysis")
    
    if 'clusters' not in data:
        st.error("Cluster assignments not found.")
        return
    
    # Cluster summary
    cluster_counts = data['clusters']['cluster'].value_counts().sort_index()
    
    st.subheader("Cluster Sizes")
    st.bar_chart(cluster_counts)
    
    # Cluster details
    st.subheader("Cluster Details")
    
    # Load clusterer if available to get labels
    cluster_labels = {}
    try:
        from nba_similarity.clustering.kmeans_cluster import KMeansClusterer
        from nba_similarity.utils.config import DEFAULT_N_CLUSTERS, RANDOM_SEED
        
        # Try to reconstruct clusterer (simplified - in production, save/load)
        n_clusters = data['clusters']['cluster'].nunique()
        clusterer = KMeansClusterer(n_clusters=n_clusters, random_seed=RANDOM_SEED)
        clusterer.fitted = True  # Mark as fitted
        
        # Get labels if possible
        try:
            cluster_labels = clusterer.get_cluster_labels()
        except:
            cluster_labels = {i: f"Cluster {i}" for i in range(n_clusters)}
    except:
        n_clusters = data['clusters']['cluster'].nunique()
        cluster_labels = {i: f"Cluster {i}" for i in range(n_clusters)}
    
    # Display clusters
    selected_cluster = st.selectbox(
        "Select cluster:",
        sorted(data['clusters']['cluster'].unique())
    )
    
    cluster_players = data['clusters'][
        data['clusters']['cluster'] == selected_cluster
    ]['player_name'].unique()
    
    st.write(f"**{cluster_labels.get(selected_cluster, f'Cluster {selected_cluster}')}**")
    st.write(f"Number of players: {len(cluster_players)}")
    
    # Show players in cluster
    cluster_df = data['clusters'][
        data['clusters']['cluster'] == selected_cluster
    ][['player_name', 'season']].sort_values('player_name')
    
    st.dataframe(cluster_df, use_container_width=True)


def show_trajectories(data):
    """Show career trajectory analysis."""
    st.header("Career Trajectory Analysis")
    
    if 'trajectory_signatures' not in data:
        st.warning("Trajectory signatures not found. Run the training pipeline.")
        return
    
    # Get unique players
    players = sorted(data['embedding_20d']['player_name'].unique())
    
    selected_player = st.selectbox("Select a player:", players)
    
    if selected_player:
        from nba_similarity.trajectory.career_trajectory import CareerTrajectoryAnalyzer
        
        analyzer = CareerTrajectoryAnalyzer()
        analyzer.trajectory_signatures = data['trajectory_signatures']
        
        top_k = st.slider("Number of trajectory comps:", 5, 15, 10)
        
        try:
            comps = analyzer.find_trajectory_comps(selected_player, top_k=top_k)
            
            st.subheader(f"Historical Trajectory Comparisons for {selected_player}")
            st.dataframe(comps, use_container_width=True)
            
            # Show trajectory over time if features available
            if 'features' in data:
                st.subheader("Career Trajectory Over Time")
                
                player_data = data['features'][
                    data['features']['player_name'] == selected_player
                ].sort_values('season')
                
                if not player_data.empty and 'points_per36' in player_data.columns:
                    fig = px.line(
                        player_data,
                        x='season',
                        y='points_per36',
                        title=f"{selected_player} - Points per 36 Minutes Over Time",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error finding trajectory comps: {e}")


if __name__ == '__main__':
    main()

