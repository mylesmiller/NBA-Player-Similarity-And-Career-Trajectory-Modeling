"""Unit tests for similarity search module."""

import pytest
import pandas as pd
import numpy as np
from nba_similarity.similarity.search import SimilaritySearcher
from nba_similarity.utils.config import RANDOM_SEED


@pytest.fixture
def sample_embeddings():
    """Create sample embedding DataFrame for testing."""
    np.random.seed(RANDOM_SEED)
    
    # Create 20D embeddings for 10 players
    n_players = 10
    n_dims = 20
    
    embeddings = np.random.randn(n_players, n_dims)
    player_names = [f"Player_{i}" for i in range(n_players)]
    
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f'pc_{i+1}' for i in range(n_dims)]
    )
    embedding_df['player_name'] = player_names
    embedding_df['season'] = '2023-24'
    
    return embedding_df


def test_similarity_searcher_initialization(sample_embeddings):
    """Test SimilaritySearcher initialization."""
    searcher = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    assert searcher.embedding_df is not None
    assert len(searcher.player_to_idx) == len(sample_embeddings)
    assert searcher.embeddings.shape[0] == len(sample_embeddings)


def test_find_similar_players(sample_embeddings):
    """Test finding similar players."""
    searcher = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    # Find similar players
    results = searcher.find_similar_players("Player_0", top_k=5, exclude_self=True)
    
    assert len(results) == 5
    assert 'player_name' in results.columns
    assert 'similarity' in results.columns
    assert 'Player_0' not in results['player_name'].values
    assert results['similarity'].is_monotonic_decreasing


def test_find_similar_players_deterministic(sample_embeddings):
    """Test that similarity search is deterministic."""
    searcher1 = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    searcher2 = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    results1 = searcher1.find_similar_players("Player_0", top_k=5)
    results2 = searcher2.find_similar_players("Player_0", top_k=5)
    
    pd.testing.assert_frame_equal(results1, results2)


def test_find_similar_players_invalid_player(sample_embeddings):
    """Test error handling for invalid player name."""
    searcher = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    with pytest.raises(ValueError, match="not found"):
        searcher.find_similar_players("Invalid_Player", top_k=5)


def test_find_similar_players_batch(sample_embeddings):
    """Test batch similarity search."""
    searcher = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    query_players = ["Player_0", "Player_1"]
    results = searcher.find_similar_players_batch(query_players, top_k=3)
    
    assert 'query_player' in results.columns
    assert 'player_name' in results.columns
    assert 'similarity' in results.columns
    assert len(results) == 6  # 2 queries * 3 results each


def test_get_embedding(sample_embeddings):
    """Test getting embedding vector."""
    searcher = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    embedding = searcher.get_embedding("Player_0")
    
    assert embedding.shape[0] == 20  # 20D embedding
    assert isinstance(embedding, np.ndarray)


def test_get_embedding_invalid_player(sample_embeddings):
    """Test error handling for invalid player in get_embedding."""
    searcher = SimilaritySearcher(sample_embeddings, random_seed=RANDOM_SEED)
    
    with pytest.raises(ValueError, match="not found"):
        searcher.get_embedding("Invalid_Player")

