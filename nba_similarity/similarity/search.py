"""Similarity search module."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class SimilaritySearcher:
    """Performs cosine similarity search on player embeddings."""
    
    def __init__(self, embedding_df: pd.DataFrame, random_seed: int = 42):
        """Initialize similarity searcher.
        
        Args:
            embedding_df: DataFrame with player embeddings (must include 'player_name').
            random_seed: Random seed for deterministic tie-breaking.
        """
        self.embedding_df = embedding_df.copy()
        self.random_seed = random_seed
        
        # Extract embedding columns
        self.embedding_cols = [col for col in embedding_df.columns 
                              if col not in ['player_name', 'season']]
        
        # Build embedding matrix
        self.embeddings = embedding_df[self.embedding_cols].values
        
        # Create player name to index mapping
        self.player_to_idx = {
            name: idx for idx, name in enumerate(embedding_df['player_name'].values)
        }
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
    
    def find_similar_players(
        self, 
        player_name: str, 
        top_k: int = 10,
        exclude_self: bool = True
    ) -> pd.DataFrame:
        """Find most similar players using cosine similarity.
        
        Args:
            player_name: Name of player to find similarities for.
            top_k: Number of similar players to return.
            exclude_self: Whether to exclude the query player from results.
            
        Returns:
            DataFrame with similar players and similarity scores.
        """
        if player_name not in self.player_to_idx:
            raise ValueError(f"Player '{player_name}' not found in embeddings.")
        
        # Get query player embedding
        query_idx = self.player_to_idx[player_name]
        query_embedding = self.embeddings[query_idx:query_idx+1]
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'player_name': self.embedding_df['player_name'].values,
            'similarity': similarities
        })
        
        # Exclude self if requested
        if exclude_self:
            results = results[results['player_name'] != player_name]
        
        # Sort by similarity (descending)
        results = results.sort_values('similarity', ascending=False)
        
        # Take top K
        results = results.head(top_k)
        
        # Reset index
        results = results.reset_index(drop=True)
        
        return results
    
    def find_similar_players_batch(
        self,
        player_names: List[str],
        top_k: int = 10,
        exclude_self: bool = True
    ) -> pd.DataFrame:
        """Find similar players for multiple query players.
        
        Args:
            player_names: List of player names.
            top_k: Number of similar players to return per query.
            exclude_self: Whether to exclude query player from results.
            
        Returns:
            DataFrame with query player, similar player, and similarity score.
        """
        all_results = []
        
        for player_name in player_names:
            try:
                similar = self.find_similar_players(
                    player_name, top_k=top_k, exclude_self=exclude_self
                )
                similar['query_player'] = player_name
                all_results.append(similar)
            except ValueError as e:
                logger.warning(f"Skipping {player_name}: {e}")
        
        if not all_results:
            return pd.DataFrame(columns=['query_player', 'player_name', 'similarity'])
        
        result_df = pd.concat(all_results, ignore_index=True)
        return result_df[['query_player', 'player_name', 'similarity']]
    
    def get_embedding(self, player_name: str) -> np.ndarray:
        """Get embedding vector for a player.
        
        Args:
            player_name: Name of player.
            
        Returns:
            Embedding vector.
        """
        if player_name not in self.player_to_idx:
            raise ValueError(f"Player '{player_name}' not found in embeddings.")
        
        idx = self.player_to_idx[player_name]
        return self.embeddings[idx]

