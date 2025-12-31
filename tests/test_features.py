"""Unit tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from nba_similarity.features.engineering import FeatureEngineer


@pytest.fixture
def sample_stats():
    """Create sample NBA statistics DataFrame."""
    np.random.seed(42)
    
    n_players = 10
    
    data = {
        'player_name': [f"Player_{i}" for i in range(n_players)],
        'season': '2023-24',
        'team': 'LAL',
        'age': np.random.randint(20, 35, n_players),
        'games': np.random.randint(50, 82, n_players),
        'games_started': np.random.randint(0, 82, n_players),
        'minutes': np.random.randint(1000, 3000, n_players),
        'field_goals': np.random.randint(200, 800, n_players),
        'field_goal_attempts': np.random.randint(400, 1500, n_players),
        'field_goal_pct': np.random.uniform(0.35, 0.55, n_players),
        'three_pointers': np.random.randint(50, 300, n_players),
        'three_point_attempts': np.random.randint(100, 700, n_players),
        'three_point_pct': np.random.uniform(0.30, 0.45, n_players),
        'two_pointers': np.random.randint(150, 500, n_players),
        'two_point_attempts': np.random.randint(300, 800, n_players),
        'two_point_pct': np.random.uniform(0.40, 0.60, n_players),
        'free_throws': np.random.randint(100, 500, n_players),
        'free_throw_attempts': np.random.randint(150, 600, n_players),
        'free_throw_pct': np.random.uniform(0.70, 0.90, n_players),
        'offensive_rebounds': np.random.randint(50, 200, n_players),
        'defensive_rebounds': np.random.randint(100, 500, n_players),
        'total_rebounds': np.random.randint(150, 700, n_players),
        'assists': np.random.randint(50, 600, n_players),
        'steals': np.random.randint(20, 150, n_players),
        'blocks': np.random.randint(10, 200, n_players),
        'turnovers': np.random.randint(50, 300, n_players),
        'personal_fouls': np.random.randint(100, 250, n_players),
        'points': np.random.randint(500, 2000, n_players)
    }
    
    return pd.DataFrame(data)


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer.feature_names == []
    assert engineer.scaler is None


def test_engineer_features(sample_stats):
    """Test feature engineering."""
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(sample_stats)
    
    assert 'player_name' in features_df.columns
    assert 'season' in features_df.columns
    assert len(engineer.feature_names) > 0
    
    # Check for per-36 features
    per36_cols = [col for col in features_df.columns if '_per36' in col]
    assert len(per36_cols) > 0
    
    # Check for shot profile features
    shot_cols = [col for col in features_df.columns if '_rate' in col]
    assert len(shot_cols) > 0
    
    # Check for efficiency features
    eff_cols = [col for col in features_df.columns if 'ts_pct' in col or 'efg_pct' in col]
    assert len(eff_cols) > 0


def test_standardize_features(sample_stats):
    """Test feature standardization."""
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(sample_stats)
    scaled_df, scaler = engineer.standardize_features(features_df, fit=True)
    
    assert scaler is not None
    assert 'player_name' in scaled_df.columns
    assert 'season' in scaled_df.columns
    
    # Check that features are standardized (mean ~0, std ~1)
    feature_cols = [col for col in scaled_df.columns 
                   if col not in ['player_name', 'season']]
    
    for col in feature_cols[:5]:  # Check first 5 features
        mean = scaled_df[col].mean()
        std = scaled_df[col].std()
        assert abs(mean) < 0.1  # Should be close to 0
        assert abs(std - 1.0) < 0.1  # Should be close to 1


def test_standardize_features_without_fit(sample_stats):
    """Test standardization without fitting scaler."""
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(sample_stats)
    
    # First fit
    scaled_df1, scaler1 = engineer.standardize_features(features_df, fit=True)
    
    # Then transform without fit
    scaled_df2, scaler2 = engineer.standardize_features(features_df, fit=False)
    
    # Should use same scaler
    assert scaler1 is scaler2


def test_standardize_features_error_if_not_fitted(sample_stats):
    """Test error when transforming without fitting."""
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(sample_stats)
    
    with pytest.raises(ValueError, match="not fitted"):
        engineer.standardize_features(features_df, fit=False)

