"""Generate sample NBA statistics data for testing."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_similarity.data.loader import NBADataLoader


def generate_sample_data(n_players: int = 50, n_seasons: int = 3, output_path: Path = None):
    """Generate sample NBA statistics data.
    
    Args:
        n_players: Number of players to generate.
        n_seasons: Number of seasons per player.
        output_path: Path to save CSV file.
    """
    np.random.seed(42)
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "raw" / "nba_stats.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    seasons = ['2021-22', '2022-23', '2023-24']
    
    for player_id in range(n_players):
        player_name = f"Player_{player_id}"
        
        for season_idx, season in enumerate(seasons[:n_seasons]):
            # Simulate career progression
            age = 20 + player_id % 15 + season_idx
            games = np.random.randint(50, 82)
            games_started = np.random.randint(0, games)
            minutes = np.random.randint(1000, 3000)
            
            # Shooting stats
            fga = np.random.randint(400, 1500)
            fg_pct = np.random.uniform(0.35, 0.55)
            fgm = int(fga * fg_pct)
            
            # Three-pointers
            three_attempts = np.random.randint(100, min(700, fga))
            three_pct = np.random.uniform(0.30, 0.45)
            three_made = int(three_attempts * three_pct)
            
            # Two-pointers
            two_attempts = fga - three_attempts
            two_made = fgm - three_made
            two_pct = two_made / two_attempts if two_attempts > 0 else 0.0
            
            # Free throws
            fta = np.random.randint(150, 600)
            ft_pct = np.random.uniform(0.70, 0.90)
            ftm = int(fta * ft_pct)
            
            # Rebounds
            off_reb = np.random.randint(50, 200)
            def_reb = np.random.randint(100, 500)
            total_reb = off_reb + def_reb
            
            # Other stats
            assists = np.random.randint(50, 600)
            steals = np.random.randint(20, 150)
            blocks = np.random.randint(10, 200)
            turnovers = np.random.randint(50, 300)
            fouls = np.random.randint(100, 250)
            
            # Points
            points = fgm * 2 + three_made + ftm
            
            data = {
                'player_name': player_name,
                'season': season,
                'team': f"Team_{player_id % 30}",
                'age': age,
                'games': games,
                'games_started': games_started,
                'minutes': minutes,
                'field_goals': fgm,
                'field_goal_attempts': fga,
                'field_goal_pct': fg_pct,
                'three_pointers': three_made,
                'three_point_attempts': three_attempts,
                'three_point_pct': three_pct,
                'two_pointers': two_made,
                'two_point_attempts': two_attempts,
                'two_point_pct': two_pct,
                'free_throws': ftm,
                'free_throw_attempts': fta,
                'free_throw_pct': ft_pct,
                'offensive_rebounds': off_reb,
                'defensive_rebounds': def_reb,
                'total_rebounds': total_reb,
                'assists': assists,
                'steals': steals,
                'blocks': blocks,
                'turnovers': turnovers,
                'personal_fouls': fouls,
                'points': points
            }
            
            all_data.append(data)
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    
    print(f"Generated sample data with {len(df)} player-season records")
    print(f"Saved to: {output_path}")
    print(f"Players: {df['player_name'].nunique()}")
    print(f"Seasons: {df['season'].nunique()}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample NBA statistics data")
    parser.add_argument('--n-players', type=int, default=50, help='Number of players')
    parser.add_argument('--n-seasons', type=int, default=3, help='Number of seasons per player')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    generate_sample_data(
        n_players=args.n_players,
        n_seasons=args.n_seasons,
        output_path=output_path
    )

