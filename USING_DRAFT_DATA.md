# Using Draft Data Format

This guide explains how to use the NBA draft data format (like `nbaplayersdraft.csv`) with the pipeline.

## Overview

The pipeline includes an automatic data adapter that can detect and transform NBA draft data format into the standard format required by the pipeline.

## Draft Data Format

The draft data CSV typically contains:
- `player`: Player name
- `year`: Draft year
- `team`: Team abbreviation
- `games`: Total games played (career)
- `minutes_played`: Total minutes (career)
- `points`: Total points (career)
- `total_rebounds`: Total rebounds (career)
- `assists`: Total assists (career)
- `field_goal_percentage`: Career field goal percentage
- `3_point_percentage`: Career 3-point percentage
- `free_throw_percentage`: Career free throw percentage
- And other advanced metrics

## Automatic Transformation

When you run the preprocessing command, the pipeline will:

1. **Auto-detect** the draft format by checking for key columns
2. **Transform** the data to standard format:
   - Maps `player` → `player_name`
   - Maps `year` → `season` (formatted as YYYY-YY)
   - Maps `minutes_played` → `minutes`
   - Estimates missing columns (field goal attempts, three-point attempts, etc.)
   - Estimates per-season stats from career totals
3. **Validate** and filter the data

## Usage

Simply use your draft CSV file directly:

```bash
python -m nba_similarity.cli preprocess --csv-file nbaplayersdraft.csv
```

The pipeline will automatically detect and transform it.

## Transformation Details

The adapter performs the following transformations:

1. **Player and Season**: Maps player name and converts year to season format
2. **Shooting Stats**: Estimates field goal attempts from points and percentages
3. **Three-Pointers**: Estimates three-point attempts and makes
4. **Two-Pointers**: Calculates from field goals minus three-pointers
5. **Free Throws**: Estimates from points and percentages
6. **Rebounds**: Splits total rebounds into offensive/defensive (25%/75% estimate)
7. **Other Stats**: Uses league averages for steals, blocks, turnovers, and fouls

## Notes

- The draft data contains **career totals**, not per-season data
- The adapter estimates per-season statistics from career totals
- Some columns (like steals, blocks) are estimated using league averages
- Players with missing or zero minutes/games are filtered out

## Example

If you have `nbaplayersdraft.csv` in your project root:

```bash
# Preprocess (auto-detects draft format)
python -m nba_similarity.cli preprocess --csv-file nbaplayersdraft.csv

# Train models
python -m nba_similarity.cli train --n-clusters 8

# Run app
python -m nba_similarity.cli app
```

The transformed data will be saved to `data/processed/processed_stats.csv` and you can proceed with the normal pipeline workflow.

