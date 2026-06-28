# Pulls every D-1 game from barttorvik.com for a set of seasons and turns the raw
# game logs into a clean, one-row-per-game table of pregame features for the 
# spread model. 
import requests
import pandas as pd
import json
import numpy as np

# Column names to change throughout the cleaning process
COLS = [
    "date", "type_code", "team", "team_conf", "opp", "venue", "result",
    "adj_oe", "adj_de", "oe", "efg", "tov_pct", "orb_pct", "ft_rate",
    "opp_oe", "opp_efg", "opp_tov_pct", "opp_orb_pct", "opp_ft_rate", "g-sc",
    "opp_conf", "side", "season", "tempo", "game_id",
    "coach", "opp_coach", "proj_margin", "home_winprob", "boxscore_json", "flag"
]

DROP_COLS_INITIAL = [
    "type_code", "team_conf", "g-sc", "opp_conf", "season",
    "coach", "opp_coach", "proj_margin", "home_winprob", "flag"
]

BOX_COLS = [
    "bs_date", "bs_gamecode", "bs_team", "bs_opp",
    "fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb",
    "x1", "trb", "x2", "x3", "x4", "tov", "pf", "pts",
    "opp_fgm", "opp_fga", "opp_tpm", "opp_tpa", "opp_ftm", "opp_fta", "opp_orb",
    "opp_x1", "opp_trb", "opp_x2", "opp_x3", "opp_x4", "opp_tov", "opp_pf", "opp_pts",
    "poss", "notes", "home_team", "away_team"
]

COLS_TO_DROP_BOXSCORE = [
    'oe', 'opp_oe', "bs_date", "bs_gamecode", "bs_team", "bs_opp",
    "fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb",
    "x1", "trb", "x2", "x3", "x4", "tov", "pf", "pts",
    "opp_fgm", "opp_fga", "opp_tpm", "opp_tpa", "opp_ftm", "opp_fta", "opp_orb",
    "opp_x1", "opp_trb", "opp_x2", "opp_x3", "opp_x4", "opp_tov", "opp_pf", "opp_pts",
    "poss", "notes", "home_team", "away_team"
]

COLUMN_NAMES = [
    'date', 'team', 'opp', 'venue', 'result', 'adjoe', 'adjde',
    'efg%', 'tor%', 'orb%', 'ftr', 'efgD%', 'torD%', 'orbD%', 'ftrD',
    'side', 'tempo', 'game_id', 'fga', '3pa', '2pa', 'fgaD', '3paD', '2paD',
    '2p%', '3p%', '3PR', '2pD%', '3pD%', '3PRD'
]

DROP_COLS_PREGAME = [
    'adjoe', 'adjde', 'result', 'efg%', 'tor%', 'orb%', 'ftr',
    'efgD%', 'torD%', 'orbD%', 'ftrD', 'tempo', 'fga', '3pa', '2pa',
    'fgaD', '3paD', '2paD', '2p%', '3p%', '3PR', '2pD%', '3pD%', '3PRD'
]

DROP_COLS_MERGE = [
    'game_id', 'side_x', 'date_s2', 'venue_s2', 'team_s2', 'opp_s2', 'margin_s2', 'side_y'
]

RENAME_COLS = [
    'date', 'venue', 'team', 'opp', 'margin',
    'team_adjoe', 'team_adjde', 'team_efg%', 'team_tor%', 'team_orb%', 'team_ftr',
    'team_efgD%', 'team_torD%', 'team_orbD%', 'team_ftrD',
    'team_2p%', 'team_2pD%', 'team_3p%', 'team_3pD%', 'team_3PR', 'team_3PRD',
    'opp_adjoe', 'opp_adjde', 'opp_efg%', 'opp_tor%', 'opp_orb%', 'opp_ftr',
    'opp_efgD%', 'opp_torD%', 'opp_orbD%', 'opp_ftrD',
    'opp_2p%', 'opp_2pD%', 'opp_3p%', 'opp_3pD%', 'opp_3PR', 'opp_3PRD'
]
# Turn home, away, and neutral venues into numerical features
VENUE_MAP = {"H": 1, "A": -1, "N": 0}


# Functions to fetch the data from barttorvik.com
def fetch_year(year):
    url = f"https://barttorvik.com/getgamestats.php?year={year}&tvalue=All"
    data = requests.get(url, timeout=60).json()
    return pd.DataFrame(data)
# get several seasons at once 
def fetch_historical_data(years):
    return {year: fetch_year(year) for year in years}


# Function to expand the boxscore column of the JSON into separate columns
# Purpose is to be able to calculate 2p%, 3p%, and 3PR for future use
def expand_boxscore(df, col="boxscore_json"):
    decoded = df[col].apply(
        lambda s: json.loads(s) if pd.notna(s) else [np.nan] * len(BOX_COLS)
    )
    bs_df = pd.DataFrame(decoded.tolist(), index=df.index)
    bs_df.columns = BOX_COLS[:bs_df.shape[1]]
    return df.join(bs_df).drop(columns=[col], errors="ignore")

# Align boxscore stats so they always refer to the team in the 'team column
# The JSON for boxscore isn't guaranteed to be written from the perspective
# of whoever is in the team column so must correct for that
def align_boxscore_and_shooting(df):
    d = df.copy()

    required = ["team", "bs_team", "bs_opp",
                "fgm", "fga", "tpm", "tpa",
                "opp_fgm", "opp_fga", "opp_tpm", "opp_tpa"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    # rows where the boxscore is flipped relative to my team column
    mismatch = d["team"] != d["bs_team"]

    pair_cols = [
        ("fgm", "opp_fgm"), ("fga", "opp_fga"), ("tpm", "opp_tpm"), ("tpa", "opp_tpa"),
        ("ftm", "opp_ftm"), ("fta", "opp_fta"), ("orb", "opp_orb"), ("trb", "opp_trb"),
        ("tov", "opp_tov"), ("pf", "opp_pf"), ("pts", "opp_pts"),
    ]
    # swap each (team, opp) pair on the flipped rows
    for a, b in pair_cols:
        if a in d.columns and b in d.columns:
            tmp = d.loc[mismatch, a].copy()
            d.loc[mismatch, a] = d.loc[mismatch, b]
            d.loc[mismatch, b] = tmp

    tmp = d.loc[mismatch, "bs_team"].copy()
    d.loc[mismatch, "bs_team"] = d.loc[mismatch, "bs_opp"]
    d.loc[mismatch, "bs_opp"] = tmp

    # Attempts
    d["team_FGA"] = d["fga"]
    d["team_3PA"] = d["tpa"]
    d["team_2PA"] = d["fga"] - d["tpa"]
    d["opp_FGA"] = d["opp_fga"]
    d["opp_3PA"] = d["opp_tpa"]
    d["opp_2PA"] = d["opp_fga"] - d["opp_tpa"]

    # Shooting metrics
    d["team_2P%"] = np.where(d["team_2PA"] > 0, (d["fgm"] - d["tpm"]) / d["team_2PA"], np.nan)
    d["team_3P%"] = np.where(d["team_3PA"] > 0, d["tpm"] / d["team_3PA"], np.nan)
    d["team_3PR"] = np.where(d["team_FGA"] > 0, d["team_3PA"] / d["team_FGA"], np.nan)
    d["opp_2P%"] = np.where(d["opp_2PA"] > 0, (d["opp_fgm"] - d["opp_tpm"]) / d["opp_2PA"], np.nan)
    d["opp_3P%"] = np.where(d["opp_3PA"] > 0, d["opp_tpm"] / d["opp_3PA"], np.nan)
    d["opp_3PR"] = np.where(d["opp_FGA"] > 0, d["opp_3PA"] / d["opp_FGA"], np.nan)

    return d

# Convert dates to datetime and sort by teams and date
def convert_dates(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    return df

# Convert result strings to a margin 
# Example "W, 70-60" -> 10
# Example "L, 70-60" -> -10
def result_to_margin(s):
    if pd.isna(s):
        return None
    s = str(s).replace(" ", "")
    wl, score = s.split(",", 1)
    a, b = score.split("-", 1)
    a, b = int(a), int(b)
    diff = a - b
    return diff if wl.upper() == "W" else -diff


# Function to aggregate past games stats played before the game, so that
# we have each team season-to-date average stats, enforcing zero lookahead.
# Averages are volume weighted by tempo or shot attempts 
# min_games takes the first x games out of the dataset as rows where an average would
# not be meaningful (still used to calculate future rows); default set 5 
def add_pregame_weighted_std(df, min_games, suffix="_STD"):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values(["team", "date"]).reset_index(drop=True)

    if "tempo" not in d.columns:
        raise KeyError("Expected a 'tempo' column to use as the weight proxy.")
    # how to weight each stat (possessions vs attempts)
    weighted_map = {
        "adjoe": "tempo", "adjde": "tempo",
        "efg%": "fga", "tor%": "tempo", "orb%": "tempo", "ftr": "fga",
        "efgD%": "fgaD", "torD%": "tempo", "orbD%": "tempo", "ftrD": "fgaD",
        "2p%": "2pa", "2pD%": "2paD", "3p%": "3pa", "3pD%": "3paD",
        "3PR": "fga", "3PRD": "fgaD"
    }

    needed = set(weighted_map.keys()) | set(weighted_map.values())
    for c in needed:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    prior_games = d.groupby("team").cumcount()

    for stat, wcol in weighted_map.items():
        if stat not in d.columns or wcol not in d.columns:
            continue

        w = d[wcol].where(d[wcol] > 0) # ignore 0/negative weights
        # weighted average of all prior games (shift(1) enforces no leakage)
        num_cum = (d[stat] * w).groupby(d["team"]).cumsum().shift(1)
        den_cum = w.groupby(d["team"]).cumsum().shift(1)

        d[f"{stat}{suffix}"] = num_cum / den_cum
        # Cut out the games that are too early in the season from the dataframe
        # dont want to train games too early in the season
        d.loc[prior_games < min_games, f"{stat}{suffix}"] = np.nan

    return d

# barttorvik.com returns 2 rows per game, one from each team's perspective
# for modeling I wanted a single row per matchup, so I split on side and merge
# the halves together on game_id
def one_row_per_game(df):
    d = df.copy()

    std_cols = [c for c in d.columns if c.endswith("_STD")]
    base_cols = ["game_id", "date", "venue", "team", "opp", "margin", "side"]

    d = d[base_cols + std_cols].copy()

    rename_cols = ["date", "venue", "team", "opp", "margin"] + std_cols
    s1 = d[d["side"] == 1].copy().rename(columns={c: f"{c}_s1" for c in rename_cols})
    s2 = d[d["side"] == 2].copy().rename(columns={c: f"{c}_s2" for c in rename_cols})

    return s1.merge(s2, left_on="game_id", right_on="game_id", how="inner")

# drop the duplicate columns from the merge and rename the columns to the final team/opp scheme 
def clean_merged(df):
    return df.drop(columns=DROP_COLS_MERGE).set_axis(RENAME_COLS, axis=1).sort_values('date').reset_index(drop=True)

# Reduce the 30 features into matchup net differentials based on the premise
# that matchup specific features matter more than how good a team's stats are in a vacuum
def collapse_features(df):
    d = df.copy()
    
    d['net_rtg_diff'] = (d['team_adjoe'] - d['team_adjde']) - (d['opp_adjoe'] - d['opp_adjde'])
    d['efg%_diff'] = d['team_efg%'] - d['team_efgD%'] + d['opp_efgD%'] - d['opp_efg%']
    d['tor%_diff'] = (d['opp_tor%'] + d['team_torD%']) - (d['team_tor%'] + d['opp_torD%'])
    d['orb%_diff'] = d['team_orb%'] - d['team_orbD%'] + d['opp_orbD%'] - d['opp_orb%']
    d['ftr_diff'] = d['team_ftr'] - d['team_ftrD'] + d['opp_ftrD'] - d['opp_ftr']
    d['2p%_diff'] = d['team_2p%'] - d['team_2pD%'] + d['opp_2pD%'] - d['opp_2p%']
    d['3p%_diff'] = d['team_3p%'] - d['team_3pD%'] + d['opp_3pD%'] - d['opp_3p%']
    d['3PR_diff'] = d['team_3PR'] - d['team_3PRD'] + d['opp_3PRD'] - d['opp_3PR']
    
    drop_cols = [
        'team_adjoe', 'team_adjde', 'opp_adjoe', 'opp_adjde',
        'team_efg%', 'team_efgD%', 'opp_efg%', 'opp_efgD%',
        'team_tor%', 'team_torD%', 'opp_tor%', 'opp_torD%',
        'team_orb%', 'team_orbD%', 'opp_orb%', 'opp_orbD%',
        'team_ftr', 'team_ftrD', 'opp_ftr', 'opp_ftrD',
        'team_2p%', 'team_2pD%', 'opp_2p%', 'opp_2pD%',
        'team_3p%', 'team_3pD%', 'opp_3p%', 'opp_3pD%',
        'team_3PR', 'team_3PRD', 'opp_3PR', 'opp_3PRD'
    ]
    
    return d.drop(columns=drop_cols)


# runs one season through the entire data cleaning and preprocessing pipeline
# model_df gets used in training
def process_single_year(df, min_games = 5):
    # Apply column labels
    df.columns = COLS[:len(df.columns)]
    df = df.drop(columns=DROP_COLS_INITIAL)
    
    # Expand boxscore
    df = expand_boxscore(df)
    
    # Align boxscore and shooting stats
    df = align_boxscore_and_shooting(df)
    
    # Drop boxscore columns
    df = df.drop(columns=COLS_TO_DROP_BOXSCORE)
    
    # Rename columns and convert dates
    df = convert_dates(df.set_axis(COLUMN_NAMES, axis=1))
    
    # Add pregame weighted stats
    df = add_pregame_weighted_std(df, min_games=min_games)
    
    # Add margin
    df['margin'] = df['result'].apply(result_to_margin)
    
    # Drop post-game stats
    df = df.drop(columns=DROP_COLS_PREGAME)
    
    # Merge to one row per game
    merged_df = one_row_per_game(df)
    merged_df = clean_merged(merged_df)
    
    # Clean up and add HCA
    merged_df.dropna(inplace=True)
    merged_df["hca"] = merged_df["venue"].map(VENUE_MAP).fillna(0).astype(int)
    
    # Create model-ready dataframe
    model_df = collapse_features(merged_df).drop(columns=['date', 'venue', 'team', 'opp'])
    
    return merged_df, model_df

# Run several seasons through the full pipeline
def load_historical_data(years = [2021, 2022, 2023, 2024, 2025], min_games = 5):
    merged_dfs = {}
    model_dfs = {}
    
    for year in years:
        print(f"Processing {year}...")
        raw_df = fetch_year(year)
        merged_df, model_df = process_single_year(raw_df, min_games=min_games)
        merged_dfs[year] = merged_df
        model_dfs[year] = model_df
    
    return merged_dfs, model_dfs


# Script entry point
if __name__ == "__main__":
    years = [2021, 2022, 2023, 2024, 2025]
    merged_dfs, model_dfs = load_historical_data(years)
    
    # Print summary
    for year in years:
        print(f"{year}: {len(model_dfs[year])} games ready for modeling")