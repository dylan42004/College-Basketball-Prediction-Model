# Backtests betting strategies against a CSV of past bets to find which edge
# thresholds, daily bet caps, and unit-sizing tiers would have been most profitable.
# "Edge" here is the gap between my model's predicted spread and the DraftKings line
# -- the bigger that gap, the more the market disagrees with the model, which is the
# whole premise of a bet.
#
# Run it as:  python backtest.py <path_to_csv>
#   e.g.      python backtest.py path/to/betting_tracker.csv
#
# The CSV needs these columns:
#   Date                  - game date
#   Predicted Spread      - my model's line, e.g. "American -0.68"
#   DK Line (Time of Bet) - the DK line when the bet went down, e.g. "American +2.5 (-112)"
#   W/L                   - "WIN" or "LOSS"

import pandas as pd
import numpy as np
import re
import sys
from itertools import product


# Edge calculation
 
# The edge is the absolute gap between the predicted spread and the DK line.
def calculate_edge(row):
    try:
        pred = row['Predicted Spread']
        dk = row['DK Line (Time of Bet)']
        
        if pd.isna(pred) or pd.isna(dk):
            return np.nan
        
        pred_spread = float(re.search(r'([+-]?\d+\.?\d*)\s*$', pred).group(1))
        dk_spread = float(re.search(r'([+-]?\d+\.?\d*)\s*(?:\([^)]+\))?\s*$', dk).group(1))
        
        pred_team = re.sub(r'[+-]?\d+\.?\d*\s*$', '', pred).strip()
        dk_team = re.sub(r'[+-]?\d+\.?\d*\s*(?:\([^)]+\))?\s*$', '', dk).strip()
        
        if pred_team != dk_team:
            dk_spread = -dk_spread
        
        return abs(pred_spread - dk_spread)
    
    except Exception as e:
        print(f"Error on row {row.name}: {e}")
        print(f"  Predicted: {row.get('Predicted Spread', 'N/A')}")
        print(f"  DK Line: {row.get('DK Line (Time of Bet)', 'N/A')}")
        return np.nan

# Backtesting
 
# Replay one strategy over the whole history and tally the results. A "strategy" is
# three knobs: the minimum edge needed to place a bet at all, a cap on bets per day
# (take the highest-edge games when there are more than the cap), and a tier table
# mapping edge size to how many units to stake. Wins pay 0.91 units per unit risked,
# i.e. -110 odds -- a flat assumption that ignores the actual juice on each line.
def backtest_strategy(df, min_edge, max_bets_per_day, edge_tiers):
    total_profit = 0
    total_wagered = 0
    wins = 0
    total_bets = 0
    
    for date, day_games in df.groupby('Date'):
        # Only games clearing the edge floor, and only the best few that day.
        eligible = day_games[day_games['edge'] >= min_edge].copy()
        eligible = eligible.sort_values('edge', ascending=False).head(max_bets_per_day)
         # Stake by tier: use the units for the highest threshold this edge clears.
        for _, game in eligible.iterrows():
            units = edge_tiers[0][1]
            for threshold, u in sorted(edge_tiers, reverse=True):
                if game['edge'] >= threshold:
                    units = u
                    break
            
            if game['W/L'] == 'WIN':
                total_profit += units * 0.91
                wins += 1
            else:
                total_profit -= units
            
            total_wagered += units
            total_bets += 1
    
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    return {
        'profit': round(total_profit, 2),
        'wagered': round(total_wagered, 2),
        'roi': round(roi, 2),
        'win_rate': round(win_rate, 2),
        'num_bets': total_bets
    }

# Grid-search the three knobs over predefined sweeps and rank every combination by
# total profit. The presets are successively narrower search spaces: 'initial' casts
# a wide net, 'expanded' widens the daily bet caps, 'refined' zooms in on the range
# that kept performing, and 'full' runs all three and dedupes. Worth remembering this
# picks the best config IN-SAMPLE -- treat the winner as a hypothesis to confirm on
# out-of-sample dates, not a guaranteed edge (see note in the README).
def optimize_strategy(df, min_edges=None, max_bets=None, tier_configs=None, preset='full'):
    presets = {
        'initial': {
            'min_edges': [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            'max_bets': [5, 7, 9, 10, 11, 13, 15, 20],
            'tier_configs': [
                [(0, 1.0)],
                [(3.0, 0.5), (4.0, 1.0), (5.0, 1.5), (6.0, 2.0)],
                [(3.0, 1.0), (5.0, 1.5), (6.0, 2.0)],
                [(2.5, 0.5), (3.5, 1.0), (4.5, 1.5), (5.5, 2.0)],
                [(3.0, 0.5), (4.0, 1.0), (5.0, 2.0)],
                [(3.0, 1.0), (4.5, 2.0)],
                [(3.0, 0.5), (4.5, 1.0), (6.0, 1.5)],
            ]
        },
        'expanded': {
            'min_edges': [1.5, 2.0, 2.5, 3.0, 3.5],
            'max_bets': [5, 7, 10, 15, 20, 30],
            'tier_configs': [
                [(0, 0.5)],
                [(0, 1.0)],
                [(2.0, 0.5), (3.0, 1.0), (4.0, 1.5), (5.0, 2.0)],
                [(2.0, 1.0), (3.5, 1.5), (5.0, 2.0)],
                [(1.5, 0.5), (2.5, 1.0), (3.5, 1.5), (4.5, 2.0)],
                [(2.0, 0.5), (3.0, 1.0), (4.0, 2.0)],
                [(2.5, 1.0), (4.0, 2.0)],
                [(2.0, 0.5), (4.0, 1.0), (5.0, 2.0)],
                [(2.0, 1.0), (5.0, 2.0)],
            ]
        },
        'refined': {
            'min_edges': [2.0, 2.5, 2.75, 3.0],
            'max_bets': [20, 25, 30, 40, 50],
            'tier_configs': [
                [(0, 1.0)],
                [(2.5, 1.0), (4.0, 1.5)],
                [(2.5, 1.0), (5.0, 2.0)],
                [(2.5, 1.0), (3.5, 1.5), (5.0, 2.0)],
            ]
        }
    }
    
    if preset == 'full':
        all_results = []
        for p in ['initial', 'expanded', 'refined']:
            config = presets[p]
            for min_edge, max_bet, tiers in product(config['min_edges'], config['max_bets'], config['tier_configs']):
                result = backtest_strategy(df, min_edge, max_bet, tiers)
                result['min_edge'] = min_edge
                result['max_bets'] = max_bet
                result['tiers'] = str(tiers)
                all_results.append(result)
        return pd.DataFrame(all_results).drop_duplicates().sort_values('profit', ascending=False)
    # A named preset fills in any sweep the caller did not override
    elif preset in presets:
        config = presets[preset]
        min_edges = min_edges or config['min_edges']
        max_bets = max_bets or config['max_bets']
        tier_configs = tier_configs or config['tier_configs']
    else:
    # Otherwise the caller has to sweep themselves
        if not all([min_edges, max_bets, tier_configs]):
            raise ValueError("Must provide min_edges, max_bets, and tier_configs for custom optimization")
    
    results = []
    for min_edge, max_bet, tiers in product(min_edges, max_bets, tier_configs):
        result = backtest_strategy(df, min_edge, max_bet, tiers)
        result['min_edge'] = min_edge
        result['max_bets'] = max_bet
        result['tiers'] = str(tiers)
        results.append(result)
    
    return pd.DataFrame(results).sort_values('profit', ascending=False)


# Analysis / reporting
 
# Quick look at how edges are spread across the whole bet history.
def print_edge_distribution(df):
    print("\n" + "="*60)
    print("EDGE DISTRIBUTION")
    print("="*60)
    print(df['edge'].describe())
    print("\nEdge buckets:")
    print(pd.cut(df['edge'], bins=[0, 2, 3, 4, 5, 6, 100]).value_counts().sort_index())

# Win rate bucketed by edge size -- the key sanity check. If the model has a real
# edge, bigger predicted gaps should win at a higher clip; if win rate is flat across
# buckets, the "edge" isn't actually predictive.
def print_win_rate_by_edge(df):

    print("\n" + "="*60)
    print("WIN RATE BY EDGE BUCKET")
    print("="*60)
    
    df_copy = df.copy()
    df_copy['edge_bucket'] = pd.cut(df_copy['edge'], bins=[0, 2, 2.5, 3, 3.5, 4, 5, 100])
    grouped = df_copy.groupby('edge_bucket', observed=True).agg(
        count=('W/L', 'count'),
        wins=('W/L', lambda x: (x == 'WIN').sum())
    )
    grouped['win_rate'] = (grouped['wins'] / grouped['count'] * 100).round(2)
    print(grouped)


# Print the top N strategies from an optimization run.
def print_results(results_df, title="STRATEGIES BY PROFIT", n=20):
    print("\n" + "="*60)
    print(f"TOP {n} {title}")
    print("="*60)
    print(results_df[['min_edge', 'max_bets', 'profit', 'roi', 'win_rate', 'num_bets', 'tiers']].head(n).to_string(index=False))

# Main to run the optimization
def main(csv_path):    
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    print("\nCalculating edges...")
    df['edge'] = df.apply(calculate_edge, axis=1)
    
    before_count = len(df)
    df = df.dropna(subset=['edge'])
    after_count = len(df)
    if before_count != after_count:
        print(f"Dropped {before_count - after_count} rows with invalid data")
    
    df = df[['Date', 'W/L', 'edge']]
    
    print_edge_distribution(df)
    print_win_rate_by_edge(df)
    
    print("\n" + "="*60)
    print("RUNNING OPTIMIZATION (this may take a moment...)")
    print("="*60)
    
    results = optimize_strategy(df, preset='full')
    print_results(results, "ALL STRATEGIES")
    
    best = results.iloc[0]
    print("\n" + "="*60)
    print("BEST STRATEGY FOUND")
    print("="*60)
    print(f"  Min Edge: {best['min_edge']}")
    print(f"  Max Bets/Day: {best['max_bets']}")
    print(f"  Unit Tiers: {best['tiers']}")
    print(f"  Profit: {best['profit']} units")
    print(f"  ROI: {best['roi']}%")
    print(f"  Win Rate: {best['win_rate']}%")
    print(f"  Total Bets: {best['num_bets']}")
    
    return df, results

# Script entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backtest.py <path_to_csv>")
        print("Example: python backtest.py \"C:/Users/d/data/betting_tracker.csv\"")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    main(csv_path)