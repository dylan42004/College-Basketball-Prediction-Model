# The live predictor. Loads the trained Huber model and runs it against current-
# season team ratings to produce a margin/spread for any matchup. Two ways in:
# scrape a given day's slate off Barttorvik and predict everything on it, or enter
# a single matchup by hand (for neutral-site and tournament games the schedule
# scrape won't have).
#
# Training pulls from getgamestats, which plain requests can reach. The current
# ratings live on trank.php, which sits behind a JS "verifying your browser" gate,
# so everything here goes through headless Selenium instead of requests.
import pandas as pd
import numpy as np
import json
import time
import re
import joblib
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# Links to current rankings and schedule
URL_TRANK_JSON = "https://barttorvik.com/trank.php?year=2026&json=1"
URL_SCHEDULE = "https://barttorvik.com/schedule.php?date={date}&conlimit="

TEAM_NAME_MAP = {}

TRANK_COLUMNS = [
    'team', 'adjoe', 'adjde', 'barthag', 'record', 'wins', 'games', 'efg%', 'efgD%',
    'ftr', 'ftrD', 'tor%', 'torD%', 'orb%', 'orbD%', 'adjt', '2p%', '2pD%', '3p%', '3pD%',
    'blank1', 'blank2', 'blank3', 'blank4', '3PR', '3PRD', '26', '27', '28', '29', '30', '31',
    '32', '33', '34', '35', '36'
]

TRANK_DROP_COLS = [
    'barthag', 'record', 'wins', 'games', 'adjt', 'blank1', 'blank2', 'blank3', 'blank4',
    '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36'
]


# Fetch current rankings
def fetch_trank_json_selenium(url= URL_TRANK_JSON, timeout= 60):
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    try:
        driver.get(url)

        def page_ready(d):
            title = (d.title or "").lower()
            cur = (d.current_url or "").lower()
            src = (d.page_source or "").lower()
            if "verifying" in title or "js_required" in cur or "verifying your browser" in src:
                return False
            return True

        WebDriverWait(driver, timeout).until(page_ready)
        time.sleep(1)

        txt = ""
        pres = driver.find_elements(By.TAG_NAME, "pre")
        if pres:
            txt = pres[0].text.strip()
        else:
            txt = driver.execute_script("return document.body.innerText;").strip()

        if not (txt.startswith("{") or txt.startswith("[")):
            raise ValueError("Did not receive JSON text. First 300 chars:\n" + txt[:300])

        data = json.loads(txt)
        return pd.DataFrame(data)

    finally:
        driver.quit()

# Name the columns and drop the ones the model doesn't use
def clean_trank_df(df):
    d = df.copy()
    d.columns = TRANK_COLUMNS
    d = d.drop(columns=TRANK_DROP_COLS)
    return d

# Fetch and clean in one call
def get_current_stats(url = URL_TRANK_JSON):
    raw_df = fetch_trank_json_selenium(url)
    return clean_trank_df(raw_df)

# Schedule scraping

# Last-resort splitter for the schedule text. Each schedule row mashes the home
# team and Barttorvik's own predicted winner together with no clean delimiter (e.g.
# "Georgia Georgia", "Tennessee St. Memphis"). With no known-team set to check
# against, this guesses where the home name ends using common name endings (St.,
# State, Tech, state names...), and if that fails just splits down the middle.
def self_parse_home_team(before_words):
    
    home_team = None
    for trank_len in range(1, min(5, len(before_words))):
        potential_home = ' '.join(before_words[:-trank_len])
        
        if potential_home:
            # Check if home team ends with common suffix
            endings = ['St.', 'St', 'State', 'Tech', 'A&M', 'OH', 'FL', 'MD',
                       'Valley', 'Green', 'Carolina', 'Diego', 'Beach', 'Jose',
                       'Island', 'Dame', 'Bluff', 'Christian', 'Central', 'Southern',
                       'Northern', 'Eastern', 'Western', 'Kentucky', 'Illinois',
                       'Indiana', 'Michigan', 'Alabama', 'Florida', 'Georgia',
                       'Tennessee', 'Louisiana', 'Mississippi', 'Colorado', 'Arizona']
            
            last_word = before_words[-(trank_len+1)] if len(before_words) > trank_len else ''
            
            if last_word in endings or last_word.endswith('.'):
                home_team = potential_home
                break
    
    # Fallback: take first half
    if not home_team:
        mid = len(before_words) // 2
        home_team = ' '.join(before_words[:max(1, mid)])
    
    return home_team

# Scrape one day's slate off the schedule page. Barttorvik renders it as a text
# table with TV networks and T-Rank picks crammed into each row, so there's no
# clean DOM to grab -- the parsing below is heuristic. valid_teams, if passed, is
# the team-name set used to pin down where a home-team name ends.
# date is YYYYMMDD (e.g. "20260117"). Returns a list of {time, away_team, home_team}.
def fetch_schedule_selenium(date, timeout= 60, valid_teams= None):
    url = URL_SCHEDULE.format(date=date)
    
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)

        def page_ready(d):
            title = (d.title or "").lower()
            cur = (d.current_url or "").lower()
            src = (d.page_source or "").lower()
            if "verifying" in title or "js_required" in cur or "verifying your browser" in src:
                return False
            return True

        WebDriverWait(driver, timeout).until(page_ready)
        time.sleep(3)  # Let table render

        games = []
        
        # Get table text and split into lines
        tables = driver.find_elements(By.TAG_NAME, "table")
        if not tables:
            return games
        
        table_text = tables[0].text
        lines = table_text.split('\n')
        
        # TV networks to filter out (including + variants)
        tv_networks = [
            'ESPN+', 'ESPN2', 'ESPNU', 'ESPN', 'FOX', 'FS1', 'FS2', 
            'CBS', 'CBSSN', 'ABC', 'BTN', 'SEC', 'ACC', 'Peacock', 
            'TNT', 'TBS', 'Big12', 'NBC', 'USA', 'TruTV', 'ACCN', 
            'SECN', 'NEC', 'Front', 'Row', 'Network', 'FloSports',
            'Big', 'Ten', 'Eleven', 'Plus', 'Stadium', 'The', 'CW',
            'WDAY', 'Xtra', 'Summit', 'League', 'MWN', 'SNY', 'DSN',
            'WITN-TV', 'WITN', 'TV', 'Flo', 'Sports', 'WAC', 'MAAC',
            'SWX', 'RSN', 'Bally', 'ROOT', 'NESN', 'MSG', 'YES',
            'Longhorn', 'LHN', 'MASN', 'Monumental', 'SportsNet'
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for time pattern at start of line (e.g., "04:00 PM")
            time_match = re.match(r'^(\d{1,2}:\d{2}\s*[AP]M)\s+(.+)$', line)
            if time_match:
                game_time = time_match.group(1)
                away_part = time_match.group(2)
                
                # Remove ranking number from away team
                away_team = re.sub(r'^\d+\s+', '', away_part).strip()
                
                # Next line should have "at {rank} {home_team} {TV} ..."
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    
                    # Pattern: "at 36 Georgia ESPN2 Georgia -1.4"
                    at_match = re.match(r'^at\s+(\d+\s+)?(.+)$', next_line)
                    if at_match:
                        home_part = at_match.group(2)
                        
                        # First, try to find a TV network
                        tv_index = -1
                        words = home_part.split()
                        
                        for idx, w in enumerate(words):
                            if w.rstrip('+') in tv_networks or w in tv_networks:
                                tv_index = idx
                                break
                        
                        if tv_index > 0:
                            # Found TV network - home team is everything before it
                            home_team = ' '.join(words[:tv_index]).strip()
                        else:
                            # No TV network found
                            # Format: "{home_team} {trank_winner} {spread} {score} {pct} {ttq}"
                            # Work backwards from the spread to find where T-Rank winner starts
                            
                            # Find the spread (e.g., "-3.5" or "+2.1")
                            spread_match = re.search(r'(-?\d+\.\d)', home_part)
                            
                            if spread_match:
                                before_spread = home_part[:spread_match.start()].strip()
                                before_words = before_spread.split()
                                
                                # If we have valid_teams, use it to find the correct split
                                if valid_teams and len(before_words) >= 2:
                                    home_team = None
                                    # Try progressively longer home team names
                                    for i in range(1, len(before_words)):
                                        potential_home = ' '.join(before_words[:i])
                                        if potential_home in valid_teams:
                                            home_team = potential_home
                                            # Don't break - keep looking for longer matches
                                        # Also try with period added (e.g., "Tennessee St" -> "Tennessee St.")
                                        elif potential_home + '.' in valid_teams:
                                            home_team = potential_home + '.'
                                    
                                    # If no exact match, try with trailing period stripped
                                    if not home_team:
                                        for i in range(1, len(before_words)):
                                            potential_home = ' '.join(before_words[:i]).rstrip('.')
                                            if potential_home in valid_teams:
                                                home_team = potential_home
                                                break
                                            # Try adding period
                                            elif potential_home + '.' in valid_teams:
                                                home_team = potential_home + '.'
                                                break
                                    
                                    # Still no match - fallback to heuristic
                                    if not home_team:
                                        home_team = self_parse_home_team(before_words)
                                else:
                                    home_team = self_parse_home_team(before_words)
                            else:
                                # No spread found, just take all words (shouldn't happen)
                                home_team = home_part
                        
                        # Clean up - but preserve periods that are part of team names like "St."
                        # Only strip trailing period if it's not part of an abbreviation
                        if home_team and home_team.endswith('.') and not home_team.endswith('St.'):
                            home_team = home_team.rstrip('.')
                        
                        if away_team and home_team:
                            games.append({
                                'time': game_time,
                                'away_team': away_team,
                                'home_team': home_team,
                            })
                        
                        i += 2  # Skip the "at" line
                        continue
            
            i += 1
        
        return games

    finally:
        driver.quit()


# Run the scrape and return df sorted by time
def fetch_schedule(date, valid_teams = None):
    games = fetch_schedule_selenium(date, valid_teams=valid_teams)
    df = pd.DataFrame(games)
    
    if df.empty:
        return df
    
    # Sort by time
    def time_to_minutes(t):
        try:
            match = re.match(r'(\d{1,2}):(\d{2})\s*([AP]M)', t)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                ampm = match.group(3)
                
                if ampm == 'PM' and hour != 12:
                    hour += 12
                elif ampm == 'AM' and hour == 12:
                    hour = 0
                
                return hour * 60 + minute
        except:
            pass
        return 9999  # Put unparseable times at the end
    
    df['_sort_key'] = df['time'].apply(time_to_minutes)
    df = df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)
    
    return df


# Predict a single matchup's margin from team1's perspective (positive = team1
# favored by that much, negative = team2). This rebuilds the EXACT differential
# features the model trained on, but from current T-Rank stats instead of
# historical game logs. 
def predict_game(
    teams_df, team1, team2, venue, model = None, feat_cols = None,
        model_path = "huber_margin_model.pkl", features_path = "feature_cols_huber.pkl"):
    if model is None:
        model = joblib.load(model_path)
    if feat_cols is None:
        feat_cols = joblib.load(features_path)

    team = teams_df[teams_df['team'] == team1].iloc[0]
    opp = teams_df[teams_df['team'] == team2].iloc[0]
    hca = {'H': 1, 'N': 0, 'A': -1}[venue]

    game = {
        'hca': hca,
        'net_rtg_diff': (team['adjoe'] - team['adjde']) - (opp['adjoe'] - opp['adjde']),
        'efg%_diff': team['efg%'] + opp['efgD%'] - team['efgD%'] - opp['efg%'],
        'tor%_diff': (opp['tor%'] + team['torD%']) - (team['tor%'] + opp['torD%']),
        'orb%_diff': team['orb%'] - team['orbD%'] + opp['orbD%'] - opp['orb%'],
        'ftr_diff': team['ftr'] - team['ftrD'] + opp['ftrD'] - opp['ftr'],
        '2p%_diff': (team['2p%'] - team['2pD%'] + opp['2pD%'] - opp['2p%']) / 100,
        '3p%_diff': (team['3p%'] - team['3pD%'] + opp['3pD%'] - opp['3p%']) / 100,
        '3PR_diff': (team['3PR'] - team['3PRD'] + opp['3PRD'] - opp['3PR']) / 100,
    }

    game_df = pd.DataFrame([game])[feat_cols] # reorder to the saved feature order
    return model.predict(game_df)[0]

# Predict a list of matchups, catching per game errors so one bad row
# doesn't take down the whole batch
def predict_games_batch(teams_df, games, model_path = "huber_margin_model.pkl", features_path = "feature_cols_huber.pkl"):
    model = joblib.load(model_path)
    feat_cols = joblib.load(features_path)

    results = []
    for team1, team2, venue in games:
        try:
            margin = predict_game(teams_df, team1, team2, venue, model, feat_cols)
            results.append({
                'team1': team1,
                'team2': team2,
                'venue': venue,
                'predicted_margin': margin,
                'predicted_winner': team1 if margin > 0 else team2,
                'predicted_spread': f"{team1} {margin:.1f}" if margin < 0 else f"{team1} -{margin:.1f}"
            })
        except Exception as e:
            results.append({
                'team1': team1,
                'team2': team2,
                'venue': venue,
                'predicted_margin': np.nan,
                'predicted_winner': 'ERROR',
                'predicted_spread': str(e)
            })

    return pd.DataFrame(results)


# Console interface
# Interactive manual entry - type two teams and a venue get a prediction. Used for neutral games, conference tournament
# games where scrape doesn't work, etc.

def prediction_console(teams_df = None, model_path = "huber_margin_model.pkl", features_path = "feature_cols_huber.pkl"):
    if teams_df is None:
        print("Fetching current team stats...")
        teams_df = get_current_stats()

    model = joblib.load(model_path)
    feat_cols = joblib.load(features_path)

    print("\nCollege Basketball Margin Predictor (Manual Entry)")
    print("Type 'quit' to exit\n")

    while True:
        team1 = input("Team 1: ").strip()
        if team1.lower() == 'quit':
            break

        team2 = input("Team 2: ").strip()
        if team2.lower() == 'quit':
            break

        venue = input("Venue (H/N/A): ").strip().upper()

        if team1 not in teams_df['team'].values:
            print(f"'{team1}' not found\n")
            continue
        if team2 not in teams_df['team'].values:
            print(f"'{team2}' not found\n")
            continue
        if venue not in ['H', 'N', 'A']:
            print("Venue must be H, N, or A\n")
            continue

        margin = predict_game(teams_df, team1, team2, venue, model, feat_cols)

        if margin > 0:
            print(f"\n>>> {team1} by {margin:.2f}\n")
        else:
            print(f"\n>>> {team2} by {-margin:.2f}\n")

# Predict every game on a date's slate. Fetches the schedule, then reconciles each
# schedule team name against the T-Rank stats table -- exact match first, then a
# substring match, then a first-word match -- because the two feeds sometimes spell
# teams differently. Names that can't be resolved are flagged rather than dropped,
# and every prediction is made from the home team's perspective.
def predict_daily(date = None, teams_df = None,
        model_path = "huber_margin_model.pkl", features_path = "feature_cols_huber.pkl"):
    if date is None:
        today = datetime.now().strftime("%Y%m%d")
        date = input(f"Enter date (YYYYMMDD) [{today}]: ").strip()
        if not date:
            date = today # default to today if no date entered
    
    print(f"\nFetching schedule for {date}...")
    schedule_df = fetch_schedule(date)
    
    if schedule_df.empty:
        print("No games found for this date.")
        return pd.DataFrame()
    
    print(f"Found {len(schedule_df)} games")
    
    # Get team stats
    if teams_df is None:
        print("Fetching current team stats...")
        teams_df = get_current_stats()
    
    # Load model
    model = joblib.load(model_path)
    feat_cols = joblib.load(features_path)
    
    # Predict each game
    results = []
    for _, game in schedule_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        
        # Apply team name mapping
        away_mapped = TEAM_NAME_MAP.get(away, away)
        home_mapped = TEAM_NAME_MAP.get(home, home)
        
        # Try to match team names
        away_match = teams_df[teams_df['team'] == away_mapped]
        home_match = teams_df[teams_df['team'] == home_mapped]
        
        if away_match.empty or home_match.empty:
            # Try partial matching
            if away_match.empty:
                away_matches = teams_df[teams_df['team'].str.contains(away_mapped, case=False, na=False, regex=False)]
                if len(away_matches) == 1:
                    away_mapped = away_matches.iloc[0]['team']
                    away_match = away_matches
                elif away_match.empty:
                    # Try first word only for multi-word teams
                    first_word = away_mapped.split()[0]
                    if len(first_word) > 3:  # Avoid matching on "St." etc
                        away_matches = teams_df[teams_df['team'].str.startswith(first_word, na=False)]
                        if len(away_matches) == 1:
                            away_mapped = away_matches.iloc[0]['team']
                            away_match = away_matches
            
            if home_match.empty:
                home_matches = teams_df[teams_df['team'].str.contains(home_mapped, case=False, na=False, regex=False)]
                if len(home_matches) == 1:
                    home_mapped = home_matches.iloc[0]['team']
                    home_match = home_matches
                elif home_match.empty:
                    # Try first word only for multi-word teams
                    first_word = home_mapped.split()[0]
                    if len(first_word) > 3:
                        home_matches = teams_df[teams_df['team'].str.startswith(first_word, na=False)]
                        if len(home_matches) == 1:
                            home_mapped = home_matches.iloc[0]['team']
                            home_match = home_matches
        
        if away_match.empty or home_match.empty:
            results.append({
                'time': game['time'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'predicted_spread': 'TEAM NOT FOUND',
                'home_margin': np.nan,
            })
            continue
        
        try:
            # Predict from home team perspective
            margin = predict_game(teams_df, home_mapped, away_mapped, 'H', model, feat_cols)
            
            if margin > 0:
                spread_str = f"{home_mapped} -{margin:.2f}"
            else:
                spread_str = f"{home_mapped} +{-margin:.2f}"
            
            results.append({
                'time': game['time'],
                'away_team': away_mapped,
                'home_team': home_mapped,
                'predicted_spread': spread_str,
                'home_margin': margin,
            })
        except Exception as e:
            results.append({
                'time': game['time'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'predicted_spread': f'ERROR: {str(e)}',
                'home_margin': np.nan,
            })
    
    results_df = pd.DataFrame(results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTIONS FOR {date}")
    print(f"{'='*60}\n")
    
    for _, row in results_df.iterrows():
        print(f"{row['time']:>8}  {row['away_team']:>20} at {row['home_team']:<20}  |  {row['predicted_spread']}")
    
    print(f"\n{'='*60}")
    
    return results_df

# Top level menu, caches team stats after the first fetch so switching options
# doesn't re-scrape every time
def main_menu():
    print("\n" + "="*50)
    print("COLLEGE BASKETBALL SPREAD PREDICTOR")
    print("="*50)
    print("\n1. Predict today's/specific date games")
    print("2. Manual prediction (neutral sites, tournaments)")
    print("3. Quit")
    
    teams_df = None
    
    while True:
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice == '1':
            if teams_df is None:
                print("Fetching current team stats...")
                teams_df = get_current_stats()
            predict_daily(teams_df=teams_df)
        
        elif choice == '2':
            if teams_df is None:
                print("Fetching current team stats...")
                teams_df = get_current_stats()
            prediction_console(teams_df=teams_df)
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please enter 1, 2, or 3.")

# script entry point
if __name__ == "__main__":
    main_menu()