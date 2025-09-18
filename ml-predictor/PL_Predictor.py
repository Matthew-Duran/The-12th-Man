# Premier League Full Table Predictor

#Imports
#Data Handling
import pandas as pd
import numpy as np
import glob
from fontTools.misc.macCreatorType import setMacCreatorAndType

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Evaluation
from sklearn.metrics import mean_squared_error

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Load all historical CSVs

#Seasons (2018-19 -> 2023-24)
csv_files = glob.glob("data/eng1_2018-*.csv") + \
            glob.glob("data/eng1_2019-*.csv") + \
            glob.glob("data/eng1_2020-*.csv") + \
            glob.glob("data/eng1_2021-*.csv") + \
            glob.glob("data/eng1_2022-*.csv") + \
            glob.glob("data/eng1_2023-*.csv")

#Combine files
dfs = [pd.read_csv(file) for file in csv_files]
matches = pd.concat(dfs, ignore_index=True)

#load Season 24-25 to format it
df_2024_25 = pd.read_csv("data/eng1_2024-25.csv")
df_2024_25 = df_2024_25.rename(columns={
    'HomeTeam': 'Team 1',
    'AwayTeam': 'Team 2',
    'FTHG': 'HomeGoals',
    'FTAG': 'AwayGoals'
})

#Create FT column
df_2024_25['FT'] = df_2024_25['HomeGoals'].astype(str) + '-' + df_2024_25['AwayGoals'].astype(str)
df_2024_25 = df_2024_25[['Date', 'Team 1', 'FT', 'Team 2']]

#Append 2024-25
matches = pd.concat([matches, df_2024_25], ignore_index=True)

# Parse new column into numeric goals

matches[['HomeGoals', 'AwayGoals']] = matches['FT'].str.split('-', expand=True)
matches['HomeGoals'] = matches['HomeGoals'].astype(int)
matches['AwayGoals'] = matches['AwayGoals'].astype(int)

# Calculate points per match

def result_points(row):
    if row['HomeGoals'] > row['AwayGoals']:
        return 3, 0 #Home, Away
    elif row['HomeGoals'] == row['AwayGoals']:
        return 1, 1 #Home, Away
    else:
        return 0, 3 #Home, Away

matches[['HomePoints', 'AwayPoints']] = matches.apply(lambda row: pd.Series(result_points(row)), axis=1)

# Aggregate team stats per season

season_features = []

#Extract season year from Date string
def extract_season(date):
    try:
        return str(date)[-4:]
    except:
        return str(date)[:4]

matches['Season'] = matches['Date'].apply(extract_season)

for season in matches['Season'].unique():
    season_df = matches[matches['Season'] == season]
    teams = pd.concat([season_df['Team 1'], season_df['Team 2']]).unique()

    for team in teams:
        home_games = season_df[season_df['Team 1'] == team]
        away_games = season_df[season_df['Team 2'] == team]

        home_points = home_games['HomePoints'].sum()
        away_points = away_games['AwayPoints'].sum()
        home_goals_for = home_games['HomeGoals'].sum()
        away_goals_for = away_games['AwayGoals'].sum()
        home_goals_against = home_games['AwayGoals'].sum()
        away_goals_against = away_games['HomeGoals'].sum()

        total_points = home_points + away_points

        goals_for = home_goals_for + away_goals_for

        goals_against = home_goals_against + away_goals_against

        goal_diff = goals_for - goals_against

        season_features.append({
            'Season': season,
            'Team': team,
            'Points': total_points,
            'GoalsFor': goals_for,
            'GoalsAgainst': goals_against,
            'GoalDiff': goal_diff,
        })

season_stats = pd.DataFrame(season_features)

# Create features from previous seasons

season_stats = season_stats.sort_values(['Team', 'Season'])
season_stats['PrevPoints'] = season_stats.groupby('Team')['Points'].shift(1)
season_stats['PrevGoalDiff'] = season_stats.groupby('Team')['GoalDiff'].shift(1)
season_stats['PrevGoalsFor'] = season_stats.groupby('Team')['GoalsFor'].shift(1)
season_stats['PrevGoalsAgainst'] = season_stats.groupby(['Team'])['GoalsAgainst'].shift(1)

# Drop rows if previous season info is missing
season_stats_ml = season_stats.dropna(subset=['PrevPoints', 'PrevGoalDiff', 'PrevGoalsFor', 'PrevGoalsAgainst'])


# Set features and target (2020 onward)


# IMPORTANT SIDE NOTE: Despite having the stats for 2018 onward, I believe 2020 onward gives a more accurate
#                      representation of modern team form and modern point totals.


# Define all teams currently in the Premier League
pl_2025_26_teams = [
    'Liverpool', 'Arsenal', 'Tottenham', 'Bournemouth', 'Chelsea',
    'Everton', 'Sunderland', 'Man City', 'Crystal Palace', 'Newcastle',
    'Fulham', 'Brentford', 'Brighton', 'Man United',
    "Nott'm Forest", 'Leeds', 'Burnley', 'West Ham',
    'Aston Villa', 'Wolves'
]

X = season_stats_ml[['PrevPoints', 'PrevGoalDiff', 'PrevGoalsFor', 'PrevGoalsAgainst']]
y = season_stats_ml['Points']

# Train on seasons 2020-21 -> 2023-24
train_idx = season_stats_ml['Season'].isin(['2020', '2021', '2022', '2023','2024'])
X_train = X[train_idx]
y_train = y[train_idx]

# Build X_test for all 20 teams in 2025-26
X_test_list = []
teams_for_test = []

for team in pl_2025_26_teams:
    # Find the last available season for team in 2020-2024
    team_history = season_stats_ml[(season_stats_ml['Team'] == team) &
                                   (season_stats_ml['Season'].isin(['2020', '2021', '2022', '2023','2024']))]
    if not team_history.empty:
        team_history = team_history.to_frame().T if isinstance(team_history, pd.Series) else team_history
        last_stats = team_history.sort_values('Season', ascending=False).iloc[0]

        X_test_list.append(last_stats[['PrevPoints', 'PrevGoalDiff', 'PrevGoalsFor', 'PrevGoalsAgainst']].values)
    else:
        # If team was not in the Premier League between 2020-2024, create a low point placeholder
        # (low because of a lack of experience in the Premier League).
        X_test_list.append([10, -20, 10, 30])
    teams_for_test.append(team)

X_test = pd.DataFrame(X_test_list, columns=['PrevPoints','PrevGoalDiff','PrevGoalsFor','PrevGoalsAgainst'])
teams_test = pd.Series(teams_for_test)

# Train model and predict

model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Generate full table

table_pred = pd.DataFrame({
    'Team': teams_test,
    'PredictedPoints': y_pred
})

# Sort descending
table_pred = table_pred.sort_values(by='PredictedPoints', ascending=False).reset_index(drop=True)
table_pred['PredictedRank'] = table_pred.index + 1

# Print the table
print("Predicted 2025-26 Premier League Table:")
table_pred.index = np.arange(1, len(table_pred)+1)
print(table_pred)

# Visualization

# Sort the table by points descending
table_pred_sorted = table_pred.sort_values('PredictedPoints', ascending=False)

norm_points = (table_pred_sorted['PredictedPoints'] - table_pred_sorted['PredictedPoints'].min()) / \
              (table_pred_sorted['PredictedPoints'].max() - table_pred_sorted['PredictedPoints'].min())

# Blue-to-red gradient
colors = [plt.cm.coolwarm(1 - val) for val in norm_points]

# Bar Plot
plt.figure(figsize=(12, 8))
sns.barplot(
    x='PredictedPoints',
    y='Team',
    hue = 'PredictedRank',
    data=table_pred_sorted,
    palette=colors
)

plt.xlabel('Predicted Points', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.title('Predicted 2025-26 Premier League Table', fontsize=16)
plt.xlim(0, table_pred_sorted['PredictedPoints'].max() + 5)
plt.tight_layout()
plt.show()