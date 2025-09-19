from bs4 import BeautifulSoup
import pandas as pd
import cloudscraper
import time

all_teams = []

scraper = cloudscraper.create_scraper()  # bypass Cloudflare
html = scraper.get('https://fbref.com/en/comps/9/Premier-League-Stats').text
soup = BeautifulSoup(html, 'lxml')
table = soup.find_all('table', class_='stats_table')[0]

links = table.find_all('a')
links = [l.get("href") for l in links if '/squads/' in l]
team_urls = [f"https://fbref.com{l}" for l in links]

for team_url in team_urls:
    team_name = team_url.split("/")[-1].replace("-Stats", "")
    data = scraper.get(team_url).text
    soup = BeautifulSoup(data, 'lxml')
    stats_table = soup.find_all('table', class_='stats_table')[0]

    # Convert table to DataFrame
    team_data = pd.read_html(str(stats_table))[0]

    # If columns are multi-level, drop the first level
    if isinstance(team_data.columns, pd.MultiIndex):
        team_data.columns = team_data.columns.droplevel(0)

    # Add team name
    team_data["Team"] = team_name
    all_teams.append(team_data)
    time.sleep(5)

# Concatenate all teams and reset index
stat_df = pd.concat(all_teams, ignore_index=True)

# Export CSV
stat_df.to_csv("stats.csv", index=False)
