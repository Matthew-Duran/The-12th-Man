#import needed libraries
from bs4 import BeautifulSoup
import pandas as pd
import cloudscraper
import time
from io import StringIO


all_teams = [] #Create empty array of teams

scraper = cloudscraper.create_scraper()  # bypass Cloudflare

#Get the main Premier League stat page
html = scraper.get('https://fbref.com/en/comps/9/Premier-League-Stats').text
print("Length of main page HTML:", len(html))

soup = BeautifulSoup(html, 'lxml')
tables = soup.find_all('table', class_='stats_table')
print("Number of tables found on main page:", len(tables))

if not tables:
    raise ValueError("No tables found on main page.")

table = tables[0]

#Extract the team links
links = [l.get("href") for l in table.find_all('a') if l.get("href") and '/squads/' in l.get("href")]
print("Number of team links found:", len(links))

if not links:
    raise ValueError("No team links found.")

team_urls = [f"https://fbref.com{l}" for l in links]

#Scrape each team
for team_url in team_urls:
    print("Scraping team URL:", team_url)
    team_name = team_url.split("/")[-1].replace("-Stats", "")
    data = scraper.get(team_url).text

    if len(data) < 1000:
        print("Team page too short, skipping:", team_url)
        continue

    soup_team = BeautifulSoup(data, 'lxml')
    tables_team = soup_team.find_all('table', class_='stats_table')
    if not tables_team:
        print("No stats table found for team:", team_name)
        continue

    stats_table = tables_team[0]

    #Convert to DataFrame
    team_data = pd.read_html(StringIO(str(stats_table)))[0]

    #Drop multi-level columns
    if isinstance(team_data.columns, pd.MultiIndex):
        team_data.columns = team_data.columns.droplevel(0)

    #Add team name
    team_data["Team"] = team_name
    all_teams.append(team_data)
    print(f"Added data for team: {team_name}, rows: {len(team_data)}")

    time.sleep(5)  #To avoid being blocked (increase if having cloudflare issues)

#Collect all teams
if all_teams:
    stat_df = pd.concat(all_teams, ignore_index=True)
    stat_df.to_csv("Prem_Stats.csv", index=False)
    print("Scraping complete! CSV saved as Prem_Stats.csv")
else:
    print("No data collected. Exiting.")