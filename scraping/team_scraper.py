import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraping.log"), logging.StreamHandler()]
)
logger = logging.getLogger("team_scraper")

class BasketballReferenceTeamScraper:
    """
    Scraper for team data from Basketball-Reference
    """

    BASE_URL = "https://www.basketball-reference.com"

    def __init__(self, output_dir="../data/raw"):
        """
        Initialize the scraper

        Args:
            output_dir: Directory to save scraped data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_team_ids(self, season=2023):
        """
        Get list of team IDs for a given season

        Args:
            season: NBA season (year when season starts)

        Returns:
            Dictionary mapping team names to their IDs
        """
        url = f"{self.BASE_URL}/leagues/NBA_{season}.html"
        logger.info(f"Fetching team list from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the team tables (Eastern and Western conferences)
            team_ids = {}

            for conference in ['E', 'W']:
                conf_table = soup.find('table', {'id': f'confs_standings_{conference}'})
                if not conf_table:
                    continue

                for row in conf_table.find_all('tr')[1:]:  # Skip header row
                    name_cell = row.find('th', {'data-stat': 'team_name'})
                    if name_cell and name_cell.find('a'):
                        team_name = name_cell.text.strip()
                        team_url = name_cell.find('a')['href']
                        team_id = team_url.split('/')[2]
                        team_ids[team_name] = team_id

            logger.info(f"Found {len(team_ids)} teams for season {season}")
            return team_ids

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching team list: {e}")
            return {}

    def get_team_stats(self, team_id, season=2023):
        """
        Scrape team stats for a specific team in a given season

        Args:
            team_id: Basketball-Reference team ID
            season: NBA season (year when season starts)

        Returns:
            Dictionary of team stats
        """
        url = f"{self.BASE_URL}/teams/{team_id}/{season}.html"
        logger.info(f"Fetching team stats for {team_id} ({season}) from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Get team stats
            team_stats = {
                'team_id': team_id,
                'season': season
            }

            # Team record
            record_div = soup.find('div', {'data-template': 'Partials/Teams/Summary'})
            if record_div:
                record_text = record_div.find('p').text
                if 'Record:' in record_text:
                    record = record_text.split('Record:')[1].split(',')[0].strip()
                    wins, losses = record.split('-')
                    team_stats['wins'] = int(wins)
                    team_stats['losses'] = int(losses)

            # Team per game stats
            per_game_table = soup.find('table', {'id': 'per_game'})
            if per_game_table:
                team_row = per_game_table.find('tfoot').find('tr')
                for stat in team_row.find_all(['td', 'th']):
                    if stat.get('data-stat'):
                        stat_name = stat['data-stat']
                        stat_value = stat.text.strip()
                        team_stats[f'per_game_{stat_name}'] = stat_value

            # Team advanced stats
            advanced_table = soup.find('table', {'id': 'advanced'})
            if advanced_table:
                team_row = advanced_table.find('tfoot').find('tr')
                for stat in team_row.find_all(['td', 'th']):
                    if stat.get('data-stat'):
                        stat_name = stat['data-stat']
                        stat_value = stat.text.strip()
                        team_stats[f'advanced_{stat_name}'] = stat_value

            # Opponent stats
            opponent_table = soup.find('table', {'id': 'per_game-opponent'})
            if opponent_table:
                team_row = opponent_table.find('tfoot').find('tr')
                for stat in team_row.find_all(['td', 'th']):
                    if stat.get('data-stat'):
                        stat_name = stat['data-stat']
                        stat_value = stat.text.strip()
                        team_stats[f'opponent_{stat_name}'] = stat_value

            logger.info(f"Scraped stats for team {team_id}")
            return team_stats

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching team stats for {team_id}: {e}")
            return {}

    def get_team_schedule(self, team_id, season=2023):
        """
        Scrape team schedule for a specific team in a given season

        Args:
            team_id: Basketball-Reference team ID
            season: NBA season (year when season starts)

        Returns:
            DataFrame of team schedule
        """
        url = f"{self.BASE_URL}/teams/{team_id}/{season}_games.html"
        logger.info(f"Fetching schedule for {team_id} ({season}) from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the games table
            games_table = soup.find('table', {'id': 'games'})

            if not games_table:
                logger.warning(f"Could not find games table for team {team_id}")
                return pd.DataFrame()

            # Extract table headers
            headers = []
            for th in games_table.find_all('th'):
                if th.get('data-stat'):
                    headers.append(th['data-stat'])

            # Extract rows
            rows = []
            for row in games_table.find_all('tr')[1:]:  # Skip header row
                if 'thead' in row.get('class', []):
                    continue

                cols = row.find_all(['td', 'th'])
                if len(cols) > 0:
                    row_data = {}
                    for i, col in enumerate(cols):
                        if i < len(headers):
                            stat_name = headers[i]
                            row_data[stat_name] = col.text.strip()

                    # Add team ID and season
                    row_data['team_id'] = team_id
                    row_data['season'] = season

                    rows.append(row_data)

            # Create DataFrame
            df = pd.DataFrame(rows)

            logger.info(f"Scraped {len(df)} games for team {team_id}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching schedule for team {team_id}: {e}")
            return pd.DataFrame()

    def scrape_teams(self, teams_list=None, seasons=[2023], include_schedule=True, delay_range=(1, 3)):
        """
        Scrape data for multiple teams across seasons

        Args:
            teams_list: List of team IDs to scrape (if None, get all teams)
            seasons: List of seasons to scrape
            include_schedule: Whether to include team schedules
            delay_range: Tuple with min and max delay between requests

        Returns:
            Tuple of (team stats DataFrame, team schedules DataFrame)
        """
        all_team_stats = []
        all_schedules = []

        for season in seasons:
            if teams_list is None:
                # Get all teams in season
                team_ids = self.get_team_ids(season)
                teams_to_scrape = list(team_ids.values())
            else:
                teams_to_scrape = teams_list

            for team_id in teams_to_scrape:
                # Add random delay to avoid being blocked
                delay = random.uniform(delay_range[0], delay_range[1])
                time.sleep(delay)

                # Get team stats
                team_stats = self.get_team_stats(team_id, season)
                if team_stats:
                    all_team_stats.append(team_stats)

                # Get team schedule
                if include_schedule:
                    delay = random.uniform(delay_range[0], delay_range[1])
                    time.sleep(delay)

                    schedule = self.get_team_schedule(team_id, season)
                    if not schedule.empty:
                        all_schedules.append(schedule)

        # Create DataFrames
        stats_df = pd.DataFrame(all_team_stats) if all_team_stats else pd.DataFrame()
        schedule_df = pd.concat(all_schedules, ignore_index=True) if all_schedules else pd.DataFrame()

        logger.info(f"Total teams scraped: {len(stats_df)}")
        logger.info(f"Total games in schedules: {len(schedule_df)}")

        return stats_df, schedule_df

    def save_data(self, stats_df, schedule_df=None, stats_filename=None, schedule_filename=None):
        """
        Save scraped data to CSV

        Args:
            stats_df: Team stats DataFrame
            schedule_df: Team schedules DataFrame
            stats_filename: Stats output filename
            schedule_filename: Schedule output filename

        Returns:
            List of paths to saved files
        """
        saved_files = []

        # Save team stats
        if stats_df is not None and not stats_df.empty:
            if stats_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stats_filename = f"team_stats_{timestamp}.csv"

            stats_path = os.path.join(self.output_dir, stats_filename)
            stats_df.to_csv(stats_path, index=False)
            logger.info(f"Team stats saved to {stats_path}")
            saved_files.append(stats_path)

        # Save team schedules
        if schedule_df is not None and not schedule_df.empty:
            if schedule_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                schedule_filename = f"team_schedules_{timestamp}.csv"

            schedule_path = os.path.join(self.output_dir, schedule_filename)
            schedule_df.to_csv(schedule_path, index=False)
            logger.info(f"Team schedules saved to {schedule_path}")
            saved_files.append(schedule_path)

        return saved_files

def main():
    """
    Main function to run the scraper
    """
    # Example usage
    scraper = BasketballReferenceTeamScraper()

    # Scrape data for a few teams from 2022-2023 season
    top_teams = [
        'DEN',  # Denver Nuggets
        'MIA',  # Miami Heat
        'BOS',  # Boston Celtics
        'LAL',  # Los Angeles Lakers
        'GSW',  # Golden State Warriors
    ]

    stats_df, schedule_df = scraper.scrape_teams(
        teams_list=top_teams,
        seasons=[2023],
        include_schedule=True,
        delay_range=(1, 2)
    )

    # Save to CSV
    scraper.save_data(
        stats_df,
        schedule_df,
        stats_filename="top_teams_stats.csv",
        schedule_filename="top_teams_schedules.csv"
    )

if __name__ == "__main__":
    main()