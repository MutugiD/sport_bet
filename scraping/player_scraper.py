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
logger = logging.getLogger("player_scraper")

class BasketballReferencePlayerScraper:
    """
    Scraper for player game logs from Basketball-Reference
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

    def get_player_ids(self, season=2023):
        """
        Get list of player IDs for a given season

        Args:
            season: NBA season (year when season starts)

        Returns:
            Dictionary mapping player names to their IDs
        """
        url = f"{self.BASE_URL}/leagues/NBA_{season}_per_game.html"
        logger.info(f"Fetching player list from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            players_table = soup.find('table', {'id': 'per_game_stats'})

            if not players_table:
                logger.error("Could not find players table on page")
                return {}

            player_ids = {}
            for row in players_table.find_all('tr')[1:]:  # Skip header row
                name_cell = row.find('td', {'data-stat': 'player'})
                if name_cell and name_cell.find('a'):
                    player_name = name_cell.text
                    player_url = name_cell.find('a')['href']
                    player_id = player_url.split('/')[-1].replace('.html', '')
                    player_ids[player_name] = player_id

            logger.info(f"Found {len(player_ids)} players for season {season}")
            return player_ids

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching player list: {e}")
            return {}

    def get_player_game_logs(self, player_id, season=2023):
        """
        Scrape game logs for a specific player in a given season

        Args:
            player_id: Basketball-Reference player ID
            season: NBA season (year when season starts)

        Returns:
            Pandas DataFrame of player's game logs
        """
        try:
            # Ensure player_id is properly formatted
            if not player_id or len(player_id) < 2:
                logger.error(f"Invalid player ID format: {player_id}")
                return pd.DataFrame()

            url = f"{self.BASE_URL}/players/{player_id[0]}/{player_id}/gamelog/{season}"
            logger.info(f"Fetching game logs for player {player_id} ({season}) from {url}")

            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the regular season game log table
            game_log_table = soup.find('table', {'id': 'pgl_basic'})

            if not game_log_table:
                logger.warning(f"Could not find game log table for player {player_id}")
                return pd.DataFrame()

            # Extract table headers
            headers = []
            for th in game_log_table.find_all('th'):
                if th.get('data-stat'):
                    headers.append(th['data-stat'])

            # Extract rows
            rows = []
            for row in game_log_table.find_all('tr')[1:]:  # Skip header row
                if 'thead' in row.get('class', []):
                    continue

                cols = row.find_all(['td', 'th'])
                if len(cols) > 0:
                    row_data = {}
                    for i, col in enumerate(cols):
                        if i < len(headers):
                            stat_name = headers[i]
                            row_data[stat_name] = col.text.strip()
                    rows.append(row_data)

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Add player ID and season columns
            if not df.empty:
                df['player_id'] = player_id
                df['season'] = season

            logger.info(f"Scraped {len(df)} games for player {player_id}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching game logs for player {player_id}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error processing player {player_id}: {e}")
            return pd.DataFrame()

    def scrape_players(self, players_list=None, seasons=[2023], delay_range=(1, 3)):
        """
        Scrape game logs for multiple players across seasons

        Args:
            players_list: List of player IDs to scrape (if None, get all players)
            seasons: List of seasons to scrape
            delay_range: Tuple with min and max delay between requests

        Returns:
            Combined DataFrame of all scraped game logs
        """
        all_game_logs = []

        for season in seasons:
            if players_list is None:
                # Get all players in season
                player_ids = self.get_player_ids(season)
                players_to_scrape = list(player_ids.values())
            else:
                players_to_scrape = players_list

            for player_id in players_to_scrape:
                # Add random delay to avoid being blocked
                delay = random.uniform(delay_range[0], delay_range[1])
                time.sleep(delay)

                game_logs = self.get_player_game_logs(player_id, season)
                if not game_logs.empty:
                    all_game_logs.append(game_logs)

        if all_game_logs:
            combined_df = pd.concat(all_game_logs, ignore_index=True)
            logger.info(f"Total games scraped: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No game logs were scraped")
            return pd.DataFrame()

    def save_data(self, df, filename=None):
        """
        Save scraped data to CSV

        Args:
            df: DataFrame to save
            filename: Output filename (if None, generate based on timestamp)

        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning("No data to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"player_data_{timestamp}.csv"

        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return output_path

def main():
    """
    Main function to run the scraper
    """
    # Example usage
    scraper = BasketballReferencePlayerScraper()

    # Scrape data for top players from 2022-2023 season
    top_players = [
        'jokicni01',  # Nikola Jokic
        'embiijo01',  # Joel Embiid
        'antetgi01',  # Giannis Antetokounmpo
        'doncilu01',  # Luka Doncic
        'tatumja01',  # Jayson Tatum
    ]

    game_logs = scraper.scrape_players(
        players_list=top_players,
        seasons=[2023, 2022],
        delay_range=(1, 2)
    )

    # Save to CSV
    scraper.save_data(game_logs, "top_players_game_logs.csv")

if __name__ == "__main__":
    main()