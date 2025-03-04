import requests
import pandas as pd
import time
import random
import os
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraping.log"), logging.StreamHandler()]
)
logger = logging.getLogger("odds_scraper")

class OddsScraper:
    """
    Scraper for NBA player prop betting odds

    Note: This uses a mock API for demonstration purposes.
    In a real implementation, you would need to:
    1. Use a proper sports betting API (like The Odds API)
    2. Or scrape from sportsbooks directly (requires more complex handling)
    """

    def __init__(self, api_key=None, output_dir="../data/raw"):
        """
        Initialize the scraper

        Args:
            api_key: API key for odds service (if applicable)
            output_dir: Directory to save scraped data
        """
        self.api_key = api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_player_props(self, date=None):
        """
        Get player props for a specific date

        Args:
            date: Date to get props for (YYYY-MM-DD format)

        Returns:
            DataFrame of player props
        """
        logger.info(f"Fetching player props for date: {date}")

        # In a real implementation, you would call an API or scrape a website
        # For this example, we'll generate mock data

        mock_data = self._generate_mock_props(date)

        props_df = pd.DataFrame(mock_data)
        logger.info(f"Retrieved {len(props_df)} player props")

        return props_df

    def _generate_mock_props(self, date):
        """
        Generate mock player props data

        Args:
            date: Date for the mock data

        Returns:
            List of dictionaries with mock props
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # List of player names for mock data
        players = [
            "Nikola Jokic", "Joel Embiid", "Giannis Antetokounmpo",
            "Luka Doncic", "Jayson Tatum", "LeBron James",
            "Stephen Curry", "Kevin Durant", "Devin Booker", "Ja Morant"
        ]

        # List of sportsbooks
        sportsbooks = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "PointsBet"]

        # Generate mock props
        mock_props = []

        for player in players:
            # Generate a baseline points prop
            baseline_points = random.uniform(18.5, 29.5)
            baseline_points = round(baseline_points * 2) / 2  # Round to nearest 0.5

            for sportsbook in sportsbooks:
                # Add some variance to each sportsbook line
                points_line = baseline_points + random.choice([-1.0, -0.5, 0, 0.5, 1.0])
                points_line = round(points_line * 2) / 2  # Round to nearest 0.5

                # Generate odds for over/under
                over_american = random.randint(-125, -105)
                under_american = random.randint(-125, -105)

                # Convert to implied probability
                over_implied = abs(over_american) / (abs(over_american) + 100) if over_american < 0 else 100 / (abs(over_american) + 100)
                under_implied = abs(under_american) / (abs(under_american) + 100) if under_american < 0 else 100 / (abs(under_american) + 100)

                # Create prop entry
                prop = {
                    "date": date,
                    "player": player,
                    "market": "Points",
                    "line": points_line,
                    "over_odds_american": over_american,
                    "under_odds_american": under_american,
                    "over_odds_decimal": round(1 + (1 / over_implied), 2),
                    "under_odds_decimal": round(1 + (1 / under_implied), 2),
                    "over_implied_probability": round(over_implied, 4),
                    "under_implied_probability": round(under_implied, 4),
                    "sportsbook": sportsbook
                }

                mock_props.append(prop)

        return mock_props

    def scrape_date_range(self, start_date, end_date):
        """
        Scrape player props for a range of dates

        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            DataFrame of player props for the date range
        """
        logger.info(f"Scraping player props from {start_date} to {end_date}")

        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_props = []
        current_dt = start_dt

        while current_dt <= end_dt:
            current_date = current_dt.strftime("%Y-%m-%d")

            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))

            # Get props for current date
            props = self.get_player_props(current_date)
            if not props.empty:
                all_props.append(props)

            # Move to next day
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            current_dt = datetime(current_dt.year, current_dt.month, current_dt.day + 1)

        # Combine all props
        if all_props:
            combined_df = pd.concat(all_props, ignore_index=True)
            logger.info(f"Total props scraped: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No props were scraped")
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
            filename = f"player_props_{timestamp}.csv"

        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return output_path

def main():
    """
    Main function to run the scraper
    """
    # Example usage
    scraper = OddsScraper()

    # For a real implementation, you would use actual dates
    # This is just an example using a 5-day range
    today = datetime.now().strftime("%Y-%m-%d")

    # In practice, for historical odds, you'd use past dates
    # For demonstration, we're using today as a reference point
    props_df = scraper.get_player_props(today)

    # Save to CSV
    scraper.save_data(props_df, "player_props_sample.csv")

    # Example of date range scraping (would be used for historical data)
    # props_df = scraper.scrape_date_range("2023-01-01", "2023-01-05")
    # scraper.save_data(props_df, "player_props_range.csv")

if __name__ == "__main__":
    main()