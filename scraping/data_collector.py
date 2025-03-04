import argparse
import logging
import os
from datetime import datetime

from scraping.player_scraper import BasketballReferencePlayerScraper
from scraping.team_scraper import BasketballReferenceTeamScraper
from scraping.odds_scraper import OddsScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraping.log"), logging.StreamHandler()]
)
logger = logging.getLogger("data_collector")

class NBADataCollector:
    """
    Unified workflow for collecting NBA data from multiple sources
    """

    def __init__(self, output_dir="../data/raw"):
        """
        Initialize the data collector

        Args:
            output_dir: Directory to save scraped data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize scrapers
        self.player_scraper = BasketballReferencePlayerScraper(output_dir=output_dir)
        self.team_scraper = BasketballReferenceTeamScraper(output_dir=output_dir)
        self.odds_scraper = OddsScraper(output_dir=output_dir)

        # Common player name to ID mapping
        self.player_id_map = {
            'Nikola Jokic': 'jokicni01',
            'Joel Embiid': 'embiijo01',
            'Giannis Antetokounmpo': 'antetgi01',
            'Luka Doncic': 'doncilu01',
            'Jayson Tatum': 'tatumja01',
            'LeBron James': 'jamesle01',
            'Stephen Curry': 'curryst01',
            'Kevin Durant': 'duranke01',
            'Devin Booker': 'bookede01',
            'Ja Morant': 'moranja01',
            'Damian Lillard': 'lillada01',
            'Trae Young': 'youngtr01',
            'Donovan Mitchell': 'mitchdo01',
            'Jaylen Brown': 'brownja02',
            'Bam Adebayo': 'adebaba01',
            'Anthony Edwards': 'edwaran01',
            'Shai Gilgeous-Alexander': 'gilgesh01',
            'Anthony Davis': 'davisan02',
            'Jimmy Butler': 'butleji01',
            'Zion Williamson': 'willizi01'
        }

    def map_player_names_to_ids(self, player_names):
        """
        Convert player names to Basketball Reference IDs

        Args:
            player_names: List of player names

        Returns:
            List of player IDs
        """
        player_ids = []

        # If we received a list of player names, try to map them to IDs
        if player_names:
            for name in player_names:
                if name in self.player_id_map:
                    player_ids.append(self.player_id_map[name])
                else:
                    logger.warning(f"Could not find ID for player: {name}. Skipping.")

        # If no valid player IDs were found, use defaults
        if not player_ids:
            logger.info("Using default player IDs")
            player_ids = list(self.player_id_map.values())[:5]  # Use first 5 players as default

        return player_ids

    def collect_player_data(self, players=None, seasons=[2023], save=True):
        """
        Collect player game logs

        Args:
            players: List of player names or IDs
            seasons: List of seasons to scrape
            save: Whether to save the data

        Returns:
            DataFrame of player game logs
        """
        logger.info("Collecting player game logs")

        # Convert player names to IDs if needed
        player_ids = self.map_player_names_to_ids(players)

        game_logs = self.player_scraper.scrape_players(
            players_list=player_ids,
            seasons=seasons,
            delay_range=(1, 2)
        )

        if save and not game_logs.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"player_data_{timestamp}.csv"
            self.player_scraper.save_data(game_logs, filename)

        return game_logs

    def collect_team_data(self, teams=None, seasons=[2023], include_schedule=True, save=True):
        """
        Collect team stats and schedules

        Args:
            teams: List of team IDs (if None, get top teams)
            seasons: List of seasons to scrape
            include_schedule: Whether to include team schedules
            save: Whether to save the data

        Returns:
            Tuple of (team stats DataFrame, team schedules DataFrame)
        """
        logger.info("Collecting team data")

        if teams is None:
            # Default to top teams if none specified
            teams = [
                'DEN',  # Denver Nuggets
                'MIA',  # Miami Heat
                'BOS',  # Boston Celtics
                'LAL',  # Los Angeles Lakers
                'GSW',  # Golden State Warriors
                'PHI',  # Philadelphia 76ers
                'MIL',  # Milwaukee Bucks
                'DAL',  # Dallas Mavericks
                'PHO',  # Phoenix Suns
                'MEM',  # Memphis Grizzlies
            ]

        stats_df, schedule_df = self.team_scraper.scrape_teams(
            teams_list=teams,
            seasons=seasons,
            include_schedule=include_schedule,
            delay_range=(1, 2)
        )

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            stats_filename = f"team_stats_{timestamp}.csv"
            schedule_filename = f"team_schedules_{timestamp}.csv"

            self.team_scraper.save_data(
                stats_df=stats_df,
                schedule_df=schedule_df,
                stats_filename=stats_filename,
                schedule_filename=schedule_filename
            )

        return stats_df, schedule_df

    def collect_odds_data(self, start_date=None, end_date=None, save=True):
        """
        Collect player prop odds

        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            save: Whether to save the data

        Returns:
            DataFrame of player props
        """
        logger.info("Collecting odds data")

        if start_date and end_date:
            props_df = self.odds_scraper.scrape_date_range(start_date, end_date)
        else:
            # Just get today's odds
            today = datetime.now().strftime("%Y-%m-%d")
            props_df = self.odds_scraper.get_player_props(today)

        if save and not props_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"player_props_{timestamp}.csv"
            self.odds_scraper.save_data(props_df, filename)

        return props_df

    def collect_all_data(self, seasons=[2023], odds_days=5, save=True):
        """
        Collect all types of data

        Args:
            seasons: List of seasons to scrape player and team data for
            odds_days: Number of days to scrape odds for (from today)
            save: Whether to save the data

        Returns:
            Tuple of (player data, team stats, team schedules, odds data)
        """
        logger.info(f"Collecting all data for seasons: {seasons}")

        # Player data
        player_data = self.collect_player_data(seasons=seasons, save=save)

        # Team data
        team_stats, team_schedules = self.collect_team_data(seasons=seasons, save=save)

        # Odds data (mock for demonstration)
        odds_data = self.collect_odds_data(save=save)

        return player_data, team_stats, team_schedules, odds_data

    def validate_data(self, player_data, team_stats, team_schedules, odds_data):
        """
        Validate the collected data

        Args:
            player_data: Player game logs DataFrame
            team_stats: Team stats DataFrame
            team_schedules: Team schedules DataFrame
            odds_data: Player props DataFrame

        Returns:
            Boolean indicating if the data is valid
        """
        valid = True

        # Check player data
        if player_data.empty:
            logger.warning("Player data is empty")
            valid = False
        else:
            logger.info(f"Player data: {len(player_data)} rows, {player_data.shape[1]} columns")
            # Check for required columns
            required_columns = ['player_id', 'season', 'pts']
            for col in required_columns:
                if col not in player_data.columns:
                    logger.warning(f"Player data missing required column: {col}")
                    valid = False

        # Check team stats
        if team_stats.empty:
            logger.warning("Team stats data is empty")
            valid = False
        else:
            logger.info(f"Team stats: {len(team_stats)} rows, {team_stats.shape[1]} columns")

        # Check team schedules
        if team_schedules.empty:
            logger.warning("Team schedules data is empty")
            valid = False
        else:
            logger.info(f"Team schedules: {len(team_schedules)} rows, {team_schedules.shape[1]} columns")

        # Check odds data
        if odds_data.empty:
            logger.warning("Odds data is empty")
            valid = False
        else:
            logger.info(f"Odds data: {len(odds_data)} rows, {odds_data.shape[1]} columns")

        return valid

def parse_args():
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Collect NBA data for betting model")

    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2023],
        help="NBA seasons to scrape (e.g., 2023 for 2022-23 season)"
    )

    parser.add_argument(
        "--players",
        type=str,
        nargs="+",
        help="Player IDs to scrape (if not specified, get top players)"
    )

    parser.add_argument(
        "--teams",
        type=str,
        nargs="+",
        help="Team IDs to scrape (if not specified, get top teams)"
    )

    parser.add_argument(
        "--odds-start-date",
        type=str,
        help="Start date for odds scraping (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--odds-end-date",
        type=str,
        help="End date for odds scraping (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/raw",
        help="Directory to save scraped data"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "players", "teams", "odds"],
        default="all",
        help="Type of data to collect"
    )

    return parser.parse_args()

def main():
    """
    Main function to run the data collector
    """
    args = parse_args()

    # Initialize data collector
    collector = NBADataCollector(output_dir=args.output_dir)

    # Collect data based on specified type
    if args.type == "all" or args.type == "players":
        player_data = collector.collect_player_data(
            players=args.players,
            seasons=args.seasons
        )
    else:
        player_data = None

    if args.type == "all" or args.type == "teams":
        team_stats, team_schedules = collector.collect_team_data(
            teams=args.teams,
            seasons=args.seasons
        )
    else:
        team_stats, team_schedules = None, None

    if args.type == "all" or args.type == "odds":
        odds_data = collector.collect_odds_data(
            start_date=args.odds_start_date,
            end_date=args.odds_end_date
        )
    else:
        odds_data = None

    # Validate data if collecting all types
    if args.type == "all":
        is_valid = collector.validate_data(player_data, team_stats, team_schedules, odds_data)
        if is_valid:
            logger.info("Data validation successful")
        else:
            logger.warning("Data validation failed")

if __name__ == "__main__":
    main()