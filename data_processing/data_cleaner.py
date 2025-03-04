import pandas as pd
import numpy as np
import os
import logging
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("data_processing.log"), logging.StreamHandler()]
)
logger = logging.getLogger("data_cleaner")

class NBADataCleaner:
    """
    Cleaner for NBA data scraped from various sources
    """

    def __init__(self, raw_dir="../data/raw", processed_dir="../data/processed"):
        """
        Initialize the data cleaner

        Args:
            raw_dir: Directory with raw data
            processed_dir: Directory to save processed data
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    def load_player_data(self, filename=None):
        """
        Load player game logs data

        Args:
            filename: Specific file to load (if None, load most recent)

        Returns:
            DataFrame of player game logs
        """
        if filename is None:
            # Find most recent player data file
            files = [f for f in os.listdir(self.raw_dir) if f.startswith("player_data_") and f.endswith(".csv")]
            if not files:
                logger.error("No player data files found")
                return pd.DataFrame()

            # Sort by timestamp in filename
            files.sort(reverse=True)
            filename = files[0]

        file_path = os.path.join(self.raw_dir, filename)
        logger.info(f"Loading player data from {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} player game records")
            return df
        except Exception as e:
            logger.error(f"Error loading player data: {e}")
            return pd.DataFrame()

    def load_team_data(self, stats_filename=None, schedule_filename=None):
        """
        Load team stats and schedule data

        Args:
            stats_filename: Specific stats file to load (if None, load most recent)
            schedule_filename: Specific schedule file to load (if None, load most recent)

        Returns:
            Tuple of (team stats DataFrame, team schedules DataFrame)
        """
        # Load team stats
        if stats_filename is None:
            # Find most recent team stats file
            files = [f for f in os.listdir(self.raw_dir) if f.startswith("team_stats_") and f.endswith(".csv")]
            if not files:
                logger.error("No team stats files found")
                stats_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                stats_filename = files[0]

                try:
                    stats_path = os.path.join(self.raw_dir, stats_filename)
                    logger.info(f"Loading team stats from {stats_path}")
                    stats_df = pd.read_csv(stats_path)
                    logger.info(f"Loaded {len(stats_df)} team stats records")
                except Exception as e:
                    logger.error(f"Error loading team stats: {e}")
                    stats_df = pd.DataFrame()
        else:
            try:
                stats_path = os.path.join(self.raw_dir, stats_filename)
                logger.info(f"Loading team stats from {stats_path}")
                stats_df = pd.read_csv(stats_path)
                logger.info(f"Loaded {len(stats_df)} team stats records")
            except Exception as e:
                logger.error(f"Error loading team stats: {e}")
                stats_df = pd.DataFrame()

        # Load team schedules
        if schedule_filename is None:
            # Find most recent team schedules file
            files = [f for f in os.listdir(self.raw_dir) if f.startswith("team_schedules_") and f.endswith(".csv")]
            if not files:
                logger.error("No team schedules files found")
                schedule_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                schedule_filename = files[0]

                try:
                    schedule_path = os.path.join(self.raw_dir, schedule_filename)
                    logger.info(f"Loading team schedules from {schedule_path}")
                    schedule_df = pd.read_csv(schedule_path)
                    logger.info(f"Loaded {len(schedule_df)} team schedule records")
                except Exception as e:
                    logger.error(f"Error loading team schedules: {e}")
                    schedule_df = pd.DataFrame()
        else:
            try:
                schedule_path = os.path.join(self.raw_dir, schedule_filename)
                logger.info(f"Loading team schedules from {schedule_path}")
                schedule_df = pd.read_csv(schedule_path)
                logger.info(f"Loaded {len(schedule_df)} team schedule records")
            except Exception as e:
                logger.error(f"Error loading team schedules: {e}")
                schedule_df = pd.DataFrame()

        return stats_df, schedule_df

    def load_odds_data(self, filename=None):
        """
        Load player props odds data

        Args:
            filename: Specific file to load (if None, load most recent)

        Returns:
            DataFrame of player props odds
        """
        if filename is None:
            # Find most recent odds data file
            files = [f for f in os.listdir(self.raw_dir) if f.startswith("player_props_") and f.endswith(".csv")]
            if not files:
                logger.error("No odds data files found")
                return pd.DataFrame()

            # Sort by timestamp in filename
            files.sort(reverse=True)
            filename = files[0]

        file_path = os.path.join(self.raw_dir, filename)
        logger.info(f"Loading odds data from {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} odds records")
            return df
        except Exception as e:
            logger.error(f"Error loading odds data: {e}")
            return pd.DataFrame()

    def clean_player_data(self, df):
        """
        Clean player game logs data

        Args:
            df: DataFrame of player game logs

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty player data, nothing to clean")
            return df

        logger.info("Cleaning player data")

        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()

        # Remove rows without game data (DNPs, inactive)
        if 'reason' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['reason'].isna()]

        # Convert numeric columns
        numeric_columns = [
            'mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct',
            'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb', 'ast',
            'stl', 'blk', 'tov', 'pf', 'pts', 'game_score'
        ]

        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

        # Handle minutes played format (convert "MM:SS" to minutes as float)
        if 'mp' in cleaned_df.columns:
            cleaned_df['mp_numeric'] = cleaned_df['mp'].apply(self._convert_minutes)

        # Fix date format
        if 'date_game' in cleaned_df.columns:
            cleaned_df['game_date'] = pd.to_datetime(cleaned_df['date_game'], errors='coerce')

        # Extract home/away indicator
        if 'game_location' in cleaned_df.columns:
            cleaned_df['is_home'] = cleaned_df['game_location'].isna()

        # Handle missing values
        for col in numeric_columns:
            if col in cleaned_df.columns:
                # Fill missing values with 0 for most stats
                if col not in ['fg_pct', 'fg3_pct', 'ft_pct']:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    # For percentages, fill with player's average
                    cleaned_df[col] = cleaned_df.groupby('player_id')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )

        # Calculate points per minute and usage proxies
        if 'mp_numeric' in cleaned_df.columns and 'pts' in cleaned_df.columns:
            # Points per minute
            cleaned_df['pts_per_min'] = cleaned_df['pts'] / cleaned_df['mp_numeric'].clip(lower=1)

            # Usage proxy (FGA + 0.44*FTA) / minutes
            if 'fga' in cleaned_df.columns and 'fta' in cleaned_df.columns:
                cleaned_df['usage_proxy'] = (cleaned_df['fga'] + 0.44 * cleaned_df['fta']) / cleaned_df['mp_numeric'].clip(lower=1)

        # Remove rows with crucial missing data
        crucial_columns = ['player_id', 'game_date', 'pts']
        for col in crucial_columns:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=[col])

        logger.info(f"Cleaned player data: {len(cleaned_df)} rows")
        return cleaned_df

    def clean_team_data(self, stats_df, schedule_df):
        """
        Clean team stats and schedule data

        Args:
            stats_df: DataFrame of team stats
            schedule_df: DataFrame of team schedules

        Returns:
            Tuple of (cleaned stats DataFrame, cleaned schedule DataFrame)
        """
        # Clean team stats
        if stats_df.empty:
            logger.warning("Empty team stats data, nothing to clean")
            cleaned_stats = pd.DataFrame()
        else:
            logger.info("Cleaning team stats")

            # Make a copy to avoid modifying the original
            cleaned_stats = stats_df.copy()

            # Convert numeric columns
            for col in cleaned_stats.columns:
                if col not in ['team_id', 'season']:
                    cleaned_stats[col] = pd.to_numeric(cleaned_stats[col], errors='coerce')

            logger.info(f"Cleaned team stats: {len(cleaned_stats)} rows")

        # Clean team schedules
        if schedule_df.empty:
            logger.warning("Empty team schedule data, nothing to clean")
            cleaned_schedule = pd.DataFrame()
        else:
            logger.info("Cleaning team schedules")

            # Make a copy to avoid modifying the original
            cleaned_schedule = schedule_df.copy()

            # Convert date column
            if 'date_game' in cleaned_schedule.columns:
                cleaned_schedule['game_date'] = pd.to_datetime(cleaned_schedule['date_game'], errors='coerce')

            # Extract scores
            if 'pts' in cleaned_schedule.columns and 'pts_opp' in cleaned_schedule.columns:
                cleaned_schedule['pts'] = pd.to_numeric(cleaned_schedule['pts'], errors='coerce')
                cleaned_schedule['pts_opp'] = pd.to_numeric(cleaned_schedule['pts_opp'], errors='coerce')

                # Add win/loss indicator
                cleaned_schedule['won'] = cleaned_schedule['pts'] > cleaned_schedule['pts_opp']

            # Extract home/away indicator
            if 'game_location' in cleaned_schedule.columns:
                cleaned_schedule['is_home'] = cleaned_schedule['game_location'].isna()

            # Handle missing values
            cleaned_schedule = cleaned_schedule.dropna(subset=['game_date'])

            logger.info(f"Cleaned team schedule: {len(cleaned_schedule)} rows")

        return cleaned_stats, cleaned_schedule

    def clean_odds_data(self, df):
        """
        Clean player props odds data

        Args:
            df: DataFrame of player props odds

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty odds data, nothing to clean")
            return df

        logger.info("Cleaning odds data")

        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()

        # Convert date column
        if 'date' in cleaned_df.columns:
            cleaned_df['game_date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')

        # Convert numeric columns
        numeric_columns = [
            'line', 'over_odds_american', 'under_odds_american',
            'over_odds_decimal', 'under_odds_decimal',
            'over_implied_probability', 'under_implied_probability'
        ]

        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

        # Standardize player names
        if 'player' in cleaned_df.columns:
            cleaned_df['player'] = cleaned_df['player'].apply(self._standardize_player_name)

        # Remove rows with crucial missing data
        crucial_columns = ['player', 'game_date', 'line']
        for col in crucial_columns:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=[col])

        logger.info(f"Cleaned odds data: {len(cleaned_df)} rows")
        return cleaned_df

    def _convert_minutes(self, mp_str):
        """
        Convert minutes played string (MM:SS) to float minutes

        Args:
            mp_str: Minutes played string

        Returns:
            Minutes as float
        """
        if pd.isna(mp_str):
            return 0.0

        try:
            if ':' in str(mp_str):
                minutes, seconds = str(mp_str).split(':')
                return int(minutes) + int(seconds) / 60
            else:
                return float(mp_str)
        except:
            return 0.0

    def _standardize_player_name(self, name):
        """
        Standardize player name format

        Args:
            name: Player name

        Returns:
            Standardized name
        """
        if pd.isna(name):
            return name

        # Remove Jr., III, etc.
        name = re.sub(r'\s+Jr\.|\s+Sr\.|\s+I+\.?|\s+IV\.?', '', name)

        # Remove nicknames in quotes
        name = re.sub(r'\s*"[^"]*"\s*', ' ', name)

        # Trim extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def save_cleaned_data(self, player_df, team_stats_df, team_schedule_df, odds_df):
        """
        Save cleaned data to processed directory

        Args:
            player_df: Cleaned player game logs
            team_stats_df: Cleaned team stats
            team_schedule_df: Cleaned team schedules
            odds_df: Cleaned odds data

        Returns:
            Dictionary with paths to saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Save player data
        if not player_df.empty:
            filename = f"cleaned_player_data_{timestamp}.csv"
            output_path = os.path.join(self.processed_dir, filename)
            player_df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned player data to {output_path}")
            saved_files['player_data'] = output_path

        # Save team stats
        if not team_stats_df.empty:
            filename = f"cleaned_team_stats_{timestamp}.csv"
            output_path = os.path.join(self.processed_dir, filename)
            team_stats_df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned team stats to {output_path}")
            saved_files['team_stats'] = output_path

        # Save team schedules
        if not team_schedule_df.empty:
            filename = f"cleaned_team_schedules_{timestamp}.csv"
            output_path = os.path.join(self.processed_dir, filename)
            team_schedule_df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned team schedules to {output_path}")
            saved_files['team_schedules'] = output_path

        # Save odds data
        if not odds_df.empty:
            filename = f"cleaned_odds_data_{timestamp}.csv"
            output_path = os.path.join(self.processed_dir, filename)
            odds_df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned odds data to {output_path}")
            saved_files['odds_data'] = output_path

        return saved_files

    def process_all_data(self):
        """
        Load, clean, and save all data types

        Returns:
            Dictionary with cleaned DataFrames
        """
        # Load raw data
        player_df = self.load_player_data()
        team_stats_df, team_schedule_df = self.load_team_data()
        odds_df = self.load_odds_data()

        # Clean data
        cleaned_player_df = self.clean_player_data(player_df)
        cleaned_team_stats_df, cleaned_team_schedule_df = self.clean_team_data(team_stats_df, team_schedule_df)
        cleaned_odds_df = self.clean_odds_data(odds_df)

        # Save cleaned data
        self.save_cleaned_data(
            cleaned_player_df,
            cleaned_team_stats_df,
            cleaned_team_schedule_df,
            cleaned_odds_df
        )

        return {
            'player_data': cleaned_player_df,
            'team_stats': cleaned_team_stats_df,
            'team_schedules': cleaned_team_schedule_df,
            'odds_data': cleaned_odds_df
        }

def main():
    """
    Main function to run the data cleaner
    """
    cleaner = NBADataCleaner()
    cleaned_data = cleaner.process_all_data()

    # Print summary of cleaned data
    for data_type, df in cleaned_data.items():
        if not df.empty:
            print(f"{data_type}: {len(df)} rows, {df.shape[1]} columns")
        else:
            print(f"{data_type}: No data found")

if __name__ == "__main__":
    main()