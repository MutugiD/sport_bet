#!/usr/bin/env python
"""
Validate the end-to-end data pipeline for NBA player points prediction model
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Import scraping modules
from scraping.player_scraper import BasketballReferencePlayerScraper
from scraping.team_scraper import BasketballReferenceTeamScraper
from scraping.odds_scraper import OddsScraper
from scraping.data_collector import NBADataCollector

# Import data processing modules
from data_processing.data_cleaner import NBADataCleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline_validation.log"), logging.StreamHandler()]
)
logger = logging.getLogger("pipeline_validation")

class PipelineValidator:
    """
    Validates the end-to-end data pipeline
    """

    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        """
        Initialize the validator

        Args:
            raw_dir: Directory for raw data
            processed_dir: Directory for processed data
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Initialize components
        self.data_collector = NBADataCollector(output_dir=raw_dir)
        self.data_cleaner = NBADataCleaner(raw_dir=raw_dir, processed_dir=processed_dir)

    def validate_data_scraping(self):
        """
        Validate the data scraping functionality

        Returns:
            Dict with DataFrames of scraped data
        """
        logger.info("Validating data scraping...")

        # Test player scraping with limited data
        test_players = [
            'jokicni01',  # Nikola Jokic
            'embiijo01',  # Joel Embiid
        ]

        test_teams = [
            'DEN',  # Denver Nuggets
            'PHI',  # Philadelphia 76ers
        ]

        # Scrape player data
        logger.info("Testing player data scraping...")
        player_data = self.data_collector.collect_player_data(
            players=test_players,
            seasons=[2023],
            save=True
        )

        # Scrape team data
        logger.info("Testing team data scraping...")
        team_stats, team_schedules = self.data_collector.collect_team_data(
            teams=test_teams,
            seasons=[2023],
            include_schedule=True,
            save=True
        )

        # Scrape odds data (mock)
        logger.info("Testing odds data scraping...")
        odds_data = self.data_collector.collect_odds_data(save=True)

        logger.info("Data scraping validation results:")
        logger.info(f"Player data: {len(player_data)} rows")
        logger.info(f"Team stats: {len(team_stats)} rows")
        logger.info(f"Team schedules: {len(team_schedules)} rows")
        logger.info(f"Odds data: {len(odds_data)} rows")

        return {
            'player_data': player_data,
            'team_stats': team_stats,
            'team_schedules': team_schedules,
            'odds_data': odds_data
        }

    def validate_data_cleaning(self, scraped_data=None):
        """
        Validate the data cleaning functionality

        Args:
            scraped_data: Dictionary with DataFrames of scraped data

        Returns:
            Dict with DataFrames of cleaned data
        """
        logger.info("Validating data cleaning...")

        if scraped_data is None:
            # Load data from files
            player_df = self.data_cleaner.load_player_data()
            team_stats_df, team_schedule_df = self.data_cleaner.load_team_data()
            odds_df = self.data_cleaner.load_odds_data()
        else:
            # Use provided data
            player_df = scraped_data['player_data']
            team_stats_df = scraped_data['team_stats']
            team_schedule_df = scraped_data['team_schedules']
            odds_df = scraped_data['odds_data']

        # Clean data
        cleaned_player_df = self.data_cleaner.clean_player_data(player_df)
        cleaned_team_stats_df, cleaned_team_schedule_df = self.data_cleaner.clean_team_data(team_stats_df, team_schedule_df)
        cleaned_odds_df = self.data_cleaner.clean_odds_data(odds_df)

        # Save cleaned data
        self.data_cleaner.save_cleaned_data(
            cleaned_player_df,
            cleaned_team_stats_df,
            cleaned_team_schedule_df,
            cleaned_odds_df
        )

        logger.info("Data cleaning validation results:")
        logger.info(f"Cleaned player data: {len(cleaned_player_df)} rows")
        logger.info(f"Cleaned team stats: {len(cleaned_team_stats_df)} rows")
        logger.info(f"Cleaned team schedules: {len(cleaned_team_schedule_df)} rows")
        logger.info(f"Cleaned odds data: {len(cleaned_odds_df)} rows")

        return {
            'player_data': cleaned_player_df,
            'team_stats': cleaned_team_stats_df,
            'team_schedules': cleaned_team_schedule_df,
            'odds_data': cleaned_odds_df
        }

    def validate_end_to_end(self):
        """
        Validate the entire data pipeline from scraping to cleaning

        Returns:
            Success status
        """
        logger.info("Starting end-to-end pipeline validation...")

        try:
            # Step 1: Data scraping
            scraped_data = self.validate_data_scraping()

            # Check if any data was scraped
            if all(len(df) == 0 for df in scraped_data.values()):
                logger.error("No data was scraped successfully")
                return False

            # Step 2: Data cleaning
            cleaned_data = self.validate_data_cleaning(scraped_data)

            # Check if any data was cleaned
            if all(len(df) == 0 for df in cleaned_data.values()):
                logger.error("No data was cleaned successfully")
                return False

            # Step 3: Data quality checks
            quality_issues = self._check_data_quality(cleaned_data)

            if quality_issues:
                logger.warning(f"Data quality issues found: {quality_issues}")
                return False

            logger.info("End-to-end pipeline validation successful!")
            return True

        except Exception as e:
            logger.error(f"Pipeline validation failed: {str(e)}")
            return False

    def _check_data_quality(self, cleaned_data):
        """
        Perform basic data quality checks

        Args:
            cleaned_data: Dictionary with DataFrames of cleaned data

        Returns:
            List of quality issues
        """
        issues = []

        # Check player data
        player_data = cleaned_data['player_data']
        if not player_data.empty:
            # Check for required columns
            required_columns = ['player_id', 'season', 'pts', 'game_date']
            missing_columns = [col for col in required_columns if col not in player_data.columns]
            if missing_columns:
                issues.append(f"Player data missing columns: {missing_columns}")

            # Check for duplicates
            if player_data.duplicated(['player_id', 'game_date']).any():
                issues.append("Player data contains duplicate player/game combinations")

            # Check for reasonable point values
            if player_data['pts'].max() > 100:
                issues.append(f"Player data contains unreasonable point values (max: {player_data['pts'].max()})")

        # Check team stats
        team_stats = cleaned_data['team_stats']
        if not team_stats.empty:
            # Check for required columns
            if 'team_id' not in team_stats.columns:
                issues.append("Team stats missing team_id column")

        # Check odds data
        odds_data = cleaned_data['odds_data']
        if not odds_data.empty:
            # Check for required columns
            required_columns = ['player', 'line', 'game_date']
            missing_columns = [col for col in required_columns if col not in odds_data.columns]
            if missing_columns:
                issues.append(f"Odds data missing columns: {missing_columns}")

            # Check for reasonable line values
            if odds_data['line'].max() > 100:
                issues.append(f"Odds data contains unreasonable line values (max: {odds_data['line'].max()})")

        return issues

def main():
    """
    Main function to run the pipeline validator
    """
    logger.info("=== NBA Player Points Prediction Pipeline Validation ===")

    validator = PipelineValidator()

    # Just validate the end-to-end pipeline
    success = validator.validate_end_to_end()

    if success:
        logger.info("Pipeline validation completed successfully")
        print("\n✅ Pipeline validation SUCCEEDED")
    else:
        logger.error("Pipeline validation failed")
        print("\n❌ Pipeline validation FAILED")

if __name__ == "__main__":
    main()