import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import argparse
import glob

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from scraping.data_collector import NBADataCollector
from data_processing.data_cleaner import NBADataCleaner
from data_processing.feature_engineering import NBAFeatureEngineer
from modelling.baseline_models import NBABaselineModeler
from modelling.advanced_models import NBAAdvancedModeler
from evaluation.odds_calculator import BettingOddsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline_evaluation.log"), logging.StreamHandler()]
)
logger = logging.getLogger("pipeline_evaluation")

class PipelineEvaluator:
    """
    Evaluate the entire NBA prediction and betting pipeline
    """

    def __init__(self, data_dir="../data", models_dir="../models", results_dir="../data/results"):
        """
        Initialize the pipeline evaluator

        Args:
            data_dir: Base directory for data
            models_dir: Directory for models
            results_dir: Directory for results
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.models_dir = models_dir
        self.results_dir = results_dir

        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def validate_data_collection(self, players=None, seasons=None, force_new=False):
        """
        Validate data collection from web sources

        Args:
            players: List of player names to collect data for (default: small test set)
            seasons: List of seasons to collect data for (default: current season)
            force_new: Force new data collection even if files exist

        Returns:
            Boolean indicating success
        """
        logger.info("Validating data collection")
        print(f"Raw data directory: {self.raw_dir}")

        # Default to test set of players if none provided
        if players is None:
            players = ["Nikola Jokic", "LeBron James", "Stephen Curry", "Joel Embiid", "Kevin Durant"]

        # Default to current season if none provided
        if seasons is None:
            seasons = ["2023"]

        try:
            # Create data collector
            collector = NBADataCollector(output_dir=self.raw_dir)
            print(f"Created NBADataCollector with output_dir={self.raw_dir}")

            # Check if files already exist
            player_files = glob.glob(os.path.join(self.raw_dir, "player_data_*.csv"))
            team_stat_files = glob.glob(os.path.join(self.raw_dir, "team_stats_*.csv"))
            props_files = glob.glob(os.path.join(self.raw_dir, "player_props_*.csv"))

            print(f"Found {len(player_files)} player files, {len(team_stat_files)} team files, {len(props_files)} props files")
            data_exists = player_files and team_stat_files and props_files

            if data_exists and not force_new:
                logger.info("Raw data files already exist. Use force_new=True to recollect data.")
                print("Raw data files already exist. Skipping data collection.")
                return True

            # Collect data
            logger.info(f"Collecting data for {len(players)} players across {len(seasons)} seasons")
            print(f"Collecting data for {len(players)} players across {len(seasons)} seasons")

            # Call collect_all_data with the seasons parameter
            collection_result = collector.collect_all_data(seasons=seasons)

            if any(df is None or (df is not None and df.empty) for df in collection_result):
                logger.error("Some data collections returned empty results")
                print("Some data collections returned empty results")
                return False

            # Verify files exist after collection
            player_files = glob.glob(os.path.join(self.raw_dir, "player_data_*.csv"))
            team_stat_files = glob.glob(os.path.join(self.raw_dir, "team_stats_*.csv"))
            props_files = glob.glob(os.path.join(self.raw_dir, "player_props_*.csv"))

            if not player_files:
                logger.error("No player data files found")
                return False

            if not team_stat_files:
                logger.error("No team stats files found")
                return False

            if not props_files:
                logger.error("No odds data files found")
                return False

            # Check if files have data
            player_df = pd.read_csv(player_files[-1])  # Get the latest file
            team_df = pd.read_csv(team_stat_files[-1])
            odds_df = pd.read_csv(props_files[-1])

            if player_df.empty or team_df.empty or odds_df.empty:
                logger.error("One or more data files are empty")
                return False

            logger.info(f"Collected {len(player_df)} player records, {len(team_df)} team records, and {len(odds_df)} odds records")
            logger.info("Data collection successful")
            return True

        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            logger.error("Pipeline validation failed at data collection stage")
            return False

    def validate_data_cleaning(self, force_new=False):
        """
        Validate data cleaning process

        Args:
            force_new: Force new data cleaning even if files exist

        Returns:
            Boolean indicating success and path to cleaned data
        """
        logger.info("Validating data cleaning")

        try:
            # Check if raw data exists
            player_file = os.path.join(self.raw_dir, "player_data.csv")
            team_file = os.path.join(self.raw_dir, "team_stats.csv")
            odds_file = os.path.join(self.raw_dir, "odds_data.csv")

            for file_path in [player_file, team_file, odds_file]:
                if not os.path.exists(file_path):
                    logger.error(f"Raw data file not found: {file_path}")
                    return False, None

            # Find most recent cleaned data file if it exists
            cleaned_files = [f for f in os.listdir(self.processed_dir)
                          if f.startswith("cleaned_player_data") and f.endswith(".csv")]

            if cleaned_files and not force_new:
                # Sort by timestamp in filename
                cleaned_files.sort(reverse=True)
                cleaned_data_path = os.path.join(self.processed_dir, cleaned_files[0])
                logger.info(f"Using existing cleaned data: {cleaned_files[0]}")

                return True, cleaned_data_path

            # Create data cleaner
            cleaner = NBADataCleaner(
                raw_data_dir=self.raw_dir,
                processed_data_dir=self.processed_dir
            )

            # Clean data
            logger.info("Cleaning raw data")
            status, cleaned_data_path = cleaner.clean_all_data()

            if not status or cleaned_data_path is None:
                logger.error("Data cleaning failed")
                return False, None

            # Verify cleaned data
            if not os.path.exists(cleaned_data_path):
                logger.error(f"Cleaned data file not created: {cleaned_data_path}")
                return False, None

            df = pd.read_csv(cleaned_data_path)
            if df.empty:
                logger.error(f"Cleaned data file is empty: {cleaned_data_path}")
                return False, None

            # Check for key columns
            required_columns = ['player_name', 'game_date', 'team', 'opponent', 'pts']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Cleaned data missing required columns: {missing_columns}")
                return False, None

            logger.info(f"Cleaned data has {len(df)} records and {len(df.columns)} columns")
            logger.info("Data cleaning validation successful")

            return True, cleaned_data_path

        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            return False, None

    def validate_feature_engineering(self, cleaned_data_path=None, force_new=False):
        """
        Validate feature engineering process

        Args:
            cleaned_data_path: Path to cleaned data (if None, will find most recent)
            force_new: Force new feature engineering even if files exist

        Returns:
            Boolean indicating success and paths to train/test datasets
        """
        logger.info("Validating feature engineering")

        try:
            # Find cleaned data if not provided
            if cleaned_data_path is None:
                cleaned_files = [f for f in os.listdir(self.processed_dir)
                              if f.startswith("cleaned_player_data") and f.endswith(".csv")]

                if not cleaned_files:
                    logger.error("No cleaned data files found")
                    return False, None, None

                # Sort by timestamp in filename
                cleaned_files.sort(reverse=True)
                cleaned_data_path = os.path.join(self.processed_dir, cleaned_files[0])
                logger.info(f"Using cleaned data: {cleaned_files[0]}")

            # Find most recent train/test datasets if they exist
            train_files = [f for f in os.listdir(self.processed_dir)
                         if f.startswith("train_dataset") and f.endswith(".csv")]
            test_files = [f for f in os.listdir(self.processed_dir)
                        if f.startswith("test_dataset") and f.endswith(".csv")]

            if train_files and test_files and not force_new:
                # Sort by timestamp in filename
                train_files.sort(reverse=True)
                test_files.sort(reverse=True)

                train_path = os.path.join(self.processed_dir, train_files[0])
                test_path = os.path.join(self.processed_dir, test_files[0])

                logger.info(f"Using existing train dataset: {train_files[0]}")
                logger.info(f"Using existing test dataset: {test_files[0]}")

                return True, train_path, test_path

            # Create feature engineer
            engineer = NBAFeatureEngineer(
                processed_data_dir=self.processed_dir
            )

            # Engineer features
            logger.info("Engineering features from cleaned data")
            status, train_path, test_path = engineer.create_modelling_dataset()

            if not status or train_path is None or test_path is None:
                logger.error("Feature engineering failed")
                return False, None, None

            # Verify train/test datasets
            for path, name in [(train_path, "training"), (test_path, "testing")]:
                if not os.path.exists(path):
                    logger.error(f"{name.capitalize()} dataset not created: {path}")
                    return False, None, None

                df = pd.read_csv(path)
                if df.empty:
                    logger.error(f"{name.capitalize()} dataset is empty: {path}")
                    return False, None, None

                # Check for key columns
                required_columns = ['player_name', 'game_date', 'pts', 'team', 'opponent']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    logger.error(f"{name.capitalize()} dataset missing required columns: {missing_columns}")
                    return False, None, None

                logger.info(f"{name.capitalize()} dataset has {len(df)} records and {len(df.columns)} columns")

            logger.info("Feature engineering validation successful")
            return True, train_path, test_path

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False, None, None

    def validate_model_training(self, train_path=None, test_path=None, force_new=False):
        """
        Validate model training process

        Args:
            train_path: Path to training dataset (if None, will find most recent)
            test_path: Path to testing dataset (if None, will find most recent)
            force_new: Force new model training even if files exist

        Returns:
            Boolean indicating success and paths to trained models
        """
        logger.info("Validating model training")

        try:
            # Find train/test datasets if not provided
            if train_path is None or test_path is None:
                train_files = [f for f in os.listdir(self.processed_dir)
                             if f.startswith("train_dataset") and f.endswith(".csv")]
                test_files = [f for f in os.listdir(self.processed_dir)
                            if f.startswith("test_dataset") and f.endswith(".csv")]

                if not train_files or not test_files:
                    logger.error("Train or test dataset files not found")
                    return False, None

                # Sort by timestamp in filename
                train_files.sort(reverse=True)
                test_files.sort(reverse=True)

                train_path = os.path.join(self.processed_dir, train_files[0])
                test_path = os.path.join(self.processed_dir, test_files[0])

                logger.info(f"Using train dataset: {train_files[0]}")
                logger.info(f"Using test dataset: {test_files[0]}")

            # Find most recent model files if they exist
            baseline_models = [f for f in os.listdir(self.models_dir)
                             if f.startswith("baseline_model_") and f.endswith(".pkl")]
            advanced_models = [f for f in os.listdir(self.models_dir)
                             if f.startswith("advanced_model_") and f.endswith(".pkl")]

            if baseline_models and advanced_models and not force_new:
                # Sort by timestamp in filename
                baseline_models.sort(reverse=True)
                advanced_models.sort(reverse=True)

                baseline_path = os.path.join(self.models_dir, baseline_models[0])
                advanced_path = os.path.join(self.models_dir, advanced_models[0])

                logger.info(f"Using existing baseline model: {baseline_models[0]}")
                logger.info(f"Using existing advanced model: {advanced_models[0]}")

                return True, [baseline_path, advanced_path]

            # Train baseline models
            logger.info("Training baseline models")
            baseline_modeler = NBABaselineModeler(models_dir=self.models_dir)
            baseline_status, baseline_metrics, baseline_path = baseline_modeler.evaluate_models(
                train_path=train_path, test_path=test_path
            )

            if not baseline_status or baseline_path is None:
                logger.error("Baseline model training failed")
                return False, None

            # Train advanced models
            logger.info("Training advanced models")
            advanced_modeler = NBAAdvancedModeler(models_dir=self.models_dir)
            advanced_status, advanced_metrics, advanced_path = advanced_modeler.evaluate_models(
                train_path=train_path, test_path=test_path
            )

            if not advanced_status or advanced_path is None:
                logger.error("Advanced model training failed")
                return False, None

            # Verify model files exist
            for path, name in [(baseline_path, "baseline"), (advanced_path, "advanced")]:
                if not os.path.exists(path):
                    logger.error(f"{name.capitalize()} model file not created: {path}")
                    return False, None

            logger.info("Model training validation successful")
            logger.info(f"Baseline model metrics: {baseline_metrics}")
            logger.info(f"Advanced model metrics: {advanced_metrics}")

            return True, [baseline_path, advanced_path]

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False, None

    def validate_odds_calculation(self, model_paths=None, test_path=None):
        """
        Validate odds calculation and betting evaluation

        Args:
            model_paths: List of paths to trained models (if None, will find most recent)
            test_path: Path to testing dataset (if None, will find most recent)

        Returns:
            Boolean indicating success
        """
        logger.info("Validating odds calculation and betting evaluation")

        try:
            # Find model files if not provided
            if model_paths is None:
                model_files = [f for f in os.listdir(self.models_dir)
                             if f.endswith(".pkl")]

                if not model_files:
                    logger.error("No model files found")
                    return False

                # Sort by timestamp in filename
                model_files.sort(reverse=True)
                model_paths = [os.path.join(self.models_dir, model_files[0])]

                logger.info(f"Using model: {model_files[0]}")

            # Find test dataset if not provided
            if test_path is None:
                test_files = [f for f in os.listdir(self.processed_dir)
                            if f.startswith("test_dataset") and f.endswith(".csv")]

                if not test_files:
                    logger.error("No test dataset files found")
                    return False

                # Sort by timestamp in filename
                test_files.sort(reverse=True)
                test_path = os.path.join(self.processed_dir, test_files[0])

                logger.info(f"Using test dataset: {test_files[0]}")

            # Load test data
            test_data = pd.read_csv(test_path)
            if test_data.empty:
                logger.error(f"Test dataset is empty: {test_path}")
                return False

            # Create synthetic market odds for testing
            market_odds = self._create_synthetic_odds(test_data)
            odds_path = os.path.join(self.processed_dir, f"synthetic_odds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            market_odds.to_csv(odds_path, index=False)
            logger.info(f"Created synthetic market odds with {len(market_odds)} records")

            # Load a model
            model_path = model_paths[0]
            calculator = BettingOddsCalculator(models_dir=self.models_dir, results_dir=self.results_dir)

            model_data = calculator.load_model(model_path)
            if model_data is None:
                logger.error(f"Failed to load model: {model_path}")
                return False

            # Sample a player for testing
            player_name = test_data['player_name'].iloc[0]
            player_data = test_data[test_data['player_name'] == player_name].iloc[0]

            # Generate points distribution
            distribution = calculator.predict_points_distribution(model_data, player_data)
            if distribution is None:
                logger.error("Failed to generate points distribution")
                return False

            logger.info(f"Generated points distribution for {player_name} with mean={distribution['mean']:.1f}")

            # Calculate fair odds
            thresholds = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5]
            fair_odds = calculator.calculate_over_under_probs(distribution, thresholds)

            if fair_odds is None:
                logger.error("Failed to calculate fair odds")
                return False

            logger.info(f"Calculated fair odds for {len(thresholds)} thresholds")

            # Compare with market odds
            player_lines = market_odds[market_odds['player'] == player_name]

            if player_lines.empty:
                logger.warning(f"No market lines found for {player_name}")
            else:
                comparison = calculator.compare_with_market(fair_odds, player_lines, player_name)

                if comparison is None:
                    logger.error("Failed to compare with market odds")
                    return False

                logger.info(f"Compared fair odds with {len(player_lines)} market lines")

                # Find value bets
                value_bets = calculator.find_value_bets(comparison)

                if value_bets is None:
                    logger.error("Failed to find value bets")
                    return False

                logger.info(f"Found {len(value_bets)} value betting opportunities")

            # Run backtest
            logger.info("Running backtest")
            backtest_results = calculator.backtest_strategy(
                test_data.head(100),  # Use a small subset for testing
                model_data,
                market_odds
            )

            if backtest_results is None:
                logger.error("Failed to run backtest")
                return False

            logger.info(f"Backtest results: {backtest_results['bets_placed']} bets placed, " +
                      f"{backtest_results['win_rate']:.1%} win rate, " +
                      f"{backtest_results['roi']:.1f}% ROI")

            logger.info("Odds calculation and betting evaluation validation successful")
            return True

        except Exception as e:
            logger.error(f"Error in odds calculation validation: {e}")
            return False

    def _create_synthetic_odds(self, test_data):
        """
        Create synthetic market odds for testing

        Args:
            test_data: Test dataset

        Returns:
            DataFrame with synthetic market odds
        """
        odds_data = []

        # Get unique players and game dates
        players = test_data['player_name'].unique()[:20]  # Limit to 20 players
        game_dates = test_data['game_date'].unique()[:10]  # Limit to 10 dates

        for player in players:
            player_data = test_data[test_data['player_name'] == player]

            for game_date in game_dates:
                game_data = player_data[player_data['game_date'] == game_date]

                if not game_data.empty:
                    # Calculate line based on recent performance
                    recent_pts = game_data['pts'].values[0]
                    line = round(recent_pts / 2, 1)  # Simplified line

                    # Create synthetic odds with vig
                    odds_data.append({
                        'player': player,
                        'game_date': game_date,
                        'line': line,
                        'over_odds': -110,  # Standard odds with vig
                        'under_odds': -110
                    })

        return pd.DataFrame(odds_data)

    def validate_full_pipeline(self, players=None, seasons=None, force_new=False):
        """
        Validate the entire pipeline from data collection to odds calculation

        Args:
            players: List of player names to collect data for
            seasons: List of seasons to collect data for
            force_new: Force new data processing

        Returns:
            Boolean indicating success
        """
        print("Starting full pipeline validation")
        logger.info("Starting full pipeline validation")

        try:
            # Step 1: Data Collection
            print("Step 1: Validating data collection")
            logger.info("Validating data collection")
            collection_success = self.validate_data_collection(players, seasons, force_new)
            if not collection_success:
                print("Data collection validation failed")
                logger.error("Pipeline validation failed at data collection stage")
                return False
            print("Data collection validation successful")

            # Step 2: Data Cleaning
            print("Step 2: Validating data cleaning")
            logger.info("Validating data cleaning")
            cleaning_success, cleaned_data_path = self.validate_data_cleaning(force_new)
            if not cleaning_success:
                print("Data cleaning validation failed")
                logger.error("Pipeline validation failed at data cleaning stage")
                return False
            print("Data cleaning validation successful")

            # Step 3: Feature Engineering
            print("Step 3: Validating feature engineering")
            logger.info("Validating feature engineering")
            engineering_success, train_path, test_path = self.validate_feature_engineering(
                cleaned_data_path, force_new
            )
            if not engineering_success:
                print("Feature engineering validation failed")
                logger.error("Pipeline validation failed at feature engineering stage")
                return False
            print("Feature engineering validation successful")

            # Step 4: Model Training
            print("Step 4: Validating model training")
            logger.info("Validating model training")
            training_success, model_paths = self.validate_model_training(
                train_path, test_path, force_new
            )
            if not training_success:
                print("Model training validation failed")
                logger.error("Pipeline validation failed at model training stage")
                return False
            print("Model training validation successful")

            # Step 5: Odds Calculation
            print("Step 5: Validating odds calculation")
            logger.info("Validating odds calculation")
            odds_success = self.validate_odds_calculation(model_paths, test_path)
            if not odds_success:
                print("Odds calculation validation failed")
                logger.error("Pipeline validation failed at odds calculation stage")
                return False
            print("Odds calculation validation successful")

            logger.info("Full pipeline validation successful")
            return True

        except Exception as e:
            print(f"Error in pipeline validation: {str(e)}")
            logger.error(f"Error in pipeline validation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """
    Run the pipeline evaluation
    """
    parser = argparse.ArgumentParser(description="Evaluate the NBA prediction pipeline")
    parser.add_argument("--players", nargs="+", help="List of player names to include")
    parser.add_argument("--seasons", nargs="+", help="List of seasons to include")
    parser.add_argument("--force-new", action="store_true", help="Force new data collection")
    args = parser.parse_args()

    try:
        print("Starting pipeline evaluation...")
        evaluator = PipelineEvaluator()
        print("Created PipelineEvaluator instance")

        success = evaluator.validate_full_pipeline(
            players=args.players,
            seasons=args.seasons,
            force_new=args.force_new
        )
        print(f"Pipeline validation {'succeeded' if success else 'failed'}")

        if success:
            print("Pipeline validation completed successfully.")
        else:
            print("Pipeline validation failed. See log for details.")
    except Exception as e:
        print(f"Error in pipeline evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())