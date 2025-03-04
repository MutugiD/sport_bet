import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from feature_engineering import NBAFeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("validate_feature_engineering.log"), logging.StreamHandler()]
)
logger = logging.getLogger("validate_feature_engineering")

class FeatureEngineeringValidator:
    """
    Validator for the feature engineering process
    """

    def __init__(self, processed_dir="../data/processed"):
        """
        Initialize the validator

        Args:
            processed_dir: Directory with processed data
        """
        self.processed_dir = processed_dir
        self.engineer = NBAFeatureEngineer(processed_dir=processed_dir, output_dir=processed_dir)

    def validate_data_loading(self):
        """
        Validate that cleaned data can be loaded correctly

        Returns:
            Dictionary with loaded data, or None if validation fails
        """
        logger.info("Validating data loading process")

        try:
            cleaned_data = self.engineer.load_cleaned_data()

            # Check if all required datasets are loaded
            for key in ['player_data', 'team_stats', 'team_schedules', 'odds_data']:
                if key not in cleaned_data:
                    logger.error(f"Missing dataset: {key}")
                    return None

                if cleaned_data[key].empty:
                    logger.warning(f"Empty dataset: {key}")

            # Log counts
            for key, df in cleaned_data.items():
                logger.info(f"Loaded {key} with {len(df)} rows")
                if not df.empty:
                    logger.info(f"Columns in {key}: {list(df.columns)}")

            logger.info("Data loading validation successful")
            return cleaned_data

        except Exception as e:
            logger.error(f"Data loading validation failed: {e}")
            return None

    def validate_player_features(self, cleaned_data=None):
        """
        Validate player feature creation

        Args:
            cleaned_data: Dictionary with cleaned data (if None, load data)

        Returns:
            DataFrame with player features, or None if validation fails
        """
        logger.info("Validating player feature creation")

        if cleaned_data is None or cleaned_data['player_data'].empty:
            logger.error("No player data available for feature creation")
            return None

        try:
            # Create player features
            player_features = self.engineer.create_player_features(cleaned_data['player_data'])

            if player_features.empty:
                logger.error("Failed to create player features - empty result")
                return None

            # Validate specific features
            expected_features = [
                'pts', 'mp_numeric', 'player_id', 'game_date',
                'pts_last_5_avg', 'pts_last_10_avg', 'pts_last_game',
                'season_avg_pts', 'exceeded_avg', 'days_rest', 'is_b2b'
            ]

            missing_features = [f for f in expected_features if f not in player_features.columns]
            if missing_features:
                logger.warning(f"Missing expected features: {missing_features}")

            # Check for NaNs
            nan_counts = player_features.isna().sum()
            features_with_nans = nan_counts[nan_counts > 0]
            if not features_with_nans.empty:
                logger.warning("Features with NaN values:")
                for feature, count in features_with_nans.items():
                    logger.warning(f"  {feature}: {count} NaNs")

            logger.info(f"Created player features: {len(player_features)} rows, {len(player_features.columns)} columns")
            logger.info(f"Sample features: {list(player_features.columns)[:10]}...")

            logger.info("Player feature creation validation successful")
            return player_features

        except Exception as e:
            logger.error(f"Player feature creation validation failed: {e}")
            return None

    def validate_team_features(self, player_features, cleaned_data=None):
        """
        Validate team feature creation

        Args:
            player_features: DataFrame with player features
            cleaned_data: Dictionary with cleaned data (if None, load data)

        Returns:
            DataFrame with player and team features, or None if validation fails
        """
        logger.info("Validating team feature creation")

        if player_features is None or player_features.empty:
            logger.error("No player features available for team feature creation")
            return None

        if cleaned_data is None:
            cleaned_data = self.engineer.load_cleaned_data()

        if cleaned_data['team_stats'].empty or cleaned_data['team_schedules'].empty:
            logger.warning("Team data is empty, team features may be limited")

        try:
            # Create team features
            team_features = self.engineer.create_team_features(
                player_features,
                cleaned_data['team_stats'],
                cleaned_data['team_schedules']
            )

            if team_features.empty:
                logger.error("Failed to create team features - empty result")
                return None

            # Check if team features were added
            new_features = set(team_features.columns) - set(player_features.columns)
            logger.info(f"Added team features: {list(new_features)}")

            if not new_features:
                logger.warning("No new team features were added")

            # Check for NaNs
            nan_counts = team_features.isna().sum()
            features_with_nans = nan_counts[nan_counts > 0]
            if not features_with_nans.empty:
                logger.warning("Features with NaN values:")
                for feature, count in features_with_nans.items():
                    logger.warning(f"  {feature}: {count} NaNs")

            logger.info(f"Created team features: {len(team_features)} rows, {len(team_features.columns)} columns")

            logger.info("Team feature creation validation successful")
            return team_features

        except Exception as e:
            logger.error(f"Team feature creation validation failed: {e}")
            return None

    def validate_matchup_features(self, team_features):
        """
        Validate matchup feature creation

        Args:
            team_features: DataFrame with player and team features

        Returns:
            DataFrame with all features, or None if validation fails
        """
        logger.info("Validating matchup feature creation")

        if team_features is None or team_features.empty:
            logger.error("No team features available for matchup feature creation")
            return None

        try:
            # Create matchup features
            matchup_features = self.engineer.create_matchup_features(team_features)

            if matchup_features.empty:
                logger.error("Failed to create matchup features - empty result")
                return None

            # Check if matchup features were added
            new_features = set(matchup_features.columns) - set(team_features.columns)
            logger.info(f"Added matchup features: {list(new_features)}")

            if not new_features and 'opp_id' in team_features.columns:
                logger.warning("No new matchup features were added despite having opponent data")

            # Check for NaNs
            nan_counts = matchup_features.isna().sum()
            features_with_nans = nan_counts[nan_counts > 0]
            if not features_with_nans.empty:
                logger.warning("Features with NaN values:")
                for feature, count in features_with_nans.items():
                    logger.warning(f"  {feature}: {count} NaNs")

            logger.info(f"Created matchup features: {len(matchup_features)} rows, {len(matchup_features.columns)} columns")

            logger.info("Matchup feature creation validation successful")
            return matchup_features

        except Exception as e:
            logger.error(f"Matchup feature creation validation failed: {e}")
            return None

    def validate_target_variables(self, features, cleaned_data=None):
        """
        Validate target variable creation

        Args:
            features: DataFrame with all features
            cleaned_data: Dictionary with cleaned data (if None, load data)

        Returns:
            DataFrame with features and targets, or None if validation fails
        """
        logger.info("Validating target variable creation")

        if features is None or features.empty:
            logger.error("No features available for target variable creation")
            return None

        if cleaned_data is None:
            cleaned_data = self.engineer.load_cleaned_data()

        try:
            # Create target variables
            full_dataset = self.engineer.create_target_variables(features, cleaned_data['odds_data'])

            if full_dataset.empty:
                logger.error("Failed to create target variables - empty result")
                return None

            # Check if target variables were added
            new_features = set(full_dataset.columns) - set(features.columns)
            logger.info(f"Added target variables: {list(new_features)}")

            # Check for expected targets
            expected_targets = [col for col in full_dataset.columns if col.startswith('over_')]
            logger.info(f"Target variables for different thresholds: {expected_targets}")

            if not expected_targets:
                logger.warning("No threshold-based target variables were created")

            # Check for NaNs
            nan_counts = full_dataset.isna().sum()
            features_with_nans = nan_counts[nan_counts > 0]
            if not features_with_nans.empty:
                logger.warning("Features with NaN values:")
                for feature, count in features_with_nans.items():
                    logger.warning(f"  {feature}: {count} NaNs")

            logger.info(f"Created full dataset: {len(full_dataset)} rows, {len(full_dataset.columns)} columns")

            logger.info("Target variable creation validation successful")
            return full_dataset

        except Exception as e:
            logger.error(f"Target variable creation validation failed: {e}")
            return None

    def validate_train_test_split(self, full_dataset):
        """
        Validate train/test splitting

        Args:
            full_dataset: DataFrame with features and targets

        Returns:
            Dictionary with train and test DataFrames, or None if validation fails
        """
        logger.info("Validating train/test splitting")

        if full_dataset is None or full_dataset.empty:
            logger.error("No data available for train/test splitting")
            return None

        try:
            # Create train/test sets
            train_test_sets = self.engineer.create_train_test_sets(full_dataset)

            if train_test_sets is None:
                logger.error("Failed to create train/test sets - none returned")
                return None

            train_df = train_test_sets.get('train')
            test_df = train_test_sets.get('test')

            if train_df is None or train_df.empty:
                logger.error("Training set is empty")
                return None

            if test_df is None or test_df.empty:
                logger.error("Testing set is empty")
                return None

            # Check training and testing proportions
            train_pct = len(train_df) / len(full_dataset) * 100
            test_pct = len(test_df) / len(full_dataset) * 100

            logger.info(f"Training set: {len(train_df)} rows ({train_pct:.1f}%)")
            logger.info(f"Testing set: {len(test_df)} rows ({test_pct:.1f}%)")

            # Check for data leakage
            if 'game_date' in train_df.columns and 'game_date' in test_df.columns:
                train_dates = pd.to_datetime(train_df['game_date'])
                test_dates = pd.to_datetime(test_df['game_date'])

                max_train_date = train_dates.max()
                min_test_date = test_dates.min()

                logger.info(f"Latest training date: {max_train_date}")
                logger.info(f"Earliest testing date: {min_test_date}")

                if max_train_date >= min_test_date:
                    logger.warning("Possible data leakage: training dates overlap with testing dates")

            logger.info("Train/test splitting validation successful")
            return train_test_sets

        except Exception as e:
            logger.error(f"Train/test splitting validation failed: {e}")
            return None

    def validate_feature_save(self, train_test_sets):
        """
        Validate saving of train/test sets

        Args:
            train_test_sets: Dictionary with train and test DataFrames

        Returns:
            Boolean indicating success
        """
        logger.info("Validating dataset saving")

        if train_test_sets is None:
            logger.error("No train/test sets available for saving")
            return False

        try:
            # Save datasets manually for validation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            train_path = os.path.join(self.processed_dir, f"validation_train_{timestamp}.csv")
            test_path = os.path.join(self.processed_dir, f"validation_test_{timestamp}.csv")

            train_test_sets['train'].to_csv(train_path, index=False)
            train_test_sets['test'].to_csv(test_path, index=False)

            logger.info(f"Saved validation training dataset to {train_path}")
            logger.info(f"Saved validation testing dataset to {test_path}")

            # Check if files exist
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.error("Failed to save datasets - files do not exist")
                return False

            # Verify file sizes
            train_size = os.path.getsize(train_path)
            test_size = os.path.getsize(test_path)

            logger.info(f"Training file size: {train_size / 1024:.1f} KB")
            logger.info(f"Testing file size: {test_size / 1024:.1f} KB")

            if train_size == 0 or test_size == 0:
                logger.error("One or both saved files are empty")
                return False

            logger.info("Dataset saving validation successful")
            return True

        except Exception as e:
            logger.error(f"Dataset saving validation failed: {e}")
            return False

    def validate_end_to_end(self):
        """
        Validate the entire feature engineering process end-to-end

        Returns:
            Boolean indicating success
        """
        logger.info("Starting end-to-end feature engineering validation")

        # Step 1: Load cleaned data
        cleaned_data = self.validate_data_loading()
        if cleaned_data is None:
            logger.error("End-to-end validation failed at data loading step")
            return False

        # Step 2: Create player features
        player_features = self.validate_player_features(cleaned_data)
        if player_features is None:
            logger.error("End-to-end validation failed at player feature creation step")
            return False

        # Step 3: Create team features
        team_features = self.validate_team_features(player_features, cleaned_data)
        if team_features is None:
            logger.error("End-to-end validation failed at team feature creation step")
            return False

        # Step 4: Create matchup features
        matchup_features = self.validate_matchup_features(team_features)
        if matchup_features is None:
            logger.error("End-to-end validation failed at matchup feature creation step")
            return False

        # Step 5: Create target variables
        full_dataset = self.validate_target_variables(matchup_features, cleaned_data)
        if full_dataset is None:
            logger.error("End-to-end validation failed at target variable creation step")
            return False

        # Step 6: Create train/test split
        train_test_sets = self.validate_train_test_split(full_dataset)
        if train_test_sets is None:
            logger.error("End-to-end validation failed at train/test splitting step")
            return False

        # Step 7: Save datasets
        save_success = self.validate_feature_save(train_test_sets)
        if not save_success:
            logger.error("End-to-end validation failed at dataset saving step")
            return False

        # Alternative: Use the engineer's complete function
        logger.info("Testing the engineer's create_modelling_dataset function")
        try:
            datasets = self.engineer.create_modelling_dataset(cleaned_data)
            if datasets is None:
                logger.warning("Engineer's create_modelling_dataset function returned None")
            else:
                logger.info(f"Engineer's function created datasets with {len(datasets['train'])} training and {len(datasets['test'])} testing records")
        except Exception as e:
            logger.error(f"Engineer's create_modelling_dataset function failed: {e}")

        logger.info("End-to-end feature engineering validation successful")
        return True

    def check_feature_quality(self, full_dataset):
        """
        Perform additional quality checks on features

        Args:
            full_dataset: DataFrame with features and targets

        Returns:
            Boolean indicating quality
        """
        logger.info("Performing feature quality checks")

        if full_dataset is None or full_dataset.empty:
            logger.error("No data available for quality checks")
            return False

        try:
            # Check feature distributions
            numeric_features = full_dataset.select_dtypes(include=['float64', 'int64']).columns
            feature_stats = full_dataset[numeric_features].describe().T

            # Add skewness and kurtosis
            feature_stats['skewness'] = full_dataset[numeric_features].skew()
            feature_stats['kurtosis'] = full_dataset[numeric_features].kurtosis()

            # Check for constant features
            constant_features = [col for col in numeric_features
                                if feature_stats.loc[col, 'std'] == 0]

            if constant_features:
                logger.warning(f"Constant features detected: {constant_features}")

            # Check for highly skewed features
            skewed_features = feature_stats[feature_stats['skewness'].abs() > 3].index.tolist()
            if skewed_features:
                logger.warning(f"Highly skewed features: {skewed_features[:10]}{'...' if len(skewed_features) > 10 else ''}")

            # Check for features with too many zeros
            zero_pcts = {}
            for col in numeric_features:
                zero_pct = (full_dataset[col] == 0).mean() * 100
                if zero_pct > 80:  # More than 80% zeros
                    zero_pcts[col] = zero_pct

            if zero_pcts:
                logger.warning("Features with high percentage of zeros:")
                for col, pct in sorted(zero_pcts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    logger.warning(f"  {col}: {pct:.1f}%")

            # Check correlations with target
            if 'pts' in numeric_features:
                target_correlations = full_dataset[numeric_features].corr()['pts'].sort_values(ascending=False)

                logger.info("Top 10 features by correlation with points:")
                for feature, corr in target_correlations.head(11).items():
                    if feature != 'pts':
                        logger.info(f"  {feature}: {corr:.4f}")

                # Check for features with no correlation
                no_corr_features = target_correlations[target_correlations.abs() < 0.01].index.tolist()
                if no_corr_features:
                    logger.warning(f"Features with no correlation to target: {no_corr_features[:10]}{'...' if len(no_corr_features) > 10 else ''}")

            # Check for multicollinearity
            high_corrs = []
            corr_matrix = full_dataset[numeric_features].corr().abs()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > 0.95 and corr_matrix.iloc[i, j] < 1:
                        high_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

            if high_corrs:
                logger.warning("Highly correlated feature pairs (r > 0.95):")
                for f1, f2, corr in sorted(high_corrs, key=lambda x: x[2], reverse=True)[:10]:
                    logger.warning(f"  {f1} & {f2}: {corr:.4f}")

            logger.info("Feature quality check complete")
            return True

        except Exception as e:
            logger.error(f"Feature quality check failed: {e}")
            return False

def main():
    """
    Main function to run feature engineering validation
    """
    try:
        print("Starting feature engineering validation...")
        validator = FeatureEngineeringValidator()

        success = validator.validate_end_to_end()

        if success:
            print("Feature engineering validation successful!")
            return 0
        else:
            print("Feature engineering validation failed. See log for details.")
            return 1
    except Exception as e:
        print(f"Error in validation process: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())