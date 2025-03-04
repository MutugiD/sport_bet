import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("feature_engineering.log"), logging.StreamHandler()]
)
logger = logging.getLogger("feature_engineering")

class NBAFeatureEngineer:
    """
    Feature engineering for NBA player points prediction
    """

    def __init__(self, processed_dir="../data/processed", output_dir="../data/processed"):
        """
        Initialize the feature engineer

        Args:
            processed_dir: Directory with cleaned data
            output_dir: Directory to save feature-engineered data
        """
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_cleaned_data(self, player_filename=None, team_stats_filename=None,
                          team_schedule_filename=None, odds_filename=None):
        """
        Load cleaned data files

        Args:
            player_filename: Player data filename (if None, load most recent)
            team_stats_filename: Team stats filename (if None, load most recent)
            team_schedule_filename: Team schedule filename (if None, load most recent)
            odds_filename: Odds data filename (if None, load most recent)

        Returns:
            Dictionary with DataFrames of cleaned data
        """
        # Load player data
        if player_filename is None:
            # Find most recent cleaned player data file
            files = [f for f in os.listdir(self.processed_dir)
                    if f.startswith("cleaned_player_data_") and f.endswith(".csv")]
            if not files:
                logger.error("No cleaned player data files found")
                player_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                player_filename = files[0]

                try:
                    player_path = os.path.join(self.processed_dir, player_filename)
                    logger.info(f"Loading player data from {player_path}")
                    player_df = pd.read_csv(player_path)
                    logger.info(f"Loaded {len(player_df)} player records")
                except Exception as e:
                    logger.error(f"Error loading player data: {e}")
                    player_df = pd.DataFrame()
        else:
            try:
                player_path = os.path.join(self.processed_dir, player_filename)
                logger.info(f"Loading player data from {player_path}")
                player_df = pd.read_csv(player_path)
                logger.info(f"Loaded {len(player_df)} player records")
            except Exception as e:
                logger.error(f"Error loading player data: {e}")
                player_df = pd.DataFrame()

        # Load team stats data
        if team_stats_filename is None:
            # Find most recent cleaned team stats file
            files = [f for f in os.listdir(self.processed_dir)
                    if f.startswith("cleaned_team_stats_") and f.endswith(".csv")]
            if not files:
                logger.error("No cleaned team stats files found")
                team_stats_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                team_stats_filename = files[0]

                try:
                    team_stats_path = os.path.join(self.processed_dir, team_stats_filename)
                    logger.info(f"Loading team stats from {team_stats_path}")
                    team_stats_df = pd.read_csv(team_stats_path)
                    logger.info(f"Loaded {len(team_stats_df)} team stats records")
                except Exception as e:
                    logger.error(f"Error loading team stats: {e}")
                    team_stats_df = pd.DataFrame()
        else:
            try:
                team_stats_path = os.path.join(self.processed_dir, team_stats_filename)
                logger.info(f"Loading team stats from {team_stats_path}")
                team_stats_df = pd.read_csv(team_stats_path)
                logger.info(f"Loaded {len(team_stats_df)} team stats records")
            except Exception as e:
                logger.error(f"Error loading team stats: {e}")
                team_stats_df = pd.DataFrame()

        # Load team schedule data
        if team_schedule_filename is None:
            # Find most recent cleaned team schedule file
            files = [f for f in os.listdir(self.processed_dir)
                    if f.startswith("cleaned_team_schedules_") and f.endswith(".csv")]
            if not files:
                logger.error("No cleaned team schedule files found")
                team_schedule_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                team_schedule_filename = files[0]

                try:
                    team_schedule_path = os.path.join(self.processed_dir, team_schedule_filename)
                    logger.info(f"Loading team schedule from {team_schedule_path}")
                    team_schedule_df = pd.read_csv(team_schedule_path)
                    logger.info(f"Loaded {len(team_schedule_df)} team schedule records")
                except Exception as e:
                    logger.error(f"Error loading team schedule: {e}")
                    team_schedule_df = pd.DataFrame()
        else:
            try:
                team_schedule_path = os.path.join(self.processed_dir, team_schedule_filename)
                logger.info(f"Loading team schedule from {team_schedule_path}")
                team_schedule_df = pd.read_csv(team_schedule_path)
                logger.info(f"Loaded {len(team_schedule_df)} team schedule records")
            except Exception as e:
                logger.error(f"Error loading team schedule: {e}")
                team_schedule_df = pd.DataFrame()

        # Load odds data
        if odds_filename is None:
            # Find most recent cleaned odds file
            files = [f for f in os.listdir(self.processed_dir)
                    if f.startswith("cleaned_odds_data_") and f.endswith(".csv")]
            if not files:
                logger.error("No cleaned odds files found")
                odds_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                odds_filename = files[0]

                try:
                    odds_path = os.path.join(self.processed_dir, odds_filename)
                    logger.info(f"Loading odds data from {odds_path}")
                    odds_df = pd.read_csv(odds_path)
                    logger.info(f"Loaded {len(odds_df)} odds records")
                except Exception as e:
                    logger.error(f"Error loading odds data: {e}")
                    odds_df = pd.DataFrame()
        else:
            try:
                odds_path = os.path.join(self.processed_dir, odds_filename)
                logger.info(f"Loading odds data from {odds_path}")
                odds_df = pd.read_csv(odds_path)
                logger.info(f"Loaded {len(odds_df)} odds records")
            except Exception as e:
                logger.error(f"Error loading odds data: {e}")
                odds_df = pd.DataFrame()

        return {
            'player_data': player_df,
            'team_stats': team_stats_df,
            'team_schedules': team_schedule_df,
            'odds_data': odds_df
        }

    def create_player_features(self, player_df):
        """
        Create features from player game logs

        Args:
            player_df: DataFrame of player game logs

        Returns:
            DataFrame with player features
        """
        if player_df.empty:
            logger.warning("Empty player data, cannot create features")
            return pd.DataFrame()

        logger.info("Creating player features")

        # Ensure game_date is datetime
        if 'game_date' in player_df.columns:
            player_df['game_date'] = pd.to_datetime(player_df['game_date'])

        # Sort by player and date
        player_df = player_df.sort_values(['player_id', 'game_date'])

        # Create rolling window features (last 5, 10, 15, 20 games)
        feature_df = player_df.copy()

        # Convert string columns to appropriate types if needed
        numeric_columns = ['pts', 'mp_numeric', 'fga', 'fg3a', 'fta',
                          'fg_pct', 'fg3_pct', 'ft_pct', 'pts_per_min',
                          'usage_proxy']

        for col in numeric_columns:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

        # Rolling average features
        windows = [5, 10, 15, 20]
        stats_to_aggregate = ['pts', 'mp_numeric', 'pts_per_min', 'usage_proxy']

        for window in windows:
            for stat in stats_to_aggregate:
                if stat in feature_df.columns:
                    col_name = f'{stat}_last_{window}_avg'
                    feature_df[col_name] = feature_df.groupby('player_id')[stat].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )

        # Game-to-game change features
        for stat in stats_to_aggregate:
            if stat in feature_df.columns:
                col_name = f'{stat}_last_game'
                feature_df[col_name] = feature_df.groupby('player_id')[stat].shift(1)

                # Percent change from last game
                col_name = f'{stat}_pct_change'
                feature_df[col_name] = feature_df.groupby('player_id')[stat].pct_change()

        # Streaks and trends
        if 'pts' in feature_df.columns:
            # Calculate if player exceeded their season average in last game
            feature_df['season_avg_pts'] = feature_df.groupby(['player_id', 'season'])['pts'].transform('mean')
            feature_df['exceeded_avg_last_game'] = feature_df.groupby('player_id')['pts'].transform(
                lambda x: (x.shift(1) > x.mean()).astype(int)
            )

            # Calculate streak of exceeding/not exceeding season average
            feature_df['exceeded_avg'] = (feature_df['pts'] > feature_df['season_avg_pts']).astype(int)
            feature_df['consec_exceeded'] = feature_df.groupby('player_id')['exceeded_avg'].transform(
                lambda x: x.groupby((x != x.shift(1)).cumsum()).cumcount() + 1
            )
            feature_df['consec_not_exceeded'] = feature_df.groupby('player_id')['exceeded_avg'].transform(
                lambda x: (~x.astype(bool)).groupby((~x.astype(bool) != (~x.astype(bool)).shift(1)).cumsum()).cumcount() + 1
            )

        # Home/Away performance
        if 'is_home' in feature_df.columns:
            # Home/Away averages
            feature_df['home_avg_pts'] = feature_df[feature_df['is_home']].groupby('player_id')['pts'].transform('mean')
            feature_df['away_avg_pts'] = feature_df[~feature_df['is_home']].groupby('player_id')['pts'].transform('mean')
            feature_df['home_away_diff'] = feature_df['home_avg_pts'] - feature_df['away_avg_pts']

        # Rest days impact (if we have consecutive game dates)
        feature_df['days_rest'] = feature_df.groupby('player_id')['game_date'].diff().dt.days
        feature_df['days_rest'] = feature_df['days_rest'].fillna(3)  # Assume standard rest for first game

        # Create rest day categories
        feature_df['is_b2b'] = (feature_df['days_rest'] == 1).astype(int)
        feature_df['is_long_rest'] = (feature_df['days_rest'] > 3).astype(int)

        # Performance after different rest periods
        rest_windows = [1, 2, 3, 4]
        for rest in rest_windows:
            mask = feature_df['days_rest'] == rest
            feature_df[f'avg_pts_after_{rest}d_rest'] = feature_df[mask].groupby('player_id')['pts'].transform('mean')

        # Fill NaNs with player's average
        for col in feature_df.columns:
            if col not in ['player_id', 'game_date', 'season'] and feature_df[col].dtype in ['float64', 'int64']:
                feature_df[col] = feature_df.groupby('player_id')[col].transform(
                    lambda x: x.fillna(x.mean())
                )

        # Fill any remaining NaNs with 0
        feature_df = feature_df.fillna(0)

        logger.info(f"Created player features dataframe with {len(feature_df)} rows and {len(feature_df.columns)} columns")
        return feature_df

    def create_team_features(self, player_df, team_stats_df, team_schedule_df):
        """
        Create features related to teams and matchups

        Args:
            player_df: DataFrame of player features
            team_stats_df: DataFrame of team stats
            team_schedule_df: DataFrame of team schedules

        Returns:
            DataFrame with added team features
        """
        if player_df.empty or team_stats_df.empty or team_schedule_df.empty:
            logger.warning("Empty data, cannot create team features")
            return player_df

        logger.info("Creating team features")

        # Make a copy to avoid modifying the original
        feature_df = player_df.copy()

        # Ensure game_date is datetime
        for df in [feature_df, team_schedule_df]:
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])

        # Add team ID to player data (from team_schedule_df)
        if 'team_id' not in feature_df.columns and 'team_id' in team_schedule_df.columns:
            # We need to use the schedule to map player game dates to team IDs
            # This assumes players are not traded mid-season, which is a simplification

            # First, create a mapping of player to team using the most common team in schedule
            if 'player_id' in feature_df.columns and 'team_id' in team_schedule_df.columns:
                # We'll use a simple approach for this example
                # In a real scenario, you'd need more complex logic to handle trades

                # For each player game, try to find the corresponding team game
                feature_df['game_date_str'] = feature_df['game_date'].dt.strftime('%Y-%m-%d')
                team_schedule_df['game_date_str'] = team_schedule_df['game_date'].dt.strftime('%Y-%m-%d')

                # Merge on game date to find the team
                # This is a simplified approach and would need refinement in a real scenario
                team_mapping = pd.DataFrame(feature_df[['player_id', 'game_date_str']].drop_duplicates())
                team_mapping = team_mapping.merge(
                    team_schedule_df[['team_id', 'game_date_str']],
                    on='game_date_str',
                    how='left'
                )

                # Get most common team for each player
                player_team_map = team_mapping.groupby('player_id')['team_id'].agg(
                    lambda x: x.value_counts().index[0] if not x.empty else None
                ).reset_index()

                # Merge back to feature_df
                feature_df = feature_df.merge(player_team_map, on='player_id', how='left')

                # Clean up
                feature_df.drop('game_date_str', axis=1, inplace=True)

        # Add team stats to player data
        if 'team_id' in feature_df.columns:
            # Merge team stats
            if not team_stats_df.empty and 'team_id' in team_stats_df.columns:
                # Select team stats to include
                team_stats_cols = ['team_id', 'season']
                # Add offensive and defensive rating if available
                if 'advanced_off_rtg' in team_stats_df.columns:
                    team_stats_cols.append('advanced_off_rtg')
                if 'advanced_def_rtg' in team_stats_df.columns:
                    team_stats_cols.append('advanced_def_rtg')
                # Add pace if available
                if 'advanced_pace' in team_stats_df.columns:
                    team_stats_cols.append('advanced_pace')

                team_stats_subset = team_stats_df[team_stats_cols].copy()

                # Rename columns to avoid conflicts
                rename_dict = {col: f'team_{col}' for col in team_stats_subset.columns
                              if col not in ['team_id', 'season']}
                team_stats_subset = team_stats_subset.rename(columns=rename_dict)

                # Merge team stats with player data
                feature_df = feature_df.merge(
                    team_stats_subset,
                    on=['team_id', 'season'],
                    how='left'
                )

        # Add opponent team features
        if 'opp_id' in feature_df.columns:
            # Similar to team stats but for opponents
            if not team_stats_df.empty and 'team_id' in team_stats_df.columns:
                # Select team stats to include for opponents
                opp_stats_cols = ['team_id', 'season']
                if 'advanced_def_rtg' in team_stats_df.columns:
                    opp_stats_cols.append('advanced_def_rtg')

                opp_stats_subset = team_stats_df[opp_stats_cols].copy()

                # Rename columns to avoid conflicts
                rename_dict = {col: f'opp_{col}' for col in opp_stats_subset.columns
                              if col not in ['team_id', 'season']}
                rename_dict['team_id'] = 'opp_id'  # Also rename team_id to opp_id for merging
                opp_stats_subset = opp_stats_subset.rename(columns=rename_dict)

                # Merge opponent stats with player data
                feature_df = feature_df.merge(
                    opp_stats_subset,
                    on=['opp_id', 'season'],
                    how='left'
                )

        # Identify back-to-back games for teams
        if 'team_id' in feature_df.columns and 'game_date' in feature_df.columns:
            # Sort by team and date
            tmp_df = feature_df[['team_id', 'game_date']].drop_duplicates().sort_values(['team_id', 'game_date'])

            # Calculate days between games
            tmp_df['days_between'] = tmp_df.groupby('team_id')['game_date'].diff().dt.days

            # Mark back-to-back games
            tmp_df['team_b2b'] = (tmp_df['days_between'] == 1).astype(int)

            # Merge back to feature_df
            feature_df = feature_df.merge(
                tmp_df[['team_id', 'game_date', 'team_b2b']],
                on=['team_id', 'game_date'],
                how='left'
            )

            # Fill NAs with 0
            feature_df['team_b2b'] = feature_df['team_b2b'].fillna(0)

        # Fill any remaining NaNs with 0
        feature_df = feature_df.fillna(0)

        logger.info(f"Added team features to dataframe (now {len(feature_df.columns)} columns)")
        return feature_df

    def create_matchup_features(self, player_df):
        """
        Create features related to player vs. specific opponents

        Args:
            player_df: DataFrame of player features

        Returns:
            DataFrame with added matchup features
        """
        if player_df.empty:
            logger.warning("Empty data, cannot create matchup features")
            return player_df

        logger.info("Creating matchup features")

        # Make a copy to avoid modifying the original
        feature_df = player_df.copy()

        # Check if we have opponent data
        if 'opp_id' not in feature_df.columns:
            logger.warning("No opponent ID column found, skipping matchup features")
            return feature_df

        # Calculate player's average points against each opponent
        avg_vs_opp = feature_df.groupby(['player_id', 'opp_id'])['pts'].mean().reset_index()
        avg_vs_opp.rename(columns={'pts': 'avg_pts_vs_opp'}, inplace=True)

        # Merge back to feature_df
        feature_df = feature_df.merge(
            avg_vs_opp,
            on=['player_id', 'opp_id'],
            how='left'
        )

        # Calculate difference from player's overall average
        if 'season_avg_pts' in feature_df.columns:
            feature_df['pts_vs_opp_diff'] = feature_df['avg_pts_vs_opp'] - feature_df['season_avg_pts']

        # Calculate average for home/away against this opponent
        if 'is_home' in feature_df.columns:
            home_avg_vs_opp = feature_df[feature_df['is_home']].groupby(['player_id', 'opp_id'])['pts'].mean().reset_index()
            home_avg_vs_opp.rename(columns={'pts': 'home_avg_pts_vs_opp'}, inplace=True)

            away_avg_vs_opp = feature_df[~feature_df['is_home']].groupby(['player_id', 'opp_id'])['pts'].mean().reset_index()
            away_avg_vs_opp.rename(columns={'pts': 'away_avg_pts_vs_opp'}, inplace=True)

            # Merge back to feature_df
            feature_df = feature_df.merge(home_avg_vs_opp, on=['player_id', 'opp_id'], how='left')
            feature_df = feature_df.merge(away_avg_vs_opp, on=['player_id', 'opp_id'], how='left')

        # Fill NaNs with player's average
        for col in ['avg_pts_vs_opp', 'pts_vs_opp_diff', 'home_avg_pts_vs_opp', 'away_avg_pts_vs_opp']:
            if col in feature_df.columns:
                feature_df[col] = feature_df.groupby('player_id')[col].transform(
                    lambda x: x.fillna(x.mean())
                )

        # Fill any remaining NaNs with 0
        feature_df = feature_df.fillna(0)

        logger.info(f"Added matchup features to dataframe (now {len(feature_df.columns)} columns)")
        return feature_df

    def create_target_variables(self, feature_df, odds_df=None):
        """
        Create target variables for modeling

        Args:
            feature_df: DataFrame of player features
            odds_df: DataFrame of betting odds

        Returns:
            DataFrame with added target variables
        """
        if feature_df.empty:
            logger.warning("Empty data, cannot create target variables")
            return feature_df

        logger.info("Creating target variables")

        # Make a copy to avoid modifying the original
        result_df = feature_df.copy()

        # The main target is 'pts' (already in the data)
        # But we also want to create over/under indicators for different point thresholds
        common_thresholds = [10.5, 15.5, 20.5, 25.5, 30.5, 35.5, 40.5]

        if 'pts' in result_df.columns:
            for threshold in common_thresholds:
                col_name = f'over_{threshold:.1f}'
                result_df[col_name] = (result_df['pts'] > threshold).astype(int)

        # If we have odds data, add additional targets
        if odds_df is not None and not odds_df.empty:
            logger.info("Adding odds-based target variables")

            # Make sure date columns are datetime
            if 'game_date' in result_df.columns:
                result_df['game_date'] = pd.to_datetime(result_df['game_date'])

            if 'game_date' in odds_df.columns:
                odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])

            # We need to match player names between the datasets
            # This is complex in real life but simplified here
            if 'player_id' in result_df.columns and 'player' in odds_df.columns:
                # Create a mapping of player ID to player name
                player_name_map = result_df[['player_id']].drop_duplicates()

                # In a real implementation, you'd need a proper mapping solution here
                # For this example, we'll assume player_id contains the player's name
                player_name_map['player_name'] = player_name_map['player_id'].apply(
                    lambda x: x.replace('01', '').replace('02', '').replace('03', '').title()
                )

                # Merge odds data based on player name and date
                # First, add player name to result_df
                result_df = result_df.merge(player_name_map, on='player_id', how='left')

                # Then merge odds data
                result_df = result_df.merge(
                    odds_df[['player', 'game_date', 'line']],
                    left_on=['player_name', 'game_date'],
                    right_on=['player', 'game_date'],
                    how='left'
                )

                # Create over/under for the actual line
                result_df['over_line'] = (result_df['pts'] > result_df['line']).astype(int)
                result_df['under_line'] = (result_df['pts'] < result_df['line']).astype(int)
                result_df['push_line'] = (result_df['pts'] == result_df['line']).astype(int)

                # Calculate difference from line
                result_df['pts_minus_line'] = result_df['pts'] - result_df['line']

        logger.info(f"Added target variables to dataframe (now {len(result_df.columns)} columns)")
        return result_df

    def create_train_test_sets(self, feature_df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets

        Args:
            feature_df: DataFrame of features and targets
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training and testing DataFrames
        """
        if feature_df.empty:
            logger.warning("Empty data, cannot create train/test sets")
            return None

        logger.info("Creating train/test sets")

        # Make a copy to avoid modifying the original
        df = feature_df.copy()

        # Ensure game_date is datetime
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])

        # Sort by date
        df = df.sort_values('game_date')

        # Time-based split is more realistic than random split for this problem
        # Use the last test_size proportion of games as test set
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Created training set with {len(train_df)} rows and test set with {len(test_df)} rows")

        return {
            'train': train_df,
            'test': test_df
        }

    def create_modelling_dataset(self, cleaned_data=None):
        """
        Create a complete dataset for modelling

        Args:
            cleaned_data: Dictionary with cleaned DataFrames (if None, load most recent)

        Returns:
            Dictionary with training and testing DataFrames
        """
        # Load data if not provided
        if cleaned_data is None:
            cleaned_data = self.load_cleaned_data()

        # Extract individual DataFrames
        player_df = cleaned_data['player_data']
        team_stats_df = cleaned_data['team_stats']
        team_schedule_df = cleaned_data['team_schedules']
        odds_df = cleaned_data['odds_data']

        # Create features
        player_features = self.create_player_features(player_df)
        team_features = self.create_team_features(player_features, team_stats_df, team_schedule_df)
        matchup_features = self.create_matchup_features(team_features)

        # Create target variables
        full_dataset = self.create_target_variables(matchup_features, odds_df)

        # Split into train/test sets
        train_test_sets = self.create_train_test_sets(full_dataset)

        # Save the datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_path = os.path.join(self.output_dir, f"train_dataset_{timestamp}.csv")
        test_path = os.path.join(self.output_dir, f"test_dataset_{timestamp}.csv")

        if train_test_sets is not None:
            train_test_sets['train'].to_csv(train_path, index=False)
            train_test_sets['test'].to_csv(test_path, index=False)

            logger.info(f"Saved training dataset to {train_path}")
            logger.info(f"Saved testing dataset to {test_path}")

        return train_test_sets

def main():
    """
    Main function to run the feature engineer
    """
    # Create feature engineer
    engineer = NBAFeatureEngineer()

    # Create modelling dataset
    datasets = engineer.create_modelling_dataset()

    # Print summary
    if datasets is not None:
        print(f"Training set: {len(datasets['train'])} rows, {len(datasets['train'].columns)} columns")
        print(f"Testing set: {len(datasets['test'])} rows, {len(datasets['test'].columns)} columns")

        # Print some key features
        print("\nKey features in dataset:")
        feature_categories = {
            'Basic': ['pts', 'mp_numeric', 'season_avg_pts', 'team_id', 'opp_id'],
            'Rolling Averages': [col for col in datasets['train'].columns if '_avg' in col],
            'Streaks': [col for col in datasets['train'].columns if 'consec_' in col or 'streak' in col],
            'Rest': [col for col in datasets['train'].columns if 'rest' in col or 'b2b' in col],
            'Matchups': [col for col in datasets['train'].columns if 'vs_opp' in col],
            'Targets': [col for col in datasets['train'].columns if 'over_' in col or 'under_' in col]
        }

        for category, features in feature_categories.items():
            present_features = [f for f in features if f in datasets['train'].columns]
            print(f"{category}: {', '.join(present_features[:5])}{'...' if len(present_features) > 5 else ''}")

if __name__ == "__main__":
    main()