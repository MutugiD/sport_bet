import pandas as pd
import numpy as np
import os
import logging
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("baseline_models.log"), logging.StreamHandler()]
)
logger = logging.getLogger("baseline_models")

class NBABaselineModeler:
    """
    Baseline models for NBA player points prediction
    """

    def __init__(self, processed_dir="../data/processed", models_dir="../models"):
        """
        Initialize the baseline modeler

        Args:
            processed_dir: Directory with processed data
            models_dir: Directory to save trained models
        """
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def load_datasets(self, train_filename=None, test_filename=None):
        """
        Load training and testing datasets

        Args:
            train_filename: Training data filename (if None, load most recent)
            test_filename: Testing data filename (if None, load most recent)

        Returns:
            Dictionary with training and testing DataFrames
        """
        # Load training data
        if train_filename is None:
            # Find most recent training dataset file
            files = [f for f in os.listdir(self.processed_dir)
                    if f.startswith("train_dataset_") and f.endswith(".csv")]
            if not files:
                logger.error("No training dataset files found")
                train_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                train_filename = files[0]

                try:
                    train_path = os.path.join(self.processed_dir, train_filename)
                    logger.info(f"Loading training data from {train_path}")
                    train_df = pd.read_csv(train_path)
                    logger.info(f"Loaded {len(train_df)} training records")
                except Exception as e:
                    logger.error(f"Error loading training data: {e}")
                    train_df = pd.DataFrame()
        else:
            try:
                train_path = os.path.join(self.processed_dir, train_filename)
                logger.info(f"Loading training data from {train_path}")
                train_df = pd.read_csv(train_path)
                logger.info(f"Loaded {len(train_df)} training records")
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                train_df = pd.DataFrame()

        # Load testing data
        if test_filename is None:
            # Find most recent testing dataset file
            files = [f for f in os.listdir(self.processed_dir)
                    if f.startswith("test_dataset_") and f.endswith(".csv")]
            if not files:
                logger.error("No testing dataset files found")
                test_df = pd.DataFrame()
            else:
                # Sort by timestamp in filename
                files.sort(reverse=True)
                test_filename = files[0]

                try:
                    test_path = os.path.join(self.processed_dir, test_filename)
                    logger.info(f"Loading testing data from {test_path}")
                    test_df = pd.read_csv(test_path)
                    logger.info(f"Loaded {len(test_df)} testing records")
                except Exception as e:
                    logger.error(f"Error loading testing data: {e}")
                    test_df = pd.DataFrame()
        else:
            try:
                test_path = os.path.join(self.processed_dir, test_filename)
                logger.info(f"Loading testing data from {test_path}")
                test_df = pd.read_csv(test_path)
                logger.info(f"Loaded {len(test_df)} testing records")
            except Exception as e:
                logger.error(f"Error loading testing data: {e}")
                test_df = pd.DataFrame()

        return {
            'train': train_df,
            'test': test_df
        }

    def historical_average_model(self, train_df, test_df):
        """
        Predict based on player's historical average points

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame

        Returns:
            Dictionary with predictions and model info
        """
        logger.info("Building historical average model")

        if train_df.empty or test_df.empty:
            logger.error("Empty data, cannot build historical average model")
            return None

        if 'player_id' not in train_df.columns or 'pts' not in train_df.columns:
            logger.error("Required columns not found in data")
            return None

        # Calculate historical average for each player
        player_avgs = train_df.groupby('player_id')['pts'].mean().reset_index()
        player_avgs.rename(columns={'pts': 'avg_pts'}, inplace=True)

        # Merge with test data to get predictions
        test_with_pred = test_df.merge(player_avgs, on='player_id', how='left')

        # For players not in training data, use overall average
        overall_avg = train_df['pts'].mean()
        test_with_pred['avg_pts'] = test_with_pred['avg_pts'].fillna(overall_avg)

        # Calculate error metrics
        y_true = test_with_pred['pts']
        y_pred = test_with_pred['avg_pts']

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        logger.info(f"Historical average model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

        return {
            'name': 'historical_average',
            'predictions': y_pred,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            },
            'model': player_avgs
        }

    def rolling_average_model(self, train_df, test_df, window=5):
        """
        Predict based on player's rolling average points

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            window: Number of games to use for rolling average

        Returns:
            Dictionary with predictions and model info
        """
        logger.info(f"Building {window}-game rolling average model")

        if train_df.empty or test_df.empty:
            logger.error("Empty data, cannot build rolling average model")
            return None

        if 'player_id' not in train_df.columns or 'pts' not in train_df.columns:
            logger.error("Required columns not found in data")
            return None

        # Look for pre-calculated rolling average
        rolling_col = f'pts_last_{window}_avg'

        if rolling_col in test_df.columns:
            logger.info(f"Using pre-calculated {rolling_col}")

            # Calculate error metrics
            y_true = test_df['pts']
            y_pred = test_df[rolling_col]

            # Handle possible missing values
            y_pred = y_pred.fillna(train_df['pts'].mean())

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            logger.info(f"Rolling average model ({window} games) - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

            return {
                'name': f'rolling_average_{window}',
                'predictions': y_pred,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'window': window
            }
        else:
            logger.warning(f"Rolling average column {rolling_col} not found in data")
            return None

    def season_average_opponent_adjusted(self, train_df, test_df):
        """
        Predict based on player's season average adjusted for opponent defense

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame

        Returns:
            Dictionary with predictions and model info
        """
        logger.info("Building season average opponent-adjusted model")

        if train_df.empty or test_df.empty:
            logger.error("Empty data, cannot build opponent-adjusted model")
            return None

        required_cols = ['player_id', 'pts', 'opp_id', 'season_avg_pts']
        if not all(col in train_df.columns for col in required_cols):
            logger.error(f"Required columns not found in data: {required_cols}")
            return None

        # Calculate opponent defense factors
        # How many points on average do players score against this opponent compared to their season average?
        train_df['pts_ratio'] = train_df['pts'] / train_df['season_avg_pts']
        opponent_factors = train_df.groupby('opp_id')['pts_ratio'].mean().reset_index()
        opponent_factors.rename(columns={'pts_ratio': 'opp_factor'}, inplace=True)

        # Merge with test data
        test_with_pred = test_df.merge(opponent_factors, on='opp_id', how='left')

        # For opponents not in training data, use factor of 1.0 (no adjustment)
        test_with_pred['opp_factor'] = test_with_pred['opp_factor'].fillna(1.0)

        # Make prediction: season average * opponent factor
        test_with_pred['pred_pts'] = test_with_pred['season_avg_pts'] * test_with_pred['opp_factor']

        # Calculate error metrics
        y_true = test_with_pred['pts']
        y_pred = test_with_pred['pred_pts']

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        logger.info(f"Opponent-adjusted model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

        return {
            'name': 'opponent_adjusted',
            'predictions': y_pred,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            },
            'model': opponent_factors
        }

    def simple_linear_regression(self, train_df, test_df):
        """
        Build a simple linear regression model

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame

        Returns:
            Dictionary with predictions and model info
        """
        logger.info("Building simple linear regression model")

        if train_df.empty or test_df.empty:
            logger.error("Empty data, cannot build linear regression model")
            return None

        # Select basic features for linear regression
        basic_features = ['season_avg_pts']

        # Add rolling averages if available
        rolling_features = [col for col in train_df.columns
                           if col.startswith('pts_last_') and col.endswith('_avg')]

        # Add opponent info if available
        opponent_features = [col for col in train_df.columns
                           if 'opp_' in col or 'vs_opp' in col]

        # Add rest features if available
        rest_features = [col for col in train_df.columns
                       if 'rest' in col or 'b2b' in col]

        # Add home/away features if available
        home_away_features = [col for col in train_df.columns
                             if 'home_' in col or 'away_' in col or col == 'is_home']

        # Combine all features
        selected_features = basic_features + rolling_features + opponent_features + rest_features + home_away_features
        selected_features = list(set(selected_features))  # Remove duplicates

        # Keep only features that exist in both train and test
        features = [f for f in selected_features
                   if f in train_df.columns and f in test_df.columns]

        if not features:
            logger.error("No usable features found for linear regression")
            return None

        # Prepare data
        X_train = train_df[features].fillna(0)
        y_train = train_df['pts']
        X_test = test_df[features].fillna(0)
        y_test = test_df['pts']

        # Train the model
        model = LinearRegression()
        try:
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate error metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logger.info(f"Linear regression model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

            # Feature importance
            importance = pd.DataFrame({
                'feature': features,
                'coefficient': model.coef_
            })
            importance = importance.sort_values('coefficient', key=abs, ascending=False)

            logger.info("Top 5 most important features:")
            for _, row in importance.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f"linear_regression_{timestamp}.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            logger.info(f"Saved linear regression model to {model_path}")

            return {
                'name': 'linear_regression',
                'predictions': y_pred,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'model': model,
                'importance': importance,
                'model_path': model_path
            }

        except Exception as e:
            logger.error(f"Error training linear regression model: {e}")
            return None

    def ridge_regression(self, train_df, test_df, alpha=1.0):
        """
        Build a ridge regression model with regularization

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            alpha: Regularization parameter

        Returns:
            Dictionary with predictions and model info
        """
        logger.info(f"Building ridge regression model (alpha={alpha})")

        if train_df.empty or test_df.empty:
            logger.error("Empty data, cannot build ridge regression model")
            return None

        # Select more features for ridge regression
        # It can handle multicollinearity better due to regularization
        numeric_features = []

        for col in train_df.columns:
            # Skip non-numeric columns, ID columns, and target
            if (col in ['pts', 'player_id', 'game_date', 'team_id', 'opp_id'] or
                train_df[col].dtype not in ['int64', 'float64']):
                continue

            # Ensure column exists in both dataframes
            if col in test_df.columns:
                numeric_features.append(col)

        if not numeric_features:
            logger.error("No usable features found for ridge regression")
            return None

        # Prepare data
        X_train = train_df[numeric_features].fillna(0)
        y_train = train_df['pts']
        X_test = test_df[numeric_features].fillna(0)
        y_test = test_df['pts']

        # Train the model
        model = Ridge(alpha=alpha)
        try:
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate error metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logger.info(f"Ridge regression model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

            # Feature importance
            importance = pd.DataFrame({
                'feature': numeric_features,
                'coefficient': model.coef_
            })
            importance = importance.sort_values('coefficient', key=abs, ascending=False)

            logger.info("Top 10 most important features:")
            for _, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f"ridge_regression_{timestamp}.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            logger.info(f"Saved ridge regression model to {model_path}")

            return {
                'name': 'ridge_regression',
                'predictions': y_pred,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'model': model,
                'importance': importance,
                'model_path': model_path,
                'features': numeric_features
            }

        except Exception as e:
            logger.error(f"Error training ridge regression model: {e}")
            return None

    def evaluate_models(self, datasets=None):
        """
        Evaluate all baseline models

        Args:
            datasets: Dictionary with train and test DataFrames (if None, load most recent)

        Returns:
            Dictionary with model results
        """
        # Load data if not provided
        if datasets is None:
            datasets = self.load_datasets()

        train_df = datasets['train']
        test_df = datasets['test']

        if train_df.empty or test_df.empty:
            logger.error("Empty data, cannot evaluate models")
            return None

        # Run all models
        results = {}

        # Historical average model
        hist_avg_result = self.historical_average_model(train_df, test_df)
        if hist_avg_result:
            results['historical_average'] = hist_avg_result

        # Rolling average models with different windows
        for window in [5, 10, 15]:
            rolling_result = self.rolling_average_model(train_df, test_df, window=window)
            if rolling_result:
                results[f'rolling_average_{window}'] = rolling_result

        # Opponent-adjusted model
        opp_adj_result = self.season_average_opponent_adjusted(train_df, test_df)
        if opp_adj_result:
            results['opponent_adjusted'] = opp_adj_result

        # Linear regression model
        linear_result = self.simple_linear_regression(train_df, test_df)
        if linear_result:
            results['linear_regression'] = linear_result

        # Ridge regression model
        ridge_result = self.ridge_regression(train_df, test_df, alpha=1.0)
        if ridge_result:
            results['ridge_regression'] = ridge_result

        # Compare model performance
        comparison = pd.DataFrame([
            {
                'model': name,
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'r2': result['metrics']['r2']
            }
            for name, result in results.items()
        ])

        # Sort by MAE (you can change to RMSE or R2)
        comparison = comparison.sort_values('mae')

        logger.info("Model performance comparison:")
        for _, row in comparison.iterrows():
            logger.info(f"  {row['model']}: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, R²={row['r2']:.4f}")

        # Save comparison table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.models_dir, f"model_comparison_{timestamp}.csv")
        comparison.to_csv(comparison_path, index=False)

        logger.info(f"Saved model comparison to {comparison_path}")

        # Visualize error distribution for the best model
        best_model = comparison.iloc[0]['model']
        best_predictions = results[best_model]['predictions']

        plt.figure(figsize=(10, 6))
        errors = test_df['pts'] - best_predictions
        sns.histplot(errors, kde=True)
        plt.title(f'Error Distribution for {best_model}')
        plt.xlabel('Prediction Error (actual - predicted)')
        plt.ylabel('Frequency')

        error_plot_path = os.path.join(self.models_dir, f"{best_model}_errors_{timestamp}.png")
        plt.savefig(error_plot_path)
        logger.info(f"Saved error distribution plot to {error_plot_path}")

        return {
            'results': results,
            'comparison': comparison,
            'best_model': best_model
        }

def main():
    """
    Main function to run the baseline modeler
    """
    # Create baseline modeler
    modeler = NBABaselineModeler()

    # Evaluate models
    evaluation = modeler.evaluate_models()

    # Print summary
    if evaluation:
        best_model = evaluation['best_model']
        best_metrics = evaluation['comparison'][evaluation['comparison']['model'] == best_model].iloc[0]

        print(f"\nBest model: {best_model}")
        print(f"MAE: {best_metrics['mae']:.2f}")
        print(f"RMSE: {best_metrics['rmse']:.2f}")
        print(f"R²: {best_metrics['r2']:.4f}")

        # Print model-specific details
        if best_model == 'linear_regression' or best_model == 'ridge_regression':
            importance = evaluation['results'][best_model]['importance']
            print("\nTop 5 features:")
            for _, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['coefficient']:.4f}")

if __name__ == "__main__":
    main()