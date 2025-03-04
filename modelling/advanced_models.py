import pandas as pd
import numpy as np
import os
import logging
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("advanced_models.log"), logging.StreamHandler()]
)
logger = logging.getLogger("advanced_models")

class NBAAdvancedModeler:
    """
    Advanced models for NBA player points prediction
    """

    def __init__(self, processed_dir="../data/processed", models_dir="../models"):
        """
        Initialize the advanced modeler

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

    def prepare_features(self, train_df, test_df, scale=True):
        """
        Prepare features for modeling

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            scale: Whether to scale numeric features

        Returns:
            Dictionary with prepared features
        """
        if train_df.empty or test_df.empty:
            logger.error("Empty DataFrames, cannot prepare features")
            return None

        try:
            # Select numeric features (excluding target)
            numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['pts', 'pts_target']]

            # Select categorical features
            cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

            logger.info(f"Selected {len(numeric_cols)} numeric features and {len(cat_cols)} categorical features")

            # Create X and y for both sets
            X_train = train_df[numeric_cols].copy()
            y_train = train_df['pts'].values

            X_test = test_df[numeric_cols].copy()
            y_test = test_df['pts'].values

            # Handle infinite values
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())  # Use train mean for test set

            # Scale features if requested
            if scale:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
                scaler = None

            return {
                'X_train': X_train.values,
                'y_train': y_train,
                'X_test': X_test.values,
                'y_test': y_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'features': numeric_cols,
                'categorical_features': cat_cols,
                'scaler': scaler
            }

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def xgboost_model(self, feature_data, params=None):
        """
        Build an XGBoost model

        Args:
            feature_data: Dictionary with feature arrays from prepare_features
            params: Custom parameters for the model (if None, use defaults)

        Returns:
            Dictionary with predictions and model info
        """
        logger.info("Building XGBoost model")

        if feature_data is None:
            logger.error("No feature data provided")
            return None

        X_train = feature_data['X_train']
        y_train = feature_data['y_train']
        X_test = feature_data['X_test']
        y_test = feature_data['y_test']
        features = feature_data['features']

        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        # Use custom params if provided
        if params is not None:
            default_params.update(params)

        # Train model
        model = XGBRegressor(**default_params)

        try:
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate error metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logger.info(f"XGBoost model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

            # Feature importance
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)

            logger.info("Top 10 most important features:")
            for _, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f"xgboost_{timestamp}.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': features,
                    'params': default_params
                }, f)

            logger.info(f"Saved XGBoost model to {model_path}")

            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance.head(20))
            plt.title('XGBoost Feature Importance')
            plt.tight_layout()

            plot_path = os.path.join(self.models_dir, f"xgb_feature_importance_{timestamp}.png")
            plt.savefig(plot_path)
            logger.info(f"Saved feature importance plot to {plot_path}")

            return {
                'name': 'xgboost',
                'predictions': y_pred,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'model': model,
                'importance': importance,
                'model_path': model_path,
                'params': default_params
            }

        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return None

    def evaluate_models(self, datasets=None):
        """
        Evaluate XGBoost model

        Args:
            datasets: Dictionary with training and testing DataFrames

        Returns:
            Dictionary with model results
        """
        logger.info("Evaluating XGBoost model")

        # Load datasets if not provided
        if datasets is None:
            logger.info("Loading datasets")
            train_df = self.load_datasets()['train']
            test_df = self.load_datasets()['test']
            datasets = {'train': train_df, 'test': test_df}

        # Prepare features
        feature_data = self.prepare_features(datasets['train'], datasets['test'])

        if feature_data is None:
            logger.error("Failed to prepare features")
            return None

        # Train and evaluate XGBoost model
        results = {}

        # XGBoost
        xgb_result = self.xgboost_model(feature_data)
        if xgb_result:
            results['xgboost'] = xgb_result

        if not results:
            logger.error("No models were successfully trained")
            return None

        # Create comparison data (even though it's just one model)
        comparison = pd.DataFrame([
            {
                'model': name,
                'mae': r['metrics']['mae'],
                'rmse': r['metrics']['rmse'],
                'r2': r['metrics']['r2']
            }
            for name, r in results.items()
        ])

        # Log model performance
        for _, row in comparison.iterrows():
            logger.info(f"  {row['model']}: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, R²={row['r2']:.4f}")

        # Save comparison to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.models_dir, f"model_comparison_{timestamp}.csv")
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"Saved model comparison to {comparison_path}")

        # Always use XGBoost as the best model
        best_model = 'xgboost'
        best_model_predictions = results[best_model]['predictions']
        best_model_metrics = results[best_model]['metrics']

        plt.figure(figsize=(12, 8))

        # Plotting actual vs predicted
        plt.subplot(2, 1, 1)
        plt.scatter(datasets['test']['pts'], best_model_predictions, alpha=0.5)
        plt.plot([datasets['test']['pts'].min(), datasets['test']['pts'].max()],
                 [datasets['test']['pts'].min(), datasets['test']['pts'].max()],
                 'r--')
        plt.xlabel('Actual Points')
        plt.ylabel('Predicted Points')
        plt.title(f'XGBoost Model - Actual vs Predicted')

        # Plotting error distribution
        plt.subplot(2, 1, 2)
        errors = datasets['test']['pts'] - best_model_predictions
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - MAE: {best_model_metrics["mae"]:.2f}, RMSE: {best_model_metrics["rmse"]:.2f}')

        plt.tight_layout()
        error_plot_path = os.path.join(self.models_dir, f"{best_model}_errors_{timestamp}.png")
        plt.savefig(error_plot_path)
        logger.info(f"Saved error distribution plot to {error_plot_path}")

        # Print model-specific details
        importance = results[best_model]['importance']
        print("\nTop 10 features:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return {
            'results': results,
            'comparison': comparison,
            'best_model': best_model
        }

def main():
    """
    Main function to run the advanced modeler
    """
    modeler = NBAAdvancedModeler()

    evaluation = modeler.evaluate_models()

    if evaluation:
        best_model = evaluation['best_model']
        metrics = evaluation['results'][best_model]['metrics']

        print(f"\nXGBoost model performance:")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.4f}")

        importance = evaluation['results'][best_model]['importance']
        print("\nTop 10 features:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("No evaluation results were generated.")

if __name__ == "__main__":
    main()