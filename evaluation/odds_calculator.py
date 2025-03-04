import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.optimize import minimize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("odds_calculator.log"), logging.StreamHandler()]
)
logger = logging.getLogger("odds_calculator")

class BettingOddsCalculator:
    """
    Calculate fair betting odds from model predictions and compare with sportsbook lines
    """

    def __init__(self, models_dir="../models", results_dir="../data/results"):
        """
        Initialize the odds calculator

        Args:
            models_dir: Directory with trained models
            results_dir: Directory to save results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def load_model(self, model_path):
        """
        Load a trained model

        Args:
            model_path: Path to the model file

        Returns:
            Dictionary with model and metadata
        """
        logger.info(f"Loading model from {model_path}")

        try:
            # Check file extension
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Check if model_data is already a dict with metadata
                if isinstance(model_data, dict) and 'model' in model_data:
                    return model_data
                else:
                    # Simple model without metadata
                    return {'model': model_data}

            elif model_path.endswith('.h5'):
                # Keras model
                try:
                    from tensorflow.keras.models import load_model
                    model = load_model(model_path)

                    # Try to load associated metadata
                    metadata_path = model_path.replace('.h5', '_metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                        return {'model': model, **metadata}
                    else:
                        return {'model': model}
                except ImportError:
                    logger.error("Could not import tensorflow to load Keras model")
                    return None
            else:
                logger.error(f"Unsupported model file format: {model_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def predict_points_distribution(self, model_data, player_data, point_range=(0, 60), steps=61):
        """
        Generate a probability distribution for points scored

        Args:
            model_data: Dictionary with model and metadata
            player_data: DataFrame with player features
            point_range: Tuple with min and max points to consider
            steps: Number of steps in the point range

        Returns:
            Dictionary with point range and probabilities
        """
        logger.info("Generating points probability distribution")

        model = model_data.get('model')
        if model is None:
            logger.error("No model found in model_data")
            return None

        try:
            # Make point prediction
            features = model_data.get('features')

            # If features list is provided, use only those features
            if features is not None:
                # Keep only features that exist in the dataset
                available_features = [f for f in features if f in player_data.columns]
                X = player_data[available_features].values.reshape(1, -1)
            else:
                # Use all numeric features except target variables
                numeric_cols = player_data.select_dtypes(include=['float64', 'int64']).columns
                target_cols = [col for col in numeric_cols if col.startswith('over_') or col == 'pts']
                feature_cols = [col for col in numeric_cols if col not in target_cols]
                X = player_data[feature_cols].values.reshape(1, -1)

            # Scale if scaler is provided
            scaler = model_data.get('scaler')
            if scaler is not None:
                X = scaler.transform(X)

            # Get mean prediction
            mean_prediction = float(model.predict(X)[0])

            # Calculate standard deviation (model-specific)
            # For this example, we'll use a simple approach based on model type
            if hasattr(model, 'predict_proba'):
                # For classifiers that provide probabilities
                std_dev = 5.0  # Fixed value for demonstration
            elif hasattr(model, 'estimators_'):
                # For ensemble methods, use the std of individual estimator predictions
                if hasattr(model, 'estimators_') and isinstance(model.estimators_, list):
                    predictions = [estimator.predict(X)[0] for estimator in model.estimators_]
                    std_dev = max(2.0, float(np.std(predictions)))
                else:
                    std_dev = 5.0
            else:
                # Default standard deviation
                std_dev = 5.0

            # Generate points range
            points = np.linspace(point_range[0], point_range[1], steps)

            # Calculate probabilities using normal distribution
            probs = norm.pdf(points, mean_prediction, std_dev)

            # Normalize to ensure sum equals 1
            probs = probs / probs.sum()

            logger.info(f"Generated distribution with mean={mean_prediction:.1f}, std={std_dev:.1f}")

            return {
                'points': points,
                'probabilities': probs,
                'mean': mean_prediction,
                'std': std_dev
            }

        except Exception as e:
            logger.error(f"Error generating points distribution: {e}")
            return None

    def calculate_over_under_probs(self, distribution, thresholds):
        """
        Calculate probabilities for over/under bets at different thresholds

        Args:
            distribution: Dictionary with points distribution
            thresholds: List of point thresholds

        Returns:
            DataFrame with over/under probabilities and fair odds
        """
        logger.info(f"Calculating over/under probabilities for thresholds: {thresholds}")

        if distribution is None:
            logger.error("No distribution provided")
            return None

        try:
            points = distribution['points']
            probs = distribution['probabilities']

            results = []

            for threshold in thresholds:
                # Find probability of scoring over threshold
                over_prob = sum(probs[points > threshold])
                under_prob = 1 - over_prob

                # Calculate fair odds (no vig)
                if over_prob > 0:
                    over_fair_decimal = 1 / over_prob
                    over_fair_american = self.decimal_to_american(over_fair_decimal)
                else:
                    over_fair_decimal = float('inf')
                    over_fair_american = float('inf')

                if under_prob > 0:
                    under_fair_decimal = 1 / under_prob
                    under_fair_american = self.decimal_to_american(under_fair_decimal)
                else:
                    under_fair_decimal = float('inf')
                    under_fair_american = float('inf')

                results.append({
                    'threshold': threshold,
                    'over_prob': over_prob,
                    'under_prob': under_prob,
                    'over_fair_decimal': over_fair_decimal,
                    'under_fair_decimal': under_fair_decimal,
                    'over_fair_american': over_fair_american,
                    'under_fair_american': under_fair_american
                })

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error calculating over/under probabilities: {e}")
            return None

    def estimate_implied_probability(self, american_odds):
        """
        Convert American odds to implied probability

        Args:
            american_odds: American style odds

        Returns:
            Implied probability (0-1)
        """
        try:
            if american_odds > 0:
                return 100 / (american_odds + 100)
            else:
                return abs(american_odds) / (abs(american_odds) + 100)
        except:
            return np.nan

    def decimal_to_american(self, decimal_odds):
        """
        Convert decimal odds to American odds

        Args:
            decimal_odds: Decimal style odds

        Returns:
            American style odds
        """
        try:
            if decimal_odds >= 2:
                return (decimal_odds - 1) * 100
            else:
                return -100 / (decimal_odds - 1)
        except:
            return np.nan

    def american_to_decimal(self, american_odds):
        """
        Convert American odds to decimal odds

        Args:
            american_odds: American style odds

        Returns:
            Decimal style odds
        """
        try:
            if american_odds > 0:
                return 1 + (american_odds / 100)
            else:
                return 1 + (100 / abs(american_odds))
        except:
            return np.nan

    def calculate_edge(self, fair_prob, market_odds):
        """
        Calculate the betting edge

        Args:
            fair_prob: Fair probability (0-1)
            market_odds: Market odds (American)

        Returns:
            Edge percentage
        """
        try:
            market_prob = self.estimate_implied_probability(market_odds)
            edge = (fair_prob * self.american_to_decimal(market_odds) - 1) * 100
            return edge
        except:
            return np.nan

    def compare_with_market(self, fair_odds_df, market_lines_df, player_name):
        """
        Compare fair odds with market odds

        Args:
            fair_odds_df: DataFrame with fair odds
            market_lines_df: DataFrame with market lines
            player_name: Name of the player

        Returns:
            DataFrame with comparison
        """
        logger.info(f"Comparing fair odds with market odds for {player_name}")

        if fair_odds_df is None or market_lines_df is None:
            logger.error("Missing odds data for comparison")
            return None

        try:
            # Filter market lines for the player
            player_lines = market_lines_df[market_lines_df['player'] == player_name]

            if player_lines.empty:
                logger.warning(f"No market lines found for {player_name}")
                return None

            results = []

            for _, line in player_lines.iterrows():
                threshold = line['line']

                # Find closest threshold in fair odds
                closest_idx = (fair_odds_df['threshold'] - threshold).abs().idxmin()
                fair_line = fair_odds_df.iloc[closest_idx]

                # Get market odds
                over_market_odds = line.get('over_odds', 0)
                under_market_odds = line.get('under_odds', 0)

                # Calculate edge
                over_edge = self.calculate_edge(fair_line['over_prob'], over_market_odds)
                under_edge = self.calculate_edge(fair_line['under_prob'], under_market_odds)

                # Determine bet recommendation
                over_recommended = over_edge > 3.0  # 3% edge threshold
                under_recommended = under_edge > 3.0

                results.append({
                    'player': player_name,
                    'line': threshold,
                    'over_fair_prob': fair_line['over_prob'],
                    'under_fair_prob': fair_line['under_prob'],
                    'over_fair_american': fair_line['over_fair_american'],
                    'under_fair_american': fair_line['under_fair_american'],
                    'over_market_american': over_market_odds,
                    'under_market_american': under_market_odds,
                    'over_edge': over_edge,
                    'under_edge': under_edge,
                    'over_recommended': over_recommended,
                    'under_recommended': under_recommended
                })

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error comparing with market: {e}")
            return None

    def plot_points_distribution(self, distribution, player_name, market_line=None):
        """
        Plot the points probability distribution

        Args:
            distribution: Dictionary with points distribution
            player_name: Name of the player
            market_line: Market line for over/under

        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting points distribution for {player_name}")

        if distribution is None:
            logger.error("No distribution to plot")
            return None

        try:
            points = distribution['points']
            probs = distribution['probabilities']
            mean = distribution['mean']

            plt.figure(figsize=(10, 6))
            plt.bar(points, probs, width=0.8, alpha=0.7)
            plt.axvline(mean, color='r', linestyle='--', label=f'Predicted mean: {mean:.1f}')

            if market_line is not None:
                plt.axvline(market_line, color='g', linestyle='-', label=f'Market line: {market_line}')

                # Calculate over probability
                over_prob = sum(probs[points > market_line])
                plt.text(market_line + 1, max(probs) * 0.9, f'Over: {over_prob:.1%}',
                         color='g', ha='left', va='top')
                plt.text(market_line - 1, max(probs) * 0.9, f'Under: {1-over_prob:.1%}',
                         color='g', ha='right', va='top')

            plt.title(f'Points Distribution - {player_name}')
            plt.xlabel('Points')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"{player_name.replace(' ', '_')}_distribution_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Saved distribution plot to {plot_path}")
            return plot_path

        except Exception as e:
            logger.error(f"Error plotting distribution: {e}")
            return None

    def kelly_criterion(self, fair_prob, market_odds, bankroll=100, fraction=0.5):
        """
        Calculate Kelly Criterion bet size

        Args:
            fair_prob: Fair probability (0-1)
            market_odds: Market odds (American)
            bankroll: Total bankroll
            fraction: Fraction of Kelly to use (0-1)

        Returns:
            Recommended bet size
        """
        try:
            # Convert to decimal
            market_decimal = self.american_to_decimal(market_odds)

            # Calculate Kelly fraction
            q = 1 - fair_prob
            b = market_decimal - 1

            if b <= 0 or fair_prob <= 0:
                return 0

            kelly = (b * fair_prob - q) / b

            # Apply fraction and bankroll
            if kelly > 0:
                return kelly * fraction * bankroll
            else:
                return 0
        except:
            return 0

    def calculate_ev(self, fair_prob, market_odds):
        """
        Calculate expected value of a bet

        Args:
            fair_prob: Fair probability (0-1)
            market_odds: Market odds (American)

        Returns:
            Expected value percentage
        """
        try:
            market_decimal = self.american_to_decimal(market_odds)
            ev = (fair_prob * (market_decimal - 1) - (1 - fair_prob)) * 100
            return ev
        except:
            return np.nan

    def find_value_bets(self, comparison_df, min_edge=3.0, min_prob=0.05):
        """
        Find valuable betting opportunities

        Args:
            comparison_df: DataFrame with odds comparison
            min_edge: Minimum edge percentage
            min_prob: Minimum fair probability

        Returns:
            DataFrame with value bets
        """
        logger.info("Finding value betting opportunities")

        if comparison_df is None or comparison_df.empty:
            logger.error("No comparison data to find value bets")
            return None

        try:
            # Filter for over bets
            over_value = comparison_df[
                (comparison_df['over_edge'] > min_edge) &
                (comparison_df['over_fair_prob'] > min_prob)
            ].copy()

            if not over_value.empty:
                over_value['bet_type'] = 'OVER'
                over_value['fair_prob'] = over_value['over_fair_prob']
                over_value['market_odds'] = over_value['over_market_american']
                over_value['edge'] = over_value['over_edge']

                # Calculate Kelly bet sizes and EV
                over_value['kelly_bet'] = over_value.apply(
                    lambda x: self.kelly_criterion(x['fair_prob'], x['market_odds']),
                    axis=1
                )
                over_value['ev'] = over_value.apply(
                    lambda x: self.calculate_ev(x['fair_prob'], x['market_odds']),
                    axis=1
                )

            # Filter for under bets
            under_value = comparison_df[
                (comparison_df['under_edge'] > min_edge) &
                (comparison_df['under_fair_prob'] > min_prob)
            ].copy()

            if not under_value.empty:
                under_value['bet_type'] = 'UNDER'
                under_value['fair_prob'] = under_value['under_fair_prob']
                under_value['market_odds'] = under_value['under_market_american']
                under_value['edge'] = under_value['under_edge']

                # Calculate Kelly bet sizes and EV
                under_value['kelly_bet'] = under_value.apply(
                    lambda x: self.kelly_criterion(x['fair_prob'], x['market_odds']),
                    axis=1
                )
                under_value['ev'] = under_value.apply(
                    lambda x: self.calculate_ev(x['fair_prob'], x['market_odds']),
                    axis=1
                )

            # Combine and sort by edge
            if 'over_value' in locals() and not over_value.empty and 'under_value' in locals() and not under_value.empty:
                value_bets = pd.concat([over_value, under_value])
            elif 'over_value' in locals() and not over_value.empty:
                value_bets = over_value
            elif 'under_value' in locals() and not under_value.empty:
                value_bets = under_value
            else:
                logger.warning("No value bets found")
                return pd.DataFrame()

            # Select and sort columns
            columns = ['player', 'line', 'bet_type', 'fair_prob', 'market_odds',
                      'edge', 'ev', 'kelly_bet']
            value_bets = value_bets[columns].sort_values('edge', ascending=False)

            logger.info(f"Found {len(value_bets)} value betting opportunities")
            return value_bets

        except Exception as e:
            logger.error(f"Error finding value bets: {e}")
            return None

    def backtest_strategy(self, test_data, model_data, market_lines, min_edge=3.0, bankroll=10000):
        """
        Backtest a betting strategy using historical data

        Args:
            test_data: DataFrame with historical test data
            model_data: Dictionary with model and metadata
            market_lines: DataFrame with historical market lines
            min_edge: Minimum edge percentage to place a bet
            bankroll: Starting bankroll

        Returns:
            Dictionary with backtest results
        """
        logger.info("Backtesting betting strategy")

        if test_data is None or test_data.empty or model_data is None or market_lines is None or market_lines.empty:
            logger.error("Missing data for backtesting")
            return None

        try:
            results = []
            initial_bankroll = bankroll
            current_bankroll = bankroll
            bets_placed = 0
            bets_won = 0

            unique_games = test_data['game_date'].unique()

            logger.info(f"Backtesting on {len(unique_games)} games")

            for game_date in unique_games:
                # Get games for this date
                date_data = test_data[test_data['game_date'] == game_date]

                for _, player_data in date_data.iterrows():
                    player_name = player_data.get('player_name', f"Player_{player_data.get('player_id', 'Unknown')}")

                    # Get market lines for this player
                    player_lines = market_lines[
                        (market_lines['player'] == player_name) &
                        (market_lines['game_date'] == game_date)
                    ]

                    if player_lines.empty:
                        continue

                    # Generate points distribution
                    distribution = self.predict_points_distribution(model_data, player_data)

                    if distribution is None:
                        continue

                    # Calculate fair odds for the lines
                    thresholds = player_lines['line'].values
                    fair_odds = self.calculate_over_under_probs(distribution, thresholds)

                    if fair_odds is None:
                        continue

                    # Compare with market
                    comparison = self.compare_with_market(fair_odds, player_lines, player_name)

                    if comparison is None:
                        continue

                    # Find value bets
                    value_bets = self.find_value_bets(comparison, min_edge=min_edge)

                    if value_bets is None or value_bets.empty:
                        continue

                    # Simulate bets
                    for _, bet in value_bets.iterrows():
                        # Calculate bet size (half Kelly)
                        bet_size = min(bet['kelly_bet'] * 0.5, current_bankroll * 0.05)

                        # Check if bet should be placed
                        if bet_size < 1:
                            continue

                        # Get actual points
                        actual_points = player_data['pts']

                        # Determine if bet won
                        if bet['bet_type'] == 'OVER':
                            bet_won = actual_points > bet['line']
                        else:  # UNDER
                            bet_won = actual_points < bet['line']

                        # Calculate profit
                        market_odds = bet['market_odds']

                        if bet_won:
                            if market_odds > 0:
                                profit = bet_size * (market_odds / 100)
                            else:
                                profit = bet_size * (100 / abs(market_odds))
                            bets_won += 1
                        else:
                            profit = -bet_size

                        # Update bankroll
                        current_bankroll += profit

                        # Record bet
                        results.append({
                            'game_date': game_date,
                            'player': player_name,
                            'line': bet['line'],
                            'bet_type': bet['bet_type'],
                            'fair_prob': bet['fair_prob'],
                            'market_odds': market_odds,
                            'edge': bet['edge'],
                            'bet_size': bet_size,
                            'actual_points': actual_points,
                            'bet_won': bet_won,
                            'profit': profit,
                            'bankroll': current_bankroll
                        })

                        bets_placed += 1

            # Calculate overall statistics
            if results:
                results_df = pd.DataFrame(results)

                total_profit = current_bankroll - initial_bankroll
                roi = (total_profit / initial_bankroll) * 100
                win_rate = bets_won / bets_placed if bets_placed > 0 else 0

                # Calculate drawdown
                results_df['drawdown'] = initial_bankroll - results_df['bankroll']
                max_drawdown = results_df['drawdown'].max()

                logger.info(f"Backtest complete: {bets_placed} bets, {bets_won} wins ({win_rate:.1%}), ROI: {roi:.1f}%")

                return {
                    'results': results_df,
                    'bets_placed': bets_placed,
                    'bets_won': bets_won,
                    'win_rate': win_rate,
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': current_bankroll,
                    'profit': total_profit,
                    'roi': roi,
                    'max_drawdown': max_drawdown
                }
            else:
                logger.warning("No bets placed in backtest")
                return {
                    'results': pd.DataFrame(),
                    'bets_placed': 0,
                    'bets_won': 0,
                    'win_rate': 0,
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': initial_bankroll,
                    'profit': 0,
                    'roi': 0,
                    'max_drawdown': 0
                }

        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None

    def plot_backtest_results(self, backtest_results):
        """
        Plot backtest results

        Args:
            backtest_results: Dictionary with backtest results

        Returns:
            Path to saved plot
        """
        logger.info("Plotting backtest results")

        if backtest_results is None or 'results' not in backtest_results or backtest_results['results'].empty:
            logger.error("No backtest results to plot")
            return None

        try:
            results_df = backtest_results['results']

            # Plot bankroll over time
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(results_df)), results_df['bankroll'], 'b-')
            plt.axhline(backtest_results['initial_bankroll'], color='r', linestyle='--',
                       label=f"Initial: ${backtest_results['initial_bankroll']}")

            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"backtest_bankroll_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()

            # Plot win rate by edge percentage
            plt.figure(figsize=(12, 6))

            # Create edge bins
            results_df['edge_bin'] = pd.cut(results_df['edge'], bins=range(0, 21, 2), right=False)

            # Calculate win rate by edge bin
            edge_win_rates = results_df.groupby('edge_bin')['bet_won'].agg(['count', 'mean'])
            edge_win_rates = edge_win_rates[edge_win_rates['count'] >= 5]  # Filter bins with few samples

            if not edge_win_rates.empty:
                plt.bar(edge_win_rates.index.astype(str), edge_win_rates['mean'], alpha=0.7)
                plt.axhline(results_df['bet_won'].mean(), color='r', linestyle='--',
                           label=f"Overall: {results_df['bet_won'].mean():.1%}")

                plt.title('Win Rate by Edge Percentage')
                plt.xlabel('Edge Bin (%)')
                plt.ylabel('Win Rate')
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Save plot
                win_rate_path = os.path.join(self.results_dir, f"backtest_winrate_{timestamp}.png")
                plt.savefig(win_rate_path)
                plt.close()

            logger.info(f"Saved backtest plots to {self.results_dir}")
            return plot_path

        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")
            return None

def main():
    """
    Main function to run the odds calculator
    """
    import argparse

    parser = argparse.ArgumentParser(description='Calculate fair betting odds from model predictions')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--data', type=str, help='Path to the test data file')
    parser.add_argument('--odds', type=str, help='Path to the market odds file')
    parser.add_argument('--player', type=str, help='Player name to analyze')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--edge', type=float, default=3.0, help='Minimum edge for value bets')

    args = parser.parse_args()

    # Create odds calculator
    calculator = BettingOddsCalculator()

    # Validate inputs
    if args.model is None:
        print("Error: No model file specified")
        parser.print_help()
        return 1

    if args.data is None:
        print("Error: No test data file specified")
        parser.print_help()
        return 1

    if args.odds is None:
        print("Error: No market odds file specified")
        parser.print_help()
        return 1

    # Load model
    model_data = calculator.load_model(args.model)
    if model_data is None:
        print("Error: Failed to load model")
        return 1

    # Load test data
    try:
        test_data = pd.read_csv(args.data)
        print(f"Loaded test data: {len(test_data)} records")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return 1

    # Load market odds
    try:
        market_odds = pd.read_csv(args.odds)
        print(f"Loaded market odds: {len(market_odds)} records")
    except Exception as e:
        print(f"Error loading market odds: {e}")
        return 1

    # Player analysis
    if args.player:
        player_data = test_data[test_data['player_name'] == args.player]

        if player_data.empty:
            print(f"No data found for player: {args.player}")
            return 1

        # Use most recent data point
        if 'game_date' in player_data.columns:
            player_data = player_data.sort_values('game_date', ascending=False).iloc[0]
        else:
            player_data = player_data.iloc[0]

        # Generate points distribution
        distribution = calculator.predict_points_distribution(model_data, player_data)

        if distribution is None:
            print("Error generating points distribution")
            return 1

        # Find player's market line
        player_lines = market_odds[market_odds['player'] == args.player]

        if not player_lines.empty:
            market_line = player_lines.iloc[0]['line']
            print(f"Market line for {args.player}: {market_line}")
        else:
            market_line = None
            print(f"No market line found for {args.player}")

        # Calculate fair odds
        thresholds = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5]
        if market_line is not None and market_line not in thresholds:
            thresholds.append(market_line)
            thresholds.sort()

        fair_odds = calculator.calculate_over_under_probs(distribution, thresholds)

        if fair_odds is None:
            print("Error calculating fair odds")
            return 1

        # Plot distribution
        calculator.plot_points_distribution(distribution, args.player, market_line)

        # Show fair odds
        print("\nFair odds for different thresholds:")
        print(fair_odds[['threshold', 'over_prob', 'under_prob',
                       'over_fair_american', 'under_fair_american']].to_string(index=False))

        # Compare with market (if available)
        if market_line is not None:
            comparison = calculator.compare_with_market(fair_odds, player_lines, args.player)

            if comparison is not None:
                print("\nComparison with market odds:")
                print(comparison[['line', 'over_fair_prob', 'under_fair_prob',
                                'over_fair_american', 'under_fair_american',
                                'over_market_american', 'under_market_american',
                                'over_edge', 'under_edge']].to_string(index=False))

                # Find value bets
                value_bets = calculator.find_value_bets(comparison, min_edge=args.edge)

                if value_bets is not None and not value_bets.empty:
                    print("\nValue betting opportunities:")
                    print(value_bets.to_string(index=False))
                else:
                    print(f"\nNo value bets found with minimum edge of {args.edge}%")

    # Backtest
    if args.backtest:
        print("\nRunning backtest...")

        # Run backtest
        backtest_results = calculator.backtest_strategy(
            test_data, model_data, market_odds, min_edge=args.edge)

        if backtest_results is None:
            print("Error running backtest")
            return 1

        # Plot results
        calculator.plot_backtest_results(backtest_results)

        # Print summary
        print(f"\nBacktest Summary:")
        print(f"Bets placed: {backtest_results['bets_placed']}")
        print(f"Bets won: {backtest_results['bets_won']} ({backtest_results['win_rate']:.1%})")
        print(f"Starting bankroll: ${backtest_results['initial_bankroll']}")
        print(f"Final bankroll: ${backtest_results['final_bankroll']:.2f}")
        print(f"Profit: ${backtest_results['profit']:.2f} ({backtest_results['roi']:.1f}%)")
        print(f"Maximum drawdown: ${backtest_results['max_drawdown']:.2f}")

        # Save results
        if 'results' in backtest_results and not backtest_results['results'].empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(calculator.results_dir, f"backtest_results_{timestamp}.csv")
            backtest_results['results'].to_csv(results_path, index=False)
            print(f"Saved detailed backtest results to {results_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())