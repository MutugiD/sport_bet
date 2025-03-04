import os
import sys
import glob
import pandas as pd
import pickle
from datetime import datetime

from evaluation.odds_calculator import BettingOddsCalculator

def main():
    print("Starting odds calculator test...")

    # Use the test model that exists
    model_path = "models/test_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    test_data_path = "data/processed/test_sample.csv"
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        return

    market_odds_path = "data/raw/sample_odds.csv"
    if not os.path.exists(market_odds_path):
        print(f"Market odds file not found at {market_odds_path}")
        return

    print(f"Using model: {model_path}")
    print(f"Using test data: {test_data_path}")
    print(f"Using market odds: {market_odds_path}")

    # Load model and data
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        test_data = pd.read_csv(test_data_path)
        market_odds = pd.read_csv(market_odds_path)

        print(f"\nTest data shape: {test_data.shape}")
        print(f"Market odds shape: {market_odds.shape}")
    except Exception as e:
        print(f"Error loading data or model: {e}")
        return

    # Initialize the odds calculator
    calculator = BettingOddsCalculator(models_dir="models", results_dir="data/results")

    # Test with a specific player
    player_name = "Nikola Jokic"
    player_data = test_data[test_data['player_name'] == player_name]
    player_odds = market_odds[market_odds['player'] == player_name]

    if player_data.empty:
        print(f"No data found for {player_name}")
        return

    print(f"\n--- Analysis for {player_name} ---")

    # Get player features
    player_features = player_data.iloc[0].to_dict()

    # Generate probability distribution
    # Note: Since we're not using the model loading functionality of the calculator,
    # we're passing the model directly to the generate_probability_distribution method
    dist = calculator.generate_probability_distribution(player_features, model=model)

    print(f"Model prediction (mean): {dist['mean']:.2f}")
    print(f"Model prediction (std): {dist['std']:.2f}")

    # Calculate fair odds for different point thresholds
    point_thresholds = [15.5, 17.5, 20.5, 22.5, 25.5, 26.5, 27.5, 30.5]

    print("\nFair odds for different point thresholds:")
    print(f"{'Threshold':<10} {'Over %':<10} {'Over (Am)':<10} {'Under %':<10} {'Under (Am)':<10}")
    print("-" * 60)

    for threshold in point_thresholds:
        over_prob, over_odds, under_prob, under_odds = calculator.calculate_over_under_odds(dist, threshold)

        print(f"{threshold:<10.1f} {over_prob*100:<10.2f} {over_odds:<10.0f} {under_prob*100:<10.2f} {under_odds:<10.0f}")

    # Compare with market odds
    if not player_odds.empty:
        market_line = player_odds.iloc[0]['line']
        market_over_odds = player_odds.iloc[0]['over_odds']
        market_under_odds = player_odds.iloc[0]['under_odds']

        over_prob, fair_over_odds, under_prob, fair_under_odds = calculator.calculate_over_under_odds(dist, market_line)

        print(f"\nMarket comparison for {market_line} points:")
        print(f"{'Type':<8} {'Market':<10} {'Fair':<10} {'Edge':<10}")
        print("-" * 40)

        # Calculate edge for over
        market_over_decimal = american_to_decimal(market_over_odds)
        fair_over_decimal = american_to_decimal(fair_over_odds)
        over_edge = (fair_over_decimal / market_over_decimal - 1) * 100 if market_over_decimal > 1 else 0

        # Calculate edge for under
        market_under_decimal = american_to_decimal(market_under_odds)
        fair_under_decimal = american_to_decimal(fair_under_odds)
        under_edge = (fair_under_decimal / market_under_decimal - 1) * 100 if market_under_decimal > 1 else 0

        print(f"Over    {market_over_odds:<10.0f} {fair_over_odds:<10.0f} {over_edge:<10.2f}%")
        print(f"Under   {market_under_odds:<10.0f} {fair_under_odds:<10.0f} {under_edge:<10.2f}%")

        # Identify value bets
        edge_threshold = 3.0  # 3% edge threshold

        print("\nValue bets:")
        if over_edge >= edge_threshold:
            print(f"✅ Over {market_line} @ {market_over_odds} has {over_edge:.2f}% edge")
        elif under_edge >= edge_threshold:
            print(f"✅ Under {market_line} @ {market_under_odds} has {under_edge:.2f}% edge")
        else:
            print("No value bets found for this line.")

    print("\nOdds calculator test completed.")

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

if __name__ == "__main__":
    main()
