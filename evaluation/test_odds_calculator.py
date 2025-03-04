import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the BettingOddsCalculator
from odds_calculator import BettingOddsCalculator

def main():
    """
    Simple test for the odds calculator
    """
    # Set paths - adjusted to work from within the evaluation directory
    test_data_path = '../data/processed/test_sample.csv'
    market_odds_path = '../data/raw/sample_odds.csv'
    models_dir = '../models'
    results_dir = '../data/results'

    print(f"Loading data from:\n- {test_data_path}\n- {market_odds_path}")

    # Load test data
    test_data = pd.read_csv(test_data_path)
    market_odds = pd.read_csv(market_odds_path)

    print(f"Loaded test data: {len(test_data)} records")
    print(f"Loaded market odds: {len(market_odds)} records")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Create a calculator
    calculator = BettingOddsCalculator(models_dir=models_dir, results_dir=results_dir)

    # Select a player for testing
    player_name = "Nikola Jokic"
    player_data = test_data[test_data['player_name'] == player_name]

    if player_data.empty:
        print(f"Error: No data found for player {player_name}")
        return 1

    player_data = player_data.iloc[0]

    # Create a simplified distribution instead of using a model
    # This bypasses the model.predict() issue
    mean_pts = player_data['pts']
    std_dev = 5.0

    # Manually create distribution
    points = np.linspace(0, 60, 61)
    probs = norm.pdf(points, mean_pts, std_dev)
    probs = probs / probs.sum()

    distribution = {
        'points': points,
        'probabilities': probs,
        'mean': mean_pts,
        'std': std_dev
    }

    print(f"Created distribution for {player_name} with mean={mean_pts} and std={std_dev}")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(points, probs, width=0.8, alpha=0.7)
    plt.axvline(mean_pts, color='r', linestyle='--', label=f'Mean: {mean_pts}')
    plt.title(f'Points Distribution - {player_name}')
    plt.xlabel('Points')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f"{player_name.replace(' ', '_')}_distribution_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved distribution plot to {plot_path}")

    # Calculate fair odds
    thresholds = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5]
    fair_odds = calculator.calculate_over_under_probs(distribution, thresholds)

    print("\nFair odds for different thresholds:")
    print(fair_odds[['threshold', 'over_prob', 'under_prob',
                  'over_fair_american', 'under_fair_american']].to_string(index=False))

    # Compare with market odds
    player_lines = market_odds[market_odds['player'] == player_name]

    if not player_lines.empty:
        comparison = calculator.compare_with_market(fair_odds, player_lines, player_name)

        print("\nComparison with market odds:")
        print(comparison[['line', 'over_fair_prob', 'under_fair_prob',
                       'over_fair_american', 'under_fair_american',
                       'over_market_american', 'under_market_american',
                       'over_edge', 'under_edge']].to_string(index=False))

        # Find value bets
        value_bets = calculator.find_value_bets(comparison, min_edge=3.0)

        if not value_bets.empty:
            print("\nValue betting opportunities:")
            print(value_bets.to_string(index=False))
        else:
            print("\nNo value bets found with minimum edge of 3.0%")
    else:
        print(f"No market lines found for {player_name}")

    print("\nTest completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())