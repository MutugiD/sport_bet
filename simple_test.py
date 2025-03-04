import os
import pandas as pd

print("Starting simple test...")

# Check if data files exist
test_data_path = 'data/processed/test_sample.csv'
market_odds_path = 'data/raw/sample_odds.csv'

print(f"Checking if data files exist:")
print(f"- {test_data_path}: {os.path.exists(test_data_path)}")
print(f"- {market_odds_path}: {os.path.exists(market_odds_path)}")

# Try to load data
if os.path.exists(test_data_path) and os.path.exists(market_odds_path):
    print("\nLoading data...")
    test_data = pd.read_csv(test_data_path)
    market_odds = pd.read_csv(market_odds_path)

    print(f"\nTest data shape: {test_data.shape}")
    print(f"Test data columns: {test_data.columns.tolist()}")
    print(f"\nMarket odds shape: {market_odds.shape}")
    print(f"Market odds columns: {market_odds.columns.tolist()}")

    # Check for specific player
    player_name = "Nikola Jokic"
    player_data = test_data[test_data['player_name'] == player_name]
    player_odds = market_odds[market_odds['player'] == player_name]

    print(f"\nData for {player_name}:")
    print(f"- Test data records: {len(player_data)}")
    print(f"- Market odds records: {len(player_odds)}")

    if not player_data.empty:
        print(f"\nPoints scored by {player_name}: {player_data.iloc[0]['pts']}")

    if not player_odds.empty:
        print(f"Point line for {player_name}: {player_odds.iloc[0]['line']}")
        print(f"Over odds: {player_odds.iloc[0]['over_odds']}, Under odds: {player_odds.iloc[0]['under_odds']}")

print("\nSimple test completed.")