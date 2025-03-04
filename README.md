# NBA Player Points Prediction & Betting Model

A comprehensive system for predicting NBA player points and identifying value in betting markets.

## Overview

This project implements an end-to-end pipeline for NBA player points prediction and betting analysis:

1. **Data Collection** - Scraping player game logs, team stats, and odds data
2. **Data Cleaning** - Processing raw data into clean, analysis-ready datasets
3. **Feature Engineering** - Creating predictive features from historical performance
4. **Model Building** - Training both baseline and advanced prediction models
5. **Betting Analysis** - Calculating fair odds and identifying value opportunities
6. **Evaluation** - Backtesting strategies and measuring performance

## System Architecture

The system is organized into modular components that handle different aspects of the data pipeline:

```
sport-bet/
├── data/
│   ├── raw/                 # Raw scraped data
│   ├── processed/           # Cleaned and processed data
│   └── results/             # Betting analysis results
├── data_collection/         # Scrapers for game logs, team stats, odds
├── data_processing/         # Cleaning and feature engineering
├── modelling/               # Prediction models (baseline and advanced)
├── evaluation/              # Odds calculation and strategy backtesting
├── models/                  # Saved trained models
└── scripts/                 # Utility scripts
```

## Key Components

### Data Collection

- `player_scraper.py` - Scrapes player game logs from Basketball-Reference
- `team_scraper.py` - Scrapes team statistics from Basketball-Reference
- `odds_scraper.py` - Mock implementation of odds data collection
- `data_collector.py` - Orchestrates all data collection tasks

### Data Processing

- `data_cleaner.py` - Cleans and standardizes raw data
- `feature_engineering.py` - Creates predictive features from player and team data
- `validate_pipeline.py` - Validates data pipeline integrity

### Modelling

- `baseline_models.py` - Simple prediction models (averages, regressions)
- `advanced_models.py` - More complex models (gradient boosting, neural networks)

### Evaluation

- `odds_calculator.py` - Converts predictions to fair odds and compares with market
- `evaluate_pipeline.py` - End-to-end pipeline validation

## Features

- Historical and rolling average player performance metrics
- Team offensive and defensive metrics
- Player vs specific opponent metrics
- Rest days and home/away impact analysis
- Multiple modeling approaches (from simple to advanced)
- Kelly criterion bet sizing
- Edge and ROI analysis
- Comprehensive backtesting framework

## Models Implemented

### Baseline Models

- Historical Average Model
- Rolling Average Model
- Season Average with Opponent Adjustment
- Simple Linear Regression
- Ridge Regression

### Advanced Models

- Gradient Boosting Regression
- Random Forest Regression
- XGBoost Regression
- LightGBM Regression
- Neural Network Regression

## Usage

There are two ways to run the commands in this project:

### Option 1: Using Python Module Format (Recommended)

Run commands from the project root using the `-m` flag to properly handle imports:

#### 1. Data Collection

```bash
python -m scraping.data_collector --players "Nikola Jokic" "LeBron James" --seasons "2023"
```

#### 2. Data Cleaning

```bash
python -m data_processing.data_cleaner
```

#### 3. Feature Engineering

```bash
python -m data_processing.feature_engineering
```

#### 4. Model Training

```bash
python -m modelling.baseline_models
python -m modelling.advanced_models
```

#### 5. Betting Analysis

```bash
python -m evaluation.odds_calculator --model models/advanced_model_TIMESTAMP.pkl --data data/processed/test_dataset_TIMESTAMP.csv --odds data/raw/odds_data.csv --player "Nikola Jokic"
```

#### 6. Full Pipeline Validation

```bash
python -m evaluation.evaluate_pipeline
```

### Option 2: Running Scripts Directly

Alternatively, if in the project root, you can run the following:

#### 1. Data Collection

```bash
cd scraping
python data_collector.py --players "Nikola Jokic" "LeBron James" --seasons "2023"
```

#### 2. Data Cleaning

```bash
cd data_processing
python data_cleaner.py
```

#### 3. Feature Engineering

```bash
cd data_processing
python feature_engineering.py
```

#### 4. Model Training

```bash
cd modelling
python baseline_models.py
python advanced_models.py
```

#### 5. Betting Analysis

```bash
cd evaluation
python odds_calculator.py --model ../models/advanced_model_TIMESTAMP.pkl --data ../data/processed/test_dataset_TIMESTAMP.csv --odds ../data/raw/odds_data.csv --player "Nikola Jokic"
```

#### 6. Full Pipeline Validation

```bash
cd evaluation
python evaluate_pipeline.py
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- LightGBM
- XGBoost
- TensorFlow (optional, for neural networks)
- matplotlib
- seaborn
- requests
- BeautifulSoup4

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

For the best experience, install the project as a development package:

```bash
pip install -e .
```

This allows Python to find all the modules regardless of which directory you're in.

## Methodology

### Prediction Approach

1. **Data Collection**: Gather historical player performance data, team statistics, and betting odds.
2. **Feature Engineering**: Create features based on recent performance, matchup history, and situational factors.
3. **Model Training**: Train multiple prediction models on historical data.
4. **Probability Distribution**: Generate a probability distribution of possible point totals.
5. **Fair Odds Calculation**: Convert probability distributions to fair betting odds.
6. **Edge Detection**: Compare fair odds to market odds to identify value opportunities.

### Betting Strategy

The system implements a Kelly criterion approach to bet sizing, which:

1. Calculates the optimal bet size based on the edge and odds
2. Adjusts for risk using fractional Kelly (typically 0.5x)
3. Limits exposure to a percentage of bankroll per bet

## Performance Evaluation

The system evaluates betting performance through:

- Win rate analysis
- Return on investment (ROI)
- Bankroll growth curves
- Maximum drawdown analysis
- Edge vs. actual performance correlation

## Future Improvements

- Incorporate player injuries and rest data
- Add lineup-based analysis
- Real-time odds feeds integration
- Expand to additional player props and markets
- Time series modeling improvements
- Web application interface

## Disclaimer

This project is for educational purposes only. Sports betting involves risk, and past performance is not indicative of future results.

## License

MIT