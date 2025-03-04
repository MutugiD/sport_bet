# NBA Betting Model - Evaluation Module
"""
This module contains tools for evaluating predictions and betting strategies.
"""

from evaluation.odds_calculator import BettingOddsCalculator
from evaluation.evaluate_pipeline import PipelineEvaluator

__all__ = [
    'BettingOddsCalculator',
    'PipelineEvaluator'
]