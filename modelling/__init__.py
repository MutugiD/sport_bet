# NBA Betting Model - Modelling Module
"""
This module contains machine learning models for NBA player points prediction.
"""

from modelling.baseline_models import NBABaselineModeler

# Try to import advanced models, but don't fail if dependencies are missing
try:
    from modelling.advanced_models import NBAAdvancedModeler
    __all__ = [
        'NBABaselineModeler',
        'NBAAdvancedModeler'
    ]
except ImportError:
    # Advanced models dependencies not available
    __all__ = [
        'NBABaselineModeler'
    ]