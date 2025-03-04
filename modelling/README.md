# Phase 2: Modelling

This directory contains scripts for developing and training predictive models for NBA player points.

## Objectives
- Develop accurate predictive models for player point scoring
- Experiment with various model architectures
- Tune hyperparameters for optimal performance
- Evaluate model performance using appropriate metrics
- Create a probabilistic output to support odds calculation

## Implementation Steps

### 1. Baseline Model Development
- Create a script for baseline models (e.g., `baseline_models.py`)
- Implement simple models as baselines:
  - Historical average
  - Moving average
  - Simple linear regression
  - Season average adjusted for opponent
- Evaluate baselines to establish performance benchmarks

### 2. Advanced Model Development
- Create scripts for more sophisticated models:
  - Linear models: `linear_models.py` (Ridge, Lasso, ElasticNet)
  - Tree-based models: `tree_models.py` (Random Forest, XGBoost, LightGBM)
  - Neural networks: `neural_models.py` (if appropriate)
- Implement model classes with consistent interfaces for:
  - Training
  - Prediction
  - Evaluation
  - Serialization/deserialization

### 3. Hyperparameter Tuning
- Create a script for hyperparameter optimization (e.g., `hyperparameter_tuning.py`)
- Implement:
  - Grid search
  - Random search
  - Bayesian optimization
- Tune model-specific parameters for optimal performance
- Document best parameter settings for each model type

### 4. Ensemble Methods
- Create a script for ensemble modeling (e.g., `ensemble_models.py`)
- Explore:
  - Voting ensembles
  - Stacking approaches
  - Model blending
  - Weighted averaging based on historical performance
- Aim to reduce variance and improve prediction stability

### 5. Probabilistic Modeling
- Create a script for generating probabilistic outputs (e.g., `probabilistic_output.py`)
- Implement methods to:
  - Estimate full probability distributions of point outcomes
  - Calculate probabilities for specific over/under thresholds
  - Quantify prediction uncertainty
  - Generate confidence intervals

### 6. Model Evaluation
- Create a script for comprehensive model evaluation (e.g., `model_evaluation.py`)
- Implement evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Classification metrics for over/under prediction
  - Calibration assessment for probabilistic outputs
  - Player-specific performance analysis

## Model Storage
- Save trained models in the `models/` directory
- Use consistent naming conventions:
  - `baseline_model_YYYY_MM_DD.pkl`
  - `linear_model_YYYY_MM_DD.pkl`
  - `xgboost_model_YYYY_MM_DD.pkl`
  - `ensemble_model_YYYY_MM_DD.pkl`
- Store model metadata and performance metrics

## Technical Requirements
- Use scikit-learn for standard ML algorithms
- Use XGBoost/LightGBM for gradient boosting
- Use PyTorch/TensorFlow for neural networks (if needed)
- Implement proper cross-validation strategies
- Ensure reproducibility of training procedures
- Document all modeling decisions and rationales

## Deliverables
- Model implementation scripts
- Trained model files
- Performance evaluation reports
- Model comparison analysis
- Documentation of model architecture and training procedures

## Tips
- Consider player-specific models for stars with sufficient data
- Explore different prediction targets (raw points, z-scores, etc.)
- Be mindful of overfitting, especially with feature-rich datasets
- Consider the temporal nature of basketball data in validation strategy
- Balance model complexity with interpretability
- Experiment with different approaches to handling player rest/injuries