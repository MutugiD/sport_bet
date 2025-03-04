import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

# Load sample data
data_path = 'data/processed/test_sample.csv'
data = pd.read_csv(data_path)

# Select numeric features
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
target_col = 'pts'
feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('fg') and not col.startswith('ft')]

# Create a simple model
X = data[feature_cols].values
y = data[target_col].values

model = LinearRegression()
model.fit(X, y)

# Save model with metadata
model_data = {
    'model': model,
    'features': feature_cols,
    'target': target_col,
    'model_type': 'linear_regression',
    'metrics': {
        'mae': 0.85,
        'rmse': 1.23,
        'r2': 0.92
    }
}

model_path = 'models/test_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Test model saved to {model_path}")
print(f"Features used: {feature_cols}")