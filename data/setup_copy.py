from setuptools import setup, find_packages

setup(
    name="nba-betting-model",
    version="0.1.0",
    description="NBA Player Points Prediction & Betting Model",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "selenium",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "matplotlib",
        "seaborn",
        "tqdm",
        "joblib",
        "pytest",
    ],
    python_requires=">=3.8",
)