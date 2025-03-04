from setuptools import setup, find_packages

setup(
    name="sport-bet",
    version="0.1.0",
    packages=find_packages(),
    description="NBA Player Points Prediction & Betting Model",
    author="Your Name",
    install_requires=[
        "requests",
        "beautifulsoup4",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
    ],
    python_requires=">=3.8",
)