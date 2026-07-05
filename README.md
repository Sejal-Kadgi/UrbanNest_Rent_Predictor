# 🏠 UrbanNest Analytics — Dynamic Rent Prediction Engine

**🔗 Live Demo:** https://Sejal-Kadgi-urbannest-rent-predictor.hf.space

## Overview
An end-to-end MLOps project that predicts monthly house rent for Mumbai, Pune, Delhi, and Hisar using a Random Forest model. Built to demonstrate the full ML development lifecycle — from data preprocessing and hyperparameter optimization to containerized deployment.

## Tools & Technologies
- **Modeling:** scikit-learn (RandomForestRegressor)
- **Hyperparameter Optimization:** Grid Search, Random Search, Bayesian Optimization (Optuna)
- **Experiment Tracking:** TrackIO
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Deployment:** Hugging Face Spaces

## What I Built
- Compared three hyperparameter optimization strategies (Grid Search, Random Search, Bayesian Optimization) on a fixed 60-trial budget using 5-fold cross-validation
- Tracked all experiments with TrackIO to compare CV MAE, runtime, and best parameters across methods
- Built an interactive Streamlit dashboard for real estate agents to get instant rent predictions
- Containerized the app with Docker for reproducible deployment
- Deployed publicly on Hugging Face Spaces via a Docker Space template

## Key Findings
- Bayesian Optimization (Optuna/TPE) converged to a good solution faster than Grid or Random Search
- SecurityDeposit and Size_ft² were the strongest predictors of rent
- BHK showed zero importance due to being constant in the dataset — identified via feature importance analysis

## Project Structure
\```
├── app.py              # Streamlit frontend
├── Dockerfile          # Container definition
├── requirements.txt    # Dependencies
├── train.ipynb         # Training, tuning, and evaluation notebook
├── Dataset/            # Training and test data
├── plots/              # Optimization comparison plots
└── screenshots/        # Experiment tracking and deployment screenshots
\```