# -*- coding: utf-8 -*-
"""
This script combines the logic from five separate code blocks into a single,
end-to-end executable file. It performs a complete machine learning pipeline:

1.  Generates a synthetic 'data.csv' if one doesn't exist.
2.  Calculates four distinct behavioral scores (component scores).
3.  Trains a unique "base model" for each component score, evaluates it, and
    plots its feature importance.
4.  Trains a "meta-model" using the predictions from the base models as its
    input features (a technique called model stacking).
5.  Evaluates the final stacked meta-model.
6.  Runs a final prediction on a new, single sample of data using the entire
    trained model pipeline to produce a final behavioral score.
"""

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import os
import warnings

# Suppress XGBoost verbosity and other warnings for cleaner output
warnings.filterwarnings("ignore")


# 2. Data Generation Function
def generate_synthetic_data(num_rows=1000, file_path="farmers_data.csv"):
    """
    Generates a synthetic 'data.csv' file for the script to run.
    This is necessary because the original data file is not provided.
    """
    if os.path.exists(file_path):
        print(f"'{file_path}' already exists. Skipping generation.")
        return

    print(f"Generating synthetic '{file_path}'...")
    data = {
        # Digital Trust Features
        "reliable_contacts_ratio": np.random.uniform(0.5, 1.0, num_rows),
        "proximity_to_defaulters_score": np.random.uniform(0, 1, num_rows),
        "digital_network_engagement_score": np.random.uniform(0, 1, num_rows),
        "social_connections_count": np.random.randint(5, 50, num_rows),
        "support_request_frequency": np.random.randint(0, 10, num_rows),
        # Resilience Features
        "time_to_resume_upi_after_shock": np.random.randint(0, 30, num_rows),
        "emi_status_last_12_months": np.random.randint(0, 5, num_rows),
        "overdraft_usage_frequency": np.random.randint(0, 10, num_rows),
        "loan_repayment_ratio": np.random.uniform(0.7, 1.0, num_rows),
        "yield_recovery_ratio": np.random.uniform(0.5, 1.2, num_rows),
        # Adaptability Features
        "income_volatility_score": np.random.uniform(0, 1, num_rows),
        "budgeting_habit_score": np.random.uniform(0, 1, num_rows),
        "agritech_tool_usage": np.random.randint(0, 2, num_rows),
        "new_crop_adoption_flag": np.random.randint(0, 2, num_rows),
        # Language Sentiment Features
        "financial_optimism_index": np.random.uniform(0, 1, num_rows),
        "stress_signal_intensity": np.random.uniform(0, 1, num_rows),
        "in_cooperative": np.random.randint(0, 2, num_rows),
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✅ '{file_path}' created successfully.")


# 3. Main Execution Function
def main():
    """
    Main function to run the entire model training and prediction pipeline.
    """
    # === Part 1: Load Data and Engineer Component Scores ===
    print("\n--- Part 1: Loading Data and Engineering Scores ---")
    data_path = "farmers_data.csv"
    generate_synthetic_data(file_path=data_path)
    df = pd.read_csv(data_path)

    # Calculate all four component scores
    df["digital_trust_graph_score"] = (
        df["reliable_contacts_ratio"] * 50 - df["proximity_to_defaulters_score"] * 0.4 +
        df["digital_network_engagement_score"] * 0.3 + df["social_connections_count"] * 0.1 +
        df["financial_optimism_index"] * 10 - df["stress_signal_intensity"] * 5 -
        df["support_request_frequency"] * 0.2
    )
    df["resilience_recovery_score"] = (
        100 - df["time_to_resume_upi_after_shock"] * 0.3 - df["emi_status_last_12_months"] * 2 -
        df["overdraft_usage_frequency"] * 1.5 + df["loan_repayment_ratio"] * 30 +
        df["yield_recovery_ratio"] * 20
    )
    df["adaptability_score"] = (
        -df["income_volatility_score"] * 0.3 + df["yield_recovery_ratio"] * 40 +
        df["budgeting_habit_score"] * 15 + df["agritech_tool_usage"] * 10 +
        df["loan_repayment_ratio"] * 20 + df["new_crop_adoption_flag"] * 5
    )
    df["language_sentiment_score"] = (
        df["financial_optimism_index"] * 30 - df["stress_signal_intensity"] * 25 +
        df["budgeting_habit_score"] * 20 + df["in_cooperative"] * 10 +
        df["agritech_tool_usage"] * 15
    )
    print("✅ All four component scores calculated.")

    # === Part 2: Train Base Models for Each Component ===
    print("\n--- Part 2: Training Base Models ---")

    # Define feature sets and targets for each base model
    model_configs = {
        "digital_trust": {
            "features": ["reliable_contacts_ratio", "proximity_to_defaulters_score", "digital_network_engagement_score", "social_connections_count", "financial_optimism_index", "stress_signal_intensity", "support_request_frequency"],
            "target": "digital_trust_graph_score",
            "model_type": "Lasso"
        },
        "resilience": {
            "features": ["time_to_resume_upi_after_shock", "emi_status_last_12_months", "overdraft_usage_frequency", "loan_repayment_ratio", "yield_recovery_ratio"],
            "target": "resilience_recovery_score",
            "model_type": "XGBoost_Grid"
        },
        "adaptability": {
            "features": ["income_volatility_score", "yield_recovery_ratio", "budgeting_habit_score", "agritech_tool_usage", "loan_repayment_ratio", "new_crop_adoption_flag"],
            "target": "adaptability_score",
            "model_type": "XGBoost_Fixed"
        },
        "language_sentiment": {
            "features": ["financial_optimism_index", "stress_signal_intensity", "budgeting_habit_score", "in_cooperative", "agritech_tool_usage"],
            "target": "language_sentiment_score",
            "model_type": "Lasso"
        }
    }

    base_models = {}
    for name, config in model_configs.items():
        print(f"\n--- Training Model for: {name.replace('_', ' ').title()} ---")
        X = df[config["features"]]
        y = df[config["target"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        if config["model_type"] == "Lasso":
            model = Lasso(alpha=0.01)
            model.fit(X_train, y_train)
        elif config["model_type"] == "XGBoost_Grid":
            xgb = XGBRegressor(random_state=42, verbosity=0)
            param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
            grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring=make_scorer(r2_score), n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"🔩 Best Params: {grid_search.best_params_}")
        elif config["model_type"] == "XGBoost_Fixed":
            best_params = {'n_estimators': 387, 'max_depth': 3, 'learning_rate': 0.065, 'subsample': 0.628}
            model = XGBRegressor(**best_params, random_state=42)
            model.fit(X_train, y_train)

        base_models[name] = model
        
        # Evaluate and plot
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"📊 Evaluation -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

#        plt.figure(figsize=(8, 4))
 #       if config["model_type"] == "Lasso":
  #          coeffs = pd.Series(model.coef_, index=config["features"]).sort_values()
   #         sns.barplot(x=coeffs.values, y=coeffs.index)
    #        plt.title(f"Feature Importance (Coefficients) - {name.replace('_', ' ').title()}")
     #   else: # XGBoost
      #      importances = pd.Series(model.feature_importances_, index=config["features"]).sort_values(ascending=False)
       #     sns.barplot(x=importances.values, y=importances.index)
        #    plt.title(f"Feature Importance - {name.replace('_', ' ').title()}")
        #plt.tight_layout()
        #plt.show()

    print("\n✅ All base models trained.")

    # === Part 3: Train Meta-Model (Stacking) ===
    print("\n--- Part 3: Training Meta-Model ---")

    # Create meta-features by getting predictions from base models on the full dataset
    meta_features_df = pd.DataFrame(index=df.index)
    for name, model in base_models.items():
        features = model_configs[name]["features"]
        meta_features_df[f"{name}_pred"] = model.predict(df[features])

    # Define the final target score (weighted average of component scores)
    df["final_behavioral_score"] = (
        df["digital_trust_graph_score"] * 0.25 + df["resilience_recovery_score"] * 0.30 +
        df["adaptability_score"] * 0.25 + df["language_sentiment_score"] * 0.20
    )
    
    X_meta = meta_features_df
    y_meta = df["final_behavioral_score"]

    # Train the meta-model
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_meta_train)
    print("✅ Meta-model trained.")

    # Evaluate the final meta-model
    y_meta_pred = meta_model.predict(X_meta_test)
    rmse_meta = np.sqrt(mean_squared_error(y_meta_test, y_meta_pred))
    r2_meta = r2_score(y_meta_test, y_meta_pred)
    print(f"\n🎯 Final Meta-Model Evaluation -> RMSE: {rmse_meta:.4f}, R² Score: {r2_meta:.4f}")
    print("\n🔍 Meta-Model Coefficients (Contribution of each base model):")
    for name, coef in zip(X_meta.columns, meta_model.coef_):
        print(f"  - {name}: {coef:.4f}")

    # === Part 4: Predict on a New Sample Input ===
    print("\n--- Part 4: Running Prediction on a New Sample ---")
    sample_input = pd.DataFrame({
        "social_connections_count": [15], "proximity_to_defaulters_score": [0.2],
        "reliable_contacts_ratio": [0.85], "digital_network_engagement_score": [0.6],
        "support_request_frequency": [2], "financial_optimism_index": [0.75],
        "stress_signal_intensity": [0.3], "time_to_resume_upi_after_shock": [3],
        "emi_status_last_12_months": [1], "overdraft_usage_frequency": [2],
        "loan_repayment_ratio": [0.9], "yield_recovery_ratio": [0.7],
        "income_volatility_score": [0.4], "budgeting_habit_score": [0.8],
        "agritech_tool_usage": [1], "new_crop_adoption_flag": [1], "in_cooperative": [1]
    })

    # 1. Get predictions from base models for the sample
    sample_meta_features = {}
    for name, model in base_models.items():
        features = model_configs[name]["features"]
        prediction = model.predict(sample_input[features])
        sample_meta_features[f"{name}_pred"] = prediction

    sample_meta_df = pd.DataFrame(sample_meta_features)

    # 2. Use the meta-model to get the final prediction
    final_prediction = meta_model.predict(sample_meta_df)

    print(f"\n🔮 Final Predicted Behavioral Score for the sample: {final_prediction[0]:.4f}")


# 4. Script Entry Point
if __name__ == "__main__":
    main()
