import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

sns.set_style("darkgrid")

DATA_PATH = "farmers.csv"

# === 1. Load dataset ===
df = pd.read_csv(DATA_PATH)

# === 2. Compute component scores ===
df["digital_trust_graph_score"] = (
    df["reliable_contacts_ratio"] * 50
    - df["proximity_to_defaulters_score"] * 0.4
    + df["digital_network_engagement_score"] * 0.3
    + df["social_connections_count"] * 0.1
    + df["financial_optimism_index"] * 10
    - df["stress_signal_intensity"] * 5
    - df["support_request_frequency"] * 0.2
)

df["resilience_recovery_score"] = (
    100
    - df["time_to_resume_upi_after_shock"] * 0.3
    - df["emi_status_last_12_months"] * 2
    - df["overdraft_usage_frequency"] * 1.5
    + df["loan_repayment_ratio"] * 30
    + df["yield_recovery_ratio"] * 20
)

df["adaptability_score"] = (
    - df["income_volatility_score"] * 0.3
    + df["yield_recovery_ratio"] * 40
    + df["budgeting_habit_score"] * 15
    + df["agritech_tool_usage"] * 10
    + df["loan_repayment_ratio"] * 20
    + df["new_crop_adoption_flag"] * 5
)

df["language_sentiment_score"] = (
    df["financial_optimism_index"] * 30
    - df["stress_signal_intensity"] * 25
    + df["budgeting_habit_score"] * 20
    + df["in_cooperative"] * 10
    + df["agritech_tool_usage"] * 15
)

# === 3. Define features per component ===
features_map = {
    "digital_trust": [
        "social_connections_count",
        "proximity_to_defaulters_score",
        "reliable_contacts_ratio",
        "digital_network_engagement_score",
        "support_request_frequency",
        "financial_optimism_index",
        "stress_signal_intensity"
    ],
    "resilience": [
        "time_to_resume_upi_after_shock",
        "emi_status_last_12_months",
        "overdraft_usage_frequency",
        "loan_repayment_ratio",
        "yield_recovery_ratio"
    ],
    "adaptability": [
        "income_volatility_score",
        "yield_recovery_ratio",
        "budgeting_habit_score",
        "agritech_tool_usage",
        "loan_repayment_ratio",
        "new_crop_adoption_flag"
    ],
    "language_sentiment": [
        "financial_optimism_index",
        "stress_signal_intensity",
        "budgeting_habit_score",
        "in_cooperative",
        "agritech_tool_usage"
    ]
}

targets = {
    "digital_trust": "digital_trust_graph_score",
    "resilience": "resilience_recovery_score",
    "adaptability": "adaptability_score",
    "language_sentiment": "language_sentiment_score"
}

# === 4. Training functions ===
def train_xgb_with_grid(X, y):
    model = XGBRegressor(random_state=42, verbosity=0)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0]
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring=make_scorer(r2_score), n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

def train_lasso(X, y, alpha=0.01):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model

# === 5. Train component models ===
models = {}
for name, feats in features_map.items():
    X = df[feats]
    y = df[targets[name]]
    
    if name in ["digital_trust", "resilience", "adaptability"]:
        model = train_xgb_with_grid(X, y)
    else:  # language_sentiment
        model = train_lasso(X, y)
    
    models[name] = model
    y_pred = model.predict(X)
    print(f"\n📊 {name.upper()} model R²: {r2_score(y, y_pred):.4f}")

# === 6. Meta-model training ===
df["final_behavioral_score"] = (
    df["digital_trust_graph_score"] * 0.25 +
    df["resilience_recovery_score"] * 0.3 +
    df["adaptability_score"] * 0.25 +
    df["language_sentiment_score"] * 0.2
)

meta_X = pd.DataFrame({f"{name}_pred": models[name].predict(df[features_map[name]]) for name in models})
meta_y = df["final_behavioral_score"]

meta_X_train, meta_X_test, meta_y_train, meta_y_test = train_test_split(meta_X, meta_y, test_size=0.2, random_state=42)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, meta_y_train)

meta_pred = meta_model.predict(meta_X_test)
print("\n🎯 Meta-model Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(meta_y_test, meta_pred)):.4f}")
print(f"R²:   {r2_score(meta_y_test, meta_pred):.4f}")

# === 7. Take user input for all features ===
print("\n=== Enter User Data for Prediction ===")
user_data = {}
for feat in sorted(set(f for feats in features_map.values() for f in feats)):
    val = input(f"Enter {feat}: ")
    user_data[feat] = float(val)

# === 8. Predict component scores ===
user_df = pd.DataFrame([user_data])
base_preds = {f"{name}_pred": models[name].predict(user_df[features_map[name]])[0] for name in models}

# === 9. Predict final score ===
final_score = meta_model.predict(pd.DataFrame([base_preds]))[0]

print("\n=== Prediction Results ===")
for name, score in base_preds.items():
    print(f"{name}: {score:.2f}")
print(f"\nFinal AltiCred Score: {final_score:.2f}")