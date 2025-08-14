import os
from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "farmers_data.csv"
MODELS_PATH = BASE_DIR / "models_joblib.pkl"
STATIC_IMG_DIR = BASE_DIR / "static" / "images"
STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

# -------------------------
# Synthetic data generator (kept for compatibility)
# -------------------------
def generate_synthetic_data(num_rows=1000, file_path=DATA_PATH):
    if file_path.exists():
        return
    rng = np.random.default_rng(42)
    data = {
        "reliable_contacts_ratio": rng.uniform(0.5, 1.0, num_rows),
        "proximity_to_defaulters_score": rng.uniform(0, 1, num_rows),
        "digital_network_engagement_score": rng.uniform(0, 1, num_rows),
        "social_connections_count": rng.integers(5, 50, num_rows),
        "support_request_frequency": rng.integers(0, 10, num_rows),
        "time_to_resume_upi_after_shock": rng.integers(0, 30, num_rows),
        "emi_status_last_12_months": rng.integers(0, 5, num_rows),
        "overdraft_usage_frequency": rng.integers(0, 10, num_rows),
        "loan_repayment_ratio": rng.uniform(0.7, 1.0, num_rows),
        "yield_recovery_ratio": rng.uniform(0.5, 1.2, num_rows),
        "income_volatility_score": rng.uniform(0, 1, num_rows),
        "budgeting_habit_score": rng.uniform(0, 1, num_rows),
        "agritech_tool_usage": rng.integers(0, 2, num_rows),
        "new_crop_adoption_flag": rng.integers(0, 2, num_rows),
        "financial_optimism_index": rng.uniform(0, 1, num_rows),
        "stress_signal_intensity": rng.uniform(0, 1, num_rows),
        "in_cooperative": rng.integers(0, 2, num_rows),
    }
    pd.DataFrame(data).to_csv(file_path, index=False)
    print(f"Created synthetic data at {file_path}")

# -------------------------
# Feature engineering
# -------------------------
def add_component_scores(df):
    df = df.copy()
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
        -df["income_volatility_score"] * 0.3
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
    return df

# -------------------------
# Model feature map
# -------------------------
def model_feature_map():
    return {
        "digital_trust": [
            "reliable_contacts_ratio",
            "proximity_to_defaulters_score",
            "digital_network_engagement_score",
            "social_connections_count",
            "financial_optimism_index",
            "stress_signal_intensity",
            "support_request_frequency",
        ],
        "resilience": [
            "time_to_resume_upi_after_shock",
            "emi_status_last_12_months",
            "overdraft_usage_frequency",
            "loan_repayment_ratio",
            "yield_recovery_ratio",
        ],
        "adaptability": [
            "income_volatility_score",
            "yield_recovery_ratio",
            "budgeting_habit_score",
            "agritech_tool_usage",
            "loan_repayment_ratio",
            "new_crop_adoption_flag",
        ],
        "language_sentiment": [
            "financial_optimism_index",
            "stress_signal_intensity",
            "budgeting_habit_score",
            "in_cooperative",
            "agritech_tool_usage",
        ],
    }

model_configs = {
    "digital_trust": {"target": "digital_trust_graph_score", "model_type": "Lasso"},
    "resilience": {"target": "resilience_recovery_score", "model_type": "XGBoost_Grid"},
    "adaptability": {"target": "adaptability_score", "model_type": "XGBoost_Fixed"},
    "language_sentiment": {"target": "language_sentiment_score", "model_type": "Lasso"},
}

# -------------------------
# Plot feature importance
# -------------------------
def plot_feature_importance(model, features, out_path: Path, title: str):
    plt.figure(figsize=(7, 3.8))
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    plt.rcParams.update({
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white"
    })
    if hasattr(model, "coef_"):
        vals = pd.Series(model.coef_, index=features).sort_values()
    else:
        vals = pd.Series(model.feature_importances_, index=features).sort_values()
    vals.plot(kind="barh", color="#ff4d4d")
    plt.title(title, color="#ff4d4d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()

# -------------------------
# Train or load models
# -------------------------
def train_and_save_models(force_retrain=False):
    if MODELS_PATH.exists() and not force_retrain:
        print("Loading models from disk...")
        return joblib.load(MODELS_PATH)

    print("Training models...")
    generate_synthetic_data()
    df = add_component_scores(pd.read_csv(DATA_PATH))

    base_models = {}
    base_eval = {}

    for name, cfg in model_configs.items():
        features = model_feature_map()[name]
        X = df[features]
        y = df[cfg["target"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if cfg["model_type"] == "Lasso":
            model = Lasso(alpha=0.01, random_state=42)
            model.fit(X_train, y_train)
        elif cfg["model_type"] == "XGBoost_Grid":
            xgb = XGBRegressor(random_state=42, verbosity=0)
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            }
            gs = GridSearchCV(
                xgb, param_grid, cv=3,
                scoring=make_scorer(r2_score),
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
        else:
            model = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.07,
                subsample=0.7, random_state=42, verbosity=0
            )
            model.fit(X_train, y_train)

        base_models[name] = model
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        base_eval[name] = {"rmse": rmse, "r2": r2}

        out_img = STATIC_IMG_DIR / f"{name}_importance.png"
        plot_feature_importance(model, features, out_img, f"{name} importance")

    meta_df = pd.DataFrame(index=df.index)
    for name, m in base_models.items():
        meta_df[f"{name}_pred"] = m.predict(df[model_feature_map()[name]])

    df["final_behavioral_score"] = (
        df["digital_trust_graph_score"] * 0.25
        + df["resilience_recovery_score"] * 0.30
        + df["adaptability_score"] * 0.25
        + df["language_sentiment_score"] * 0.20
    )

    X_meta = meta_df
    y_meta = df["final_behavioral_score"]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_meta, y_meta, test_size=0.2, random_state=42
    )
    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(X_train_m, y_train_m)
    y_pred_meta = meta_model.predict(X_test_m)
    meta_eval = {
        "rmse": float(np.sqrt(mean_squared_error(y_test_m, y_pred_meta))),
        "r2": float(r2_score(y_test_m, y_pred_meta))
    }

    payload = {
        "base_models": base_models,
        "meta_model": meta_model,
        "base_eval": base_eval,
        "meta_eval": meta_eval
    }
    joblib.dump(payload, MODELS_PATH)
    print("Models saved.")
    return payload

MODELS = train_and_save_models()

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    images = sorted([p.name for p in STATIC_IMG_DIR.glob("*.png")])
    return render_template(
        "index3.html",
        images=images,
        base_eval=MODELS["base_eval"],
        meta_eval=MODELS["meta_eval"]
    )

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    mode = payload.get("mode", "full")
    user_data = payload.get("user_data", {})

    df_ref = pd.read_csv(DATA_PATH)

    # Only keep features used in models
    model_feats = sorted({f for feats in model_feature_map().values() for f in feats})
    df_ref = df_ref[model_feats]

    sample = {}
    for feat in df_ref.columns:
        val = user_data.get(feat, None)
        if val is None or str(val) == "":
            if pd.api.types.is_numeric_dtype(df_ref[feat]):
                sample[feat] = float(df_ref[feat].median())
            else:
                sample[feat] = df_ref[feat].mode()[0]
        else:
            try:
                sample[feat] = float(val)
            except ValueError:
                sample[feat] = val  # leave as string if needed

    sample_df = pd.DataFrame([sample])

    base_preds = {}
    for name, model in MODELS["base_models"].items():
        feats = model_feature_map()[name]
        pred = model.predict(sample_df[feats])[0]
        base_preds[f"{name}_pred"] = float(pred)

    if mode == "component":
        comp = payload.get("component", None)
        if comp not in MODELS["base_models"]:
            return jsonify({"error": "Invalid component requested"}), 400
        return jsonify({"component": comp, "score": base_preds[f"{comp}_pred"]})

    meta_input = pd.DataFrame([base_preds])
    final_score = float(MODELS["meta_model"].predict(meta_input)[0])

    return jsonify({
        "alticred_score": final_score,
        "base_preds": base_preds,
        "images": [url_for("static", filename=f"images/{fn}") for fn in sorted(os.listdir(STATIC_IMG_DIR))],
        "base_eval": MODELS["base_eval"],
        "meta_eval": MODELS["meta_eval"],
    })

@app.route("/static/images/<path:filename>")
def serve_img(filename):
    return send_from_directory(STATIC_IMG_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)