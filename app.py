# app.py

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, './models')

# Import your scorer classes
from alticred_salaried import AltiCredScorer as SalariedScorer

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load Models on Startup ---
print("Loading models... Please wait.")
salaried_scorer = SalariedScorer()
# self_employed_scorer = SelfEmployedScorer()
# student_scorer = StudentScorer()
print("All models loaded successfully!")

# --- Helper function for safe numeric conversion ---
def safe_float(value):
    """
    Safely converts a value to float.
    Returns 0.0 if value is None, empty, or cannot be converted.
    """
    try:
        # Handle booleans stored as strings
        if str(value).lower() in ['true', 'yes']:
            return 1.0
        elif str(value).lower() in ['false', 'no']:
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0  # Default value if invalid

# --- Define a Route for the Frontend ---
@app.route('/')
def home():
    """Renders the main user interface page."""
    return render_template('index2.html')

# --- Define a Route for Scoring ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives user data, selects a model, and returns a score."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        user_input = data.get('user_data')

        if not model_type or not user_input:
            return jsonify({'error': 'Missing model_type or user_data'}), 400

        # Fields that must be numeric
        numeric_fields = [
            'defaulter_neighbors', 'verified_neighbors', 'monthly_credit_bills',
            'bnpl_utilization_rate', 'mortgage_months_left', 'income-expense ratio',
            'owns_home', 'monthly_rent', 'recovery_days', 'sentiment_score'
        ]

        # Safely convert all numeric fields
        for field in numeric_fields:
            if field in user_input:
                user_input[field] = safe_float(user_input[field])

        # --- Select the Model and Predict ---
        if model_type == 'salaried':
            score = salaried_scorer.predict_alticred_score(user_input)
        # elif model_type == 'self_employed':
        #     score = self_employed_scorer.predict_alticred_score(user_input)
        # elif model_type == 'student':
        #     score = student_scorer.predict_alticred_score(user_input)
        else:
            return jsonify({'error': 'Invalid model type specified'}), 400

        return jsonify({'alticred_score': f"{score:.4f}"})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
